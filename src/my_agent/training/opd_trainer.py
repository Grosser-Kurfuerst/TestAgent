"""Unified four-role trainer using one shared adapter and stop-gradient teacher."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import ceil, isfinite
from pathlib import Path
from typing import Any

from my_agent.policy.contracts import ChunkedKLPolicy, EVOLVER_ROLES, TokenBatch, TrainablePolicy
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_sha256,
    hash_artifact_path,
    require_matching_policy_identity,
)
from my_agent.training.checkpoint_manifest import (
    CheckpointManifest,
    load_checkpoint_manifest,
    output_identity_for_adapter,
    write_checkpoint_manifests,
)
from my_agent.policy.identity import load_policy_identity_manifest
from my_agent.training.opd_collator import OPDCollator
from my_agent.training.opd_dataset import OPDLearnerDataset, RoleSampler
from my_agent.training.opd_loss import chunked_hidden_state_kl, gather_completion_logits


OPD_TRAIN_CONFIG_SCHEMA_VERSION = "opd-train-config-v1"


@dataclass(frozen=True)
class SharedAdapterConfig:
    name: str = "shared"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    task_type: str = "CAUSAL_LM"
    bias: str = "none"
    modules_to_save: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("shared adapter name must not be blank")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
            or isinstance(self.alpha, bool)
            or not isinstance(self.alpha, int)
            or self.alpha < 1
        ):
            raise ValueError("shared adapter name/rank/alpha are invalid")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("shared adapter dropout must be in [0, 1)")
        payload = canonical_adapter_payload(self)
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "target_modules", tuple(payload["target_modules"]))
        object.__setattr__(self, "task_type", str(payload["task_type"]))
        object.__setattr__(self, "bias", str(payload["bias"]))
        object.__setattr__(self, "modules_to_save", tuple(payload["modules_to_save"]))

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return canonical_adapter_payload(self)

    @property
    def adapter_config_hash(self) -> str:
        return canonical_sha256(self.canonical_payload)


@dataclass(frozen=True)
class OPDTrainerConfig:
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_gradient_norm: float = 1.0
    vocab_chunk_size: int = 4_096
    mixed_precision: str = "no"
    seed: int = 0
    samples_per_epoch: int | None = None
    role_sampling_weights: Mapping[str, float] = field(default_factory=lambda: {
        role: 1.0 for role in sorted(EVOLVER_ROLES)
    })
    shared_adapter: SharedAdapterConfig = field(default_factory=SharedAdapterConfig)
    schema_version: str = OPD_TRAIN_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPD_TRAIN_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported OPD trainer config schema")
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, batch_size, and gradient accumulation must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer configuration is invalid")
        if self.warmup_steps < 0 or self.max_gradient_norm <= 0.0:
            raise ValueError("scheduler/gradient configuration is invalid")
        if self.vocab_chunk_size < 1:
            raise ValueError("vocab_chunk_size must be positive")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
        if set(self.role_sampling_weights) != EVOLVER_ROLES:
            raise ValueError("formal OPD role weights must cover all four roles")
        if any(float(value) <= 0.0 for value in self.role_sampling_weights.values()):
            raise ValueError("role sampling weights must be positive")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OPDTrainerConfig":
        adapter_data = data.get("shared_adapter", {})
        if not isinstance(adapter_data, Mapping):
            raise ValueError("shared_adapter config must be an object")
        role_weights = data.get("role_sampling_weights", {
            role: 1.0 for role in sorted(EVOLVER_ROLES)
        })
        if not isinstance(role_weights, Mapping):
            raise ValueError("role_sampling_weights must be an object")
        samples_per_epoch = data.get("samples_per_epoch")
        shared_adapter = SharedAdapterConfig(
            name=str(adapter_data.get("name", "shared")),
            rank=int(adapter_data.get("rank", 16)),
            alpha=int(adapter_data.get("alpha", 32)),
            dropout=float(adapter_data.get("dropout", 0.0)),
            target_modules=_optional_string_tuple(
                adapter_data.get(
                    "target_modules", ("q_proj", "k_proj", "v_proj", "o_proj")
                ),
                field_name="shared_adapter.target_modules",
            ),
            task_type=str(adapter_data.get("task_type", "CAUSAL_LM")),
            bias=str(adapter_data.get("bias", "none")),
            modules_to_save=_optional_string_tuple(
                adapter_data.get("modules_to_save"),
                field_name="shared_adapter.modules_to_save",
            ),
        )
        configured_adapter_hash = adapter_data.get("adapter_config_hash")
        if (
            configured_adapter_hash is not None
            and configured_adapter_hash != shared_adapter.adapter_config_hash
        ):
            raise ValueError("shared_adapter adapter_config_hash does not match its config")
        return cls(
            schema_version=str(data.get("schema_version", OPD_TRAIN_CONFIG_SCHEMA_VERSION)),
            epochs=int(data.get("epochs", 1)),
            batch_size=int(data.get("batch_size", 1)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 1)),
            learning_rate=float(data.get("learning_rate", 2e-4)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            warmup_steps=int(data.get("warmup_steps", 0)),
            max_gradient_norm=float(data.get("max_gradient_norm", 1.0)),
            vocab_chunk_size=int(data.get("vocab_chunk_size", 4_096)),
            mixed_precision=str(data.get("mixed_precision", "no")),
            seed=int(data.get("seed", 0)),
            samples_per_epoch=(None if samples_per_epoch is None else int(samples_per_epoch)),
            role_sampling_weights={str(key): float(value) for key, value in role_weights.items()},
            shared_adapter=shared_adapter,
        )


@dataclass(frozen=True)
class OPDTrainResult:
    checkpoint_dir: Path
    checkpoint_manifest_path: Path
    identity_manifest_path: Path
    output_identity: PolicyIdentity
    manifest: CheckpointManifest


class OPDTrainer:
    def __init__(
        self,
        *,
        policy: TrainablePolicy,
        dataset: OPDLearnerDataset,
        validation_dataset: OPDLearnerDataset | None = None,
        config: OPDTrainerConfig,
        torch_module: Any | None = None,
        accelerator: Any | None = None,
    ) -> None:
        require_matching_policy_identity(dataset.initialization_identity, policy.identity())
        if not isinstance(policy, ChunkedKLPolicy):
            raise ValueError("OPD trainer requires hidden-state chunked-KL policy support")
        if validation_dataset is not None:
            require_matching_policy_identity(
                dataset.initialization_identity,
                validation_dataset.initialization_identity,
            )
            if validation_dataset.collection_round != dataset.collection_round:
                raise ValueError("validation dataset crosses collection rounds")
            if (
                validation_dataset.ablation != dataset.ablation
                or validation_dataset.ablation_recipe_hash != dataset.ablation_recipe_hash
            ):
                raise ValueError("validation dataset uses a different ablation recipe")
        self.policy = policy
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.config = config
        self.torch = torch_module if torch_module is not None else _load_torch()
        self.accelerator = (
            accelerator
            if accelerator is not None
            else build_training_accelerator(config)
        )

    def train(
        self,
        checkpoint_dir: str | Path,
        *,
        reload_identity_verifier: Callable[[Path, PolicyIdentity], bool],
    ) -> OPDTrainResult:
        policy_model = getattr(self.policy, "model", None)
        if policy_model is None:
            raise ValueError("OPD trainer requires a policy exposing its shared model")
        adapter_name = validate_shared_adapter_config(
            policy_model,
            self.config.shared_adapter,
        )
        model = _hidden_state_training_model(self.policy, policy_model, self.torch)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("shared adapter exposes no trainable parameters")
        optimizer = self.torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        active_role_weights = {
            role: self.config.role_sampling_weights[role]
            for role in self.dataset.statistics.role_counts
        }
        sampler = RoleSampler(
            self.dataset,
            role_weights=active_role_weights,
            num_samples=self.config.samples_per_epoch,
            seed=self.config.seed,
        )
        collator = OPDCollator(self.policy, torch_module=self.torch)
        data_loader = self.torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=collator,
        )
        validation_loader = (
            self.torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            if self.validation_dataset is not None
            else None
        )
        scheduler_total_steps = ceil(
            len(data_loader) / self.config.gradient_accumulation_steps
        ) * self.config.epochs
        scheduler = _build_scheduler(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=scheduler_total_steps,
        )
        prepared = self.accelerator.prepare(*(
            (model, optimizer, data_loader, scheduler, validation_loader)
            if validation_loader is not None
            else (model, optimizer, data_loader, scheduler)
        ))
        if validation_loader is not None:
            model, optimizer, data_loader, scheduler, validation_loader = prepared
        else:
            model, optimizer, data_loader, scheduler = prepared
        optimizer_steps = ceil(
            len(data_loader) / self.config.gradient_accumulation_steps
        ) * self.config.epochs
        model.train()
        unwrapped_training_model = self.accelerator.unwrap_model(model)
        output_weight, output_bias = self.policy.output_projection(
            model=unwrapped_training_model.policy_model,
        )
        if output_weight.requires_grad or (
            output_bias is not None and output_bias.requires_grad
        ):
            raise ValueError("formal OPD shared adapter must keep the LM output head frozen")
        optimizer.zero_grad()
        role_kl_sums: defaultdict[str, float] = defaultdict(float)
        role_token_counts: Counter[str] = Counter()
        task_group_counts: Counter[str] = Counter()
        sampled_role_counts: Counter[str] = Counter()
        gradient_norms: list[float] = []
        role_gradient_norms: defaultdict[str, list[float]] = defaultdict(list)
        accumulated_roles: set[str] = set()

        for epoch in range(self.config.epochs):
            sampler.set_epoch(epoch)
            for batch in data_loader:
                with self.accelerator.accumulate(model):
                    accumulated_roles.update(batch["roles"])
                    teacher_hidden, student_hidden = self._dual_forward(batch, model)
                    teacher_completion = gather_completion_logits(
                        teacher_hidden,
                        batch["teacher_prediction_indices"],
                        batch["completion_mask"],
                        torch_module=self.torch,
                    )
                    student_completion = gather_completion_logits(
                        student_hidden,
                        batch["student_prediction_indices"],
                        batch["completion_mask"],
                        torch_module=self.torch,
                    )
                    loss_output = chunked_hidden_state_kl(
                        teacher_completion,
                        student_completion,
                        output_weight,
                        output_bias,
                        batch["completion_mask"],
                        vocab_chunk_size=self.config.vocab_chunk_size,
                        torch_module=self.torch,
                    )
                    self.accelerator.backward(loss_output.loss)
                    if self.accelerator.sync_gradients:
                        norm = self.accelerator.clip_grad_norm_(
                            trainable_parameters,
                            self.config.max_gradient_norm,
                        )
                        norm_value = float(norm.detach().float().item())
                        gradient_norms.append(norm_value)
                        for role in accumulated_roles:
                            role_gradient_norms[role].append(norm_value)
                        accumulated_roles.clear()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                _accumulate_batch_metrics(
                    loss_output.per_token_kl,
                    batch["completion_mask"],
                    batch["roles"],
                    batch["task_groups"],
                    role_kl_sums,
                    role_token_counts,
                    task_group_counts,
                    sampled_role_counts,
                )

        validation_role_kl, validation_role_tokens = self._evaluate(
            validation_loader,
            model,
            output_weight,
            output_bias,
        )
        (
            role_kl_sums,
            role_token_counts,
            task_group_counts,
            sampled_role_counts,
            gradient_norms,
            role_gradient_norms,
            validation_role_kl,
            validation_role_tokens,
        ) = _merge_distributed_metrics(
            accelerator=self.accelerator,
            torch=self.torch,
            role_kl_sums=role_kl_sums,
            role_token_counts=role_token_counts,
            task_group_counts=task_group_counts,
            sampled_role_counts=sampled_role_counts,
            gradient_norms=gradient_norms,
            role_gradient_norms=role_gradient_norms,
            validation_role_kl=validation_role_kl,
            validation_role_tokens=validation_role_tokens,
        )
        self.accelerator.wait_for_everyone()
        root = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_path = root / "opd_checkpoint_manifest.json"
        identity_path = root / "policy_identity_manifest.json"
        output_identity: PolicyIdentity | None = None
        manifest: CheckpointManifest | None = None
        if self.accelerator.is_main_process:
            root.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(model)
            _save_pretrained(unwrapped.policy_model, root)
            tokenizer = getattr(self.policy, "tokenizer", None)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(root)
            optimizer_path = root / "optimizer.pt"
            scheduler_path = root / "scheduler.pt"
            self.accelerator.save(optimizer.state_dict(), optimizer_path)
            self.accelerator.save(scheduler.state_dict(), scheduler_path)
            output_identity = output_identity_for_adapter(
                self.dataset.initialization_identity,
                root,
            )
            state_artifacts = {
                "optimizer": hash_artifact_path(optimizer_path),
                "scheduler": hash_artifact_path(scheduler_path),
            }
            role_kl = {
                role: role_kl_sums[role] / role_token_counts[role]
                for role in sorted(role_token_counts)
            }
            manifest = CheckpointManifest(
                collection_round=self.dataset.collection_round,
                initialization_identity=self.dataset.initialization_identity,
                output_identity=output_identity,
                learner_dataset_hash=self.dataset.learner_dataset_hash,
                export_manifest_hash=_required_export_manifest_hash(self.dataset),
                role_sampling_weights=active_role_weights,
                raw_role_counts=self.dataset.statistics.role_counts,
                valid_role_counts=self.dataset.statistics.role_counts,
                sampled_role_counts=dict(sampled_role_counts),
                task_group_counts=dict(task_group_counts),
                optimizer={
                    "name": "AdamW",
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                },
                scheduler={
                    "name": "linear",
                    "warmup_steps": self.config.warmup_steps,
                    "optimizer_steps": optimizer_steps,
                    "underlying_scheduler_steps": scheduler_total_steps,
                },
                state_artifacts=state_artifacts,
                train_role_kl=role_kl,
                train_role_tokens=dict(role_token_counts),
                validation_role_kl=validation_role_kl,
                validation_role_tokens=validation_role_tokens,
                gradient_norm={
                    "mean": sum(gradient_norms) / len(gradient_norms),
                    "max": max(gradient_norms),
                },
                mixed_step_gradient_norm_by_role={
                    role: sum(values) / len(values)
                    for role, values in sorted(role_gradient_norms.items())
                },
                shared_adapter_name=adapter_name,
                shared_adapter_config=self.config.shared_adapter.canonical_payload,
                adapter_config_hash=self.config.shared_adapter.adapter_config_hash,
                reload_identity_verified=False,
                ablation=self.dataset.ablation,
                ablation_recipe_hash=self.dataset.ablation_recipe_hash,
                dataset_source_hashes=self.dataset.dataset_source_hashes,
            )
            write_checkpoint_manifests(root, manifest)
        self.accelerator.wait_for_everyone()
        setattr(self.policy, "model", None)
        del model, policy_model, unwrapped_training_model, output_weight, output_bias
        del trainable_parameters, optimizer, data_loader, scheduler
        self.accelerator.free_memory()
        if self.accelerator.is_main_process:
            assert output_identity is not None and manifest is not None
            verified = bool(reload_identity_verifier(root, output_identity))
            if not verified:
                raise ValueError("saved Transformers policy identity reload verification failed")
            manifest = CheckpointManifest(
                **{
                    **manifest.__dict__,
                    "reload_identity_verified": True,
                }
            )
            checkpoint_path, identity_path = write_checkpoint_manifests(root, manifest)
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            output_identity = load_policy_identity_manifest(identity_path)
            manifest = load_checkpoint_manifest(checkpoint_path)
        assert output_identity is not None and manifest is not None
        return OPDTrainResult(root, checkpoint_path, identity_path, output_identity, manifest)

    def _dual_forward(self, batch: Mapping[str, Any], model: Any) -> tuple[Any, Any]:
        """Use the same current model for both branches; only teacher is stop-gradient."""

        with self.torch.no_grad():
            teacher_hidden = model(
                input_ids=batch["teacher_input_ids"],
                attention_mask=batch["teacher_attention_mask"],
                assistant_loss_mask=batch["teacher_assistant_loss_mask"],
            )
        student_hidden = model(
            input_ids=batch["student_input_ids"],
            attention_mask=batch["student_attention_mask"],
            assistant_loss_mask=batch["student_assistant_loss_mask"],
        )
        return teacher_hidden, student_hidden

    def _evaluate(
        self,
        data_loader: Any | None,
        model: Any,
        output_weight: Any,
        output_bias: Any | None,
    ) -> tuple[dict[str, float], dict[str, int]]:
        if data_loader is None:
            return {}, {}
        role_kl_sums: defaultdict[str, float] = defaultdict(float)
        role_token_counts: Counter[str] = Counter()
        task_groups: Counter[str] = Counter()
        role_samples: Counter[str] = Counter()
        with self.torch.no_grad():
            for batch in data_loader:
                teacher_hidden, student_hidden = self._dual_forward(batch, model)
                teacher_completion = gather_completion_logits(
                    teacher_hidden,
                    batch["teacher_prediction_indices"],
                    batch["completion_mask"],
                    torch_module=self.torch,
                )
                student_completion = gather_completion_logits(
                    student_hidden,
                    batch["student_prediction_indices"],
                    batch["completion_mask"],
                    torch_module=self.torch,
                )
                output = chunked_hidden_state_kl(
                    teacher_completion,
                    student_completion,
                    output_weight,
                    output_bias,
                    batch["completion_mask"],
                    vocab_chunk_size=self.config.vocab_chunk_size,
                    torch_module=self.torch,
                )
                _accumulate_batch_metrics(
                    output.per_token_kl,
                    batch["completion_mask"],
                    batch["roles"],
                    batch["task_groups"],
                    role_kl_sums,
                    role_token_counts,
                    task_groups,
                    role_samples,
                )
        return (
            {
                role: role_kl_sums[role] / role_token_counts[role]
                for role in sorted(role_token_counts)
            },
            dict(role_token_counts),
        )


def attach_or_validate_shared_adapter(
    model: Any,
    config: SharedAdapterConfig,
) -> Any:
    peft_config = getattr(model, "peft_config", None)
    if peft_config:
        validate_shared_adapter_config(model, config)
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("shared OPD adapter requires the 'opd-train' extra") from exc
    adapter = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        task_type=config.task_type,
        bias=config.bias,
        modules_to_save=None,
    )
    model = get_peft_model(model, adapter, adapter_name=config.name)
    validate_shared_adapter_config(model, config)
    return model


def canonical_adapter_payload(config: Any) -> dict[str, Any]:
    rank = _adapter_value(config, "rank", "r")
    alpha = _adapter_value(config, "alpha", "lora_alpha")
    dropout = _adapter_value(config, "dropout", "lora_dropout")
    target_modules = _adapter_value(config, "target_modules")
    task_type = _adapter_value(config, "task_type")
    bias = _adapter_value(config, "bias")
    modules_to_save = _adapter_value(config, "modules_to_save", default=None)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("adapter rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha < 1:
        raise ValueError("adapter alpha must be a positive integer")
    if isinstance(dropout, bool):
        raise ValueError("adapter dropout must be a finite float")
    dropout_value = float(dropout)
    if not isfinite(dropout_value) or not 0.0 <= dropout_value < 1.0:
        raise ValueError("adapter dropout must be in [0, 1)")
    normalized_targets = _normalize_target_modules(target_modules)
    normalized_task_type = _normalize_task_type(task_type)
    normalized_bias = str(bias).strip().lower()
    if normalized_task_type != "CAUSAL_LM":
        raise ValueError("formal shared adapter task_type must be CAUSAL_LM")
    if normalized_bias != "none":
        raise ValueError("formal shared adapter bias must be none")
    normalized_modules_to_save = _normalize_modules_to_save(modules_to_save)
    if normalized_modules_to_save:
        raise ValueError("formal shared adapter modules_to_save must be empty")
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout_value,
        "target_modules": list(normalized_targets),
        "task_type": normalized_task_type,
        "bias": normalized_bias,
        "modules_to_save": list(normalized_modules_to_save),
    }


def validate_shared_adapter_config(model: Any, expected: SharedAdapterConfig) -> str:
    adapter_name = require_one_shared_adapter(model)
    actual = model.peft_config[adapter_name]
    actual_payload = canonical_adapter_payload(actual)
    expected_payload = expected.canonical_payload
    if actual_payload != expected_payload:
        raise ValueError(
            "shared adapter config mismatch: "
            f"expected={expected_payload}, actual={actual_payload}"
        )
    return adapter_name


def require_one_shared_adapter(model: Any) -> str:
    configurations = getattr(model, "peft_config", None)
    if not isinstance(configurations, Mapping) or len(configurations) != 1:
        raise ValueError("formal OPD training requires exactly one shared adapter")
    return str(next(iter(configurations)))


def _adapter_value(config: Any, *names: str, default: Any = ...) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if default is not ...:
        return default
    raise ValueError(f"adapter config is missing {names[0]}")


def _normalize_target_modules(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("formal adapter target_modules must be an explicit collection")
    normalized = tuple(sorted({str(item).strip() for item in value}))
    if not normalized or any(not item for item in normalized):
        raise ValueError("shared adapter target_modules must not be empty")
    if any(item.lower() == "all" or any(char in item for char in "*?[]") for item in normalized):
        raise ValueError("formal adapter target_modules cannot use all, regex, or wildcards")
    return normalized


def _normalize_task_type(value: Any) -> str:
    enum_value = getattr(value, "value", value)
    normalized = str(enum_value).strip().upper()
    if normalized.startswith("TASKTYPE."):
        normalized = normalized.split(".", 1)[1]
    return normalized


def _normalize_modules_to_save(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("modules_to_save must be an explicit collection or null")
    return tuple(sorted({str(item).strip() for item in value if str(item).strip()}))


def _optional_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{field_name} must be an array or null")
    return tuple(str(item) for item in value)


def _hidden_state_training_model(
    policy: ChunkedKLPolicy,
    policy_model: Any,
    torch: Any,
) -> Any:
    class HiddenStateTrainingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_model = policy_model

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            assistant_loss_mask: Any,
        ) -> Any:
            return policy.forward_hidden_states(TokenBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                assistant_loss_mask=assistant_loss_mask,
            ), model=self.policy_model)

    return HiddenStateTrainingModel()


def _accumulate_batch_metrics(
    per_token_kl: Any,
    completion_mask: Any,
    roles: tuple[str, ...],
    task_groups: tuple[str, ...],
    role_kl_sums: defaultdict[str, float],
    role_token_counts: Counter[str],
    task_group_counts: Counter[str],
    sampled_role_counts: Counter[str],
) -> None:
    values = per_token_kl.detach().float().cpu()
    masks = completion_mask.detach().float().cpu()
    for index, role in enumerate(roles):
        active = masks[index]
        role_kl_sums[role] += float((values[index] * active).sum().item())
        role_token_counts[role] += int(active.sum().item())
        sampled_role_counts[role] += 1
        task_group_counts[task_groups[index]] += 1


def _required_export_manifest_hash(dataset: OPDLearnerDataset) -> str:
    value = dataset.export_manifest_hash
    if value is None:
        raise ValueError("OPD trainer requires an export manifest hash")
    return value


def _merge_distributed_metrics(
    *,
    accelerator: Any,
    torch: Any,
    role_kl_sums: Mapping[str, float],
    role_token_counts: Mapping[str, int],
    task_group_counts: Mapping[str, int],
    sampled_role_counts: Mapping[str, int],
    gradient_norms: list[float],
    role_gradient_norms: Mapping[str, list[float]],
    validation_role_kl: Mapping[str, float],
    validation_role_tokens: Mapping[str, int],
) -> tuple[
    defaultdict[str, float],
    Counter[str],
    Counter[str],
    Counter[str],
    list[float],
    defaultdict[str, list[float]],
    dict[str, float],
    dict[str, int],
]:
    payload = {
        "role_kl_sums": dict(role_kl_sums),
        "role_token_counts": dict(role_token_counts),
        "task_group_counts": dict(task_group_counts),
        "sampled_role_counts": dict(sampled_role_counts),
        "gradient_norms": list(gradient_norms),
        "role_gradient_norms": {
            role: list(values) for role, values in role_gradient_norms.items()
        },
        "validation_kl_sums": {
            role: validation_role_kl[role] * validation_role_tokens[role]
            for role in validation_role_tokens
        },
        "validation_role_tokens": dict(validation_role_tokens),
    }
    payloads = [payload]
    if int(getattr(accelerator, "num_processes", 1)) > 1:
        payloads = [None] * int(accelerator.num_processes)
        torch.distributed.all_gather_object(payloads, payload)
    merged_kl: defaultdict[str, float] = defaultdict(float)
    merged_tokens: Counter[str] = Counter()
    merged_groups: Counter[str] = Counter()
    merged_samples: Counter[str] = Counter()
    merged_norms: list[float] = []
    merged_role_norms: defaultdict[str, list[float]] = defaultdict(list)
    validation_sums: defaultdict[str, float] = defaultdict(float)
    validation_tokens: Counter[str] = Counter()
    for item in payloads:
        if not isinstance(item, Mapping):
            raise RuntimeError("distributed OPD metric payload is invalid")
        for role, value in item["role_kl_sums"].items():
            merged_kl[role] += float(value)
        merged_tokens.update(item["role_token_counts"])
        merged_groups.update(item["task_group_counts"])
        merged_samples.update(item["sampled_role_counts"])
        merged_norms.extend(float(value) for value in item["gradient_norms"])
        for role, values in item["role_gradient_norms"].items():
            merged_role_norms[role].extend(float(value) for value in values)
        for role, value in item["validation_kl_sums"].items():
            validation_sums[role] += float(value)
        validation_tokens.update(item["validation_role_tokens"])
    validation_averages = {
        role: validation_sums[role] / validation_tokens[role]
        for role in sorted(validation_tokens)
    }
    return (
        merged_kl,
        merged_tokens,
        merged_groups,
        merged_samples,
        merged_norms,
        merged_role_norms,
        validation_averages,
        dict(validation_tokens),
    )


def _save_pretrained(model: Any, path: Path) -> None:
    if not hasattr(model, "save_pretrained"):
        raise ValueError("shared adapter model does not support save_pretrained")
    try:
        model.save_pretrained(path, safe_serialization=True)
    except TypeError:
        model.save_pretrained(path)


def build_training_accelerator(
    config: OPDTrainerConfig,
    *,
    cpu: bool = False,
) -> Any:
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError("multi-device OPD training requires the 'opd-train' extra") from exc
    return Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        cpu=cpu,
    )


def _build_scheduler(optimizer: Any, *, warmup_steps: int, total_steps: int) -> Any:
    try:
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError("OPD scheduler requires the 'opd-train' extra") from exc
    return get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("OPD training requires the 'opd-train' extra") from exc
    return torch


__all__ = [
    "OPD_TRAIN_CONFIG_SCHEMA_VERSION",
    "OPDTrainResult",
    "OPDTrainer",
    "OPDTrainerConfig",
    "SharedAdapterConfig",
    "attach_or_validate_shared_adapter",
    "canonical_adapter_payload",
    "build_training_accelerator",
    "require_one_shared_adapter",
    "validate_shared_adapter_config",
]
