from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import tempfile
import unittest

from my_agent.training.collection_round import build_collection_round
from my_agent.opd_data.attribution import build_round_attribution
from my_agent.training.opd_dataset import OPDLearnerDataset
from my_agent.training.opd_trainer import (
    OPDTrainer,
    OPDTrainerConfig,
    SharedAdapterConfig,
    _merge_distributed_metrics,
)
from my_agent.training.checkpoint_manifest import output_identity_for_adapter
from tests.training.opd_round_fixtures import FakeTrainablePolicy, round_fixture


class _TinyAdapterModel:
    def __init__(self, torch) -> None:
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 12)
                self.output = torch.nn.Linear(12, 256, bias=False)
                self.adapter = torch.nn.Linear(12, 12, bias=False)
                self.embedding.weight.requires_grad_(False)
                self.output.weight.requires_grad_(False)
                adapter = SharedAdapterConfig()
                self.peft_config = {"shared": SimpleNamespace(
                    r=adapter.rank,
                    lora_alpha=adapter.alpha,
                    lora_dropout=adapter.dropout,
                    target_modules=set(adapter.target_modules),
                    task_type=adapter.task_type,
                    bias=adapter.bias,
                    modules_to_save=None,
                )}
                self.forward_grad_enabled: list[bool] = []

            def hidden_states(self, *, input_ids, attention_mask):
                del attention_mask
                self.forward_grad_enabled.append(torch.is_grad_enabled())
                base = self.embedding(input_ids).cumsum(dim=1)
                return base + self.adapter(base)

            def forward(self, *, input_ids, attention_mask):
                hidden = self.hidden_states(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                return SimpleNamespace(logits=self.output(hidden))

            def save_pretrained(self, path, safe_serialization=True):
                del safe_serialization
                root = Path(path)
                root.mkdir(parents=True, exist_ok=True)
                (root / "adapter_config.json").write_text(
                    '{"adapter_name":"shared"}\n',
                    encoding="utf-8",
                )
                torch.save(self.state_dict(), root / "adapter_model.bin")

        self.module = Module()


class _TinyTokenizer:
    pad_token_id = 0

    def save_pretrained(self, path) -> None:
        Path(path, "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


class _TinyTrainingPolicy(FakeTrainablePolicy):
    def __init__(self, torch) -> None:
        super().__init__()
        self.model = _TinyAdapterModel(torch).module
        self.tokenizer = _TinyTokenizer()

    def tokenize(self, request):
        from my_agent.policy.contracts import TokenBatch

        prefix = 20 if len(request.messages) > 2 else 10
        ids = ((prefix, _role_token(request.role)),)
        return TokenBatch(ids, ((1, 1),), ((0, 0),))

    def forward_logits(self, batch):
        raise AssertionError("trainer must not materialize full-vocabulary logits")

    def forward_hidden_states(self, batch, *, model=None):
        active = self.model if model is None else model
        return active.hidden_states(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )

    def output_projection(self, *, model=None):
        active = self.model if model is None else model
        return active.output.weight, active.output.bias


class OnPolicyIdentityTests(unittest.TestCase):
    def test_smoke_round_trains_all_roles_into_one_new_identity(self) -> None:
        import torch
        from accelerate import Accelerator

        fixture = round_fixture()
        policy = _TinyTrainingPolicy(torch)
        original_model = policy.model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attribution_result = build_round_attribution(
                collection_round=0,
                tasks=fixture.tasks,
                outcomes=fixture.outcomes,
                repositories=fixture.repositories,
                output_dir=root / "attribution",
            )
            round_result = build_collection_round(
                collection_round=0,
                policy=policy,
                tasks=fixture.tasks,
                outcomes=fixture.outcomes,
                repositories=fixture.repositories,
                maintenance=fixture.maintenance,
                decision_events=fixture.decisions,
                attribution=attribution_result.records,
                output_dir=root / "d0",
                seed=5,
            )
            dataset = OPDLearnerDataset.from_files(
                round_result.learner_path,
                round_result.manifest_path,
            )
            trainer = OPDTrainer(
                policy=policy,
                dataset=dataset,
                config=OPDTrainerConfig(
                    batch_size=2,
                    vocab_chunk_size=31,
                    seed=9,
                ),
                torch_module=torch,
                accelerator=Accelerator(cpu=True, mixed_precision="no"),
            )

            def reload_identity(path, expected):
                reloaded = _TinyTrainingPolicy(torch)
                reloaded.model.load_state_dict(torch.load(
                    path / "adapter_model.bin",
                    map_location="cpu",
                    weights_only=True,
                ))
                sample = dataset[0]
                request_batch = reloaded.tokenize(_request_for_sample(sample))
                input_ids = torch.tensor(request_batch.input_ids, dtype=torch.long)
                attention_mask = torch.tensor(request_batch.attention_mask, dtype=torch.long)
                logits = reloaded.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                return (
                    tuple(logits.shape[:2]) == tuple(input_ids.shape)
                    and output_identity_for_adapter(dataset.initialization_identity, path)
                    == expected
                )

            result = trainer.train(
                root / "m1",
                reload_identity_verifier=reload_identity,
            )

        self.assertNotEqual(
            result.output_identity.identity_hash,
            dataset.initialization_identity.identity_hash,
        )
        self.assertIsNotNone(result.output_identity.adapter_hash)
        self.assertEqual(set(result.manifest.train_role_kl), {
            "selection", "action", "writing", "maintenance",
        })
        self.assertEqual(result.manifest.shared_adapter_name, "shared")
        self.assertTrue(result.manifest.reload_identity_verified)
        self.assertIn(False, original_model.forward_grad_enabled)
        self.assertIn(True, original_model.forward_grad_enabled)

    def test_multi_process_metrics_are_merged_before_manifesting(self) -> None:
        remote = {
            "role_kl_sums": {"action": 2.0},
            "role_token_counts": {"action": 4},
            "task_group_counts": {"group-b": 2},
            "sampled_role_counts": {"action": 2},
            "gradient_norms": [3.0],
            "role_gradient_norms": {"action": [3.0]},
            "validation_kl_sums": {"action": 1.0},
            "validation_role_tokens": {"action": 2},
        }

        class Distributed:
            @staticmethod
            def all_gather_object(outputs, local):
                outputs[0] = local
                outputs[1] = remote

        fake_torch = SimpleNamespace(distributed=Distributed())
        accelerator = SimpleNamespace(num_processes=2)
        merged = _merge_distributed_metrics(
            accelerator=accelerator,
            torch=fake_torch,
            role_kl_sums={"selection": 1.0},
            role_token_counts={"selection": 2},
            task_group_counts={"group-a": 1},
            sampled_role_counts={"selection": 1},
            gradient_norms=[1.0],
            role_gradient_norms={"selection": [1.0]},
            validation_role_kl={"selection": 0.25},
            validation_role_tokens={"selection": 2},
        )

        self.assertEqual(merged[1], Counter({"action": 4, "selection": 2}))
        self.assertEqual(merged[2], Counter({"group-b": 2, "group-a": 1}))
        self.assertEqual(merged[6], {"action": 0.5, "selection": 0.25})


def _role_token(role: str) -> int:
    return {"selection": 1, "action": 2, "writing": 3, "maintenance": 4}[role]


def _request_for_sample(sample):
    from my_agent.policy.contracts import DecisionRequest

    return DecisionRequest(
        role=sample.role,
        purpose="opd_learner",
        messages=sample.canonical_student_messages,
        tools=sample.canonical_tools,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
    )


if __name__ == "__main__":
    unittest.main()
