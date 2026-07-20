from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import yaml

from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_sha256,
    load_policy_identity_manifest,
)
from my_agent.policy.transformers_policy import hash_adapter_artifacts
from my_agent.training.sft_registration import register_sft_checkpoint


BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "revision-1"


def _adapter_config(**overrides) -> dict:
    values = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "modules_to_save": None,
        "base_model_name_or_path": BASE_MODEL,
    }
    values.update(overrides)
    return values


def _opd_config(path: Path) -> Path:
    path.write_text(yaml.safe_dump({
        "trainer": {
            "shared_adapter": {
                "name": "shared",
                "rank": 16,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "CAUSAL_LM",
                "bias": "none",
                "modules_to_save": [],
            }
        }
    }), encoding="utf-8")
    return path


def _trainer_output(path: Path, **config_overrides) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(_adapter_config(**config_overrides)), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    checkpoint = path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(_adapter_config(r=8)), encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"old adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    adapter = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "modules_to_save": [],
    }
    manifest = {
        "schema_version": "agentcli-legacy-sft-training-v1",
        "base_model": BASE_MODEL,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "template": "qwen3_nothink",
        "llamafactory_version": "0.9.4",
        "adapter_config": adapter,
        "adapter_config_hash": canonical_sha256(adapter),
    }
    (path / "sft_training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def _policy_loader(config):
    adapter_hash = hash_adapter_artifacts(config.policy_adapter_path)
    identity = PolicyIdentity(
        base_model=config.policy_base_model,
        base_revision=config.policy_base_revision,
        checkpoint_hash="sha256:" + "1" * 64,
        adapter_hash=adapter_hash,
        tokenizer_revision=config.policy_tokenizer_revision,
        tokenizer_hash="sha256:" + "2" * 64,
        chat_template_hash=canonical_sha256("template"),
    )
    peft_config = SimpleNamespace(**_adapter_config())
    return SimpleNamespace(
        model=SimpleNamespace(peft_config={"default": peft_config}),
        identity=lambda: identity,
    )


def test_registers_clean_m0_adapter_and_identity(tmp_path: Path) -> None:
    trainer = _trainer_output(tmp_path / "trainer")
    output = tmp_path / "M0"

    result = register_sft_checkpoint(
        trainer_output=trainer,
        output=output,
        base_model=BASE_MODEL,
        base_revision=REVISION,
        tokenizer_revision=REVISION,
        opd_config=_opd_config(tmp_path / "opd.yaml"),
        policy_loader=_policy_loader,
    )

    assert sorted(path.name for path in result.adapter_dir.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert load_policy_identity_manifest(result.identity_manifest_path) == result.identity
    assert result.training_manifest_path == output / "sft_training_manifest.json"
    assert result.training_manifest_path.is_file()
    assert result.identity.adapter_hash == hash_adapter_artifacts(result.adapter_dir)
    assert not (result.adapter_dir / "trainer_state.json").exists()
    assert not (result.adapter_dir / "checkpoint-10").exists()


def test_rejects_adapter_that_does_not_match_opd(tmp_path: Path) -> None:
    trainer = _trainer_output(tmp_path / "trainer", r=8)

    with pytest.raises(ValueError, match="does not match OPD"):
        register_sft_checkpoint(
            trainer_output=trainer,
            output=tmp_path / "M0",
            base_model=BASE_MODEL,
            base_revision=REVISION,
            tokenizer_revision=REVISION,
            opd_config=_opd_config(tmp_path / "opd.yaml"),
            policy_loader=_policy_loader,
        )


def test_requires_final_adapter_at_trainer_output_root(tmp_path: Path) -> None:
    trainer = tmp_path / "trainer"
    checkpoint = trainer / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")

    with pytest.raises(FileNotFoundError, match="root must contain"):
        register_sft_checkpoint(
            trainer_output=trainer,
            output=tmp_path / "M0",
            base_model=BASE_MODEL,
            base_revision=REVISION,
            tokenizer_revision=REVISION,
            opd_config=_opd_config(tmp_path / "opd.yaml"),
            policy_loader=_policy_loader,
        )


def test_rejects_nonempty_m0_output(tmp_path: Path) -> None:
    trainer = _trainer_output(tmp_path / "trainer")
    output = tmp_path / "M0"
    output.mkdir()
    (output / "old.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        register_sft_checkpoint(
            trainer_output=trainer,
            output=output,
            base_model=BASE_MODEL,
            base_revision=REVISION,
            tokenizer_revision=REVISION,
            opd_config=_opd_config(tmp_path / "opd.yaml"),
            policy_loader=_policy_loader,
        )


def test_rejects_training_manifest_identity_mismatch(tmp_path: Path) -> None:
    trainer = _trainer_output(tmp_path / "trainer")
    manifest_path = trainer / "sft_training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_model"] = "Other/Model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="base model"):
        register_sft_checkpoint(
            trainer_output=trainer,
            output=tmp_path / "M0",
            base_model=BASE_MODEL,
            base_revision=REVISION,
            tokenizer_revision=REVISION,
            opd_config=_opd_config(tmp_path / "opd.yaml"),
            policy_loader=_policy_loader,
        )


def test_requires_training_manifest(tmp_path: Path) -> None:
    trainer = _trainer_output(tmp_path / "trainer")
    (trainer / "sft_training_manifest.json").unlink()

    with pytest.raises(FileNotFoundError):
        register_sft_checkpoint(
            trainer_output=trainer,
            output=tmp_path / "M0",
            base_model=BASE_MODEL,
            base_revision=REVISION,
            tokenizer_revision=REVISION,
            opd_config=_opd_config(tmp_path / "opd.yaml"),
            policy_loader=_policy_loader,
        )


def test_rejects_adapter_base_model_mismatch(tmp_path: Path) -> None:
    trainer = _trainer_output(
        tmp_path / "trainer", base_model_name_or_path="Other/Model"
    )

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        register_sft_checkpoint(
            trainer_output=trainer,
            output=tmp_path / "M0",
            base_model=BASE_MODEL,
            base_revision=REVISION,
            tokenizer_revision=REVISION,
            opd_config=_opd_config(tmp_path / "opd.yaml"),
            policy_loader=_policy_loader,
        )
