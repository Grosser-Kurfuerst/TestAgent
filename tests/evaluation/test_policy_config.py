from __future__ import annotations

from pathlib import Path

import pytest

from my_agent.config import AgentConfig
from my_agent.evaluation.policy_config import configure_evaluation_policy
from my_agent.evaluation.policy_config import validate_evaluation_policy_identity
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    policy_identity_manifest_payload,
)


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=4,
        command_timeout=20,
        trace_dir=tmp_path / "traces",
        use_fake_llm=True,
    )


def _identity(*, adapter: bool) -> PolicyIdentity:
    return PolicyIdentity(
        base_model="fixture/model",
        base_revision="revision-1",
        checkpoint_hash=canonical_sha256("checkpoint"),
        adapter_hash=canonical_sha256("adapter") if adapter else None,
        tokenizer_revision="tokenizer-1",
        tokenizer_hash=canonical_sha256("tokenizer"),
        chat_template_hash=canonical_sha256("template"),
    )


def _write_manifest(path: Path, identity: PolicyIdentity) -> None:
    path.write_bytes(canonical_json_bytes(policy_identity_manifest_payload(identity)) + b"\n")


def test_configure_evaluation_policy_preserves_eval_manifest_defaults(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    assert configure_evaluation_policy(
        config,
        checkpoint=tmp_path / "unused",
        identity_manifest=None,
    ) is config


def test_configure_evaluation_policy_resolves_adapter_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    identity = _identity(adapter=True)
    manifest = tmp_path / "policy_identity_manifest.json"
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    _write_manifest(manifest, identity)

    resolved = configure_evaluation_policy(
        config,
        checkpoint=checkpoint,
        identity_manifest=manifest,
    )

    assert resolved.policy_backend == "transformers"
    assert resolved.policy_base_model == identity.base_model
    assert resolved.policy_base_revision == identity.base_revision
    assert resolved.policy_tokenizer_revision == identity.tokenizer_revision
    assert resolved.policy_adapter_path == checkpoint.resolve()
    assert resolved.policy_identity_manifest == manifest.resolve()


def test_configure_evaluation_policy_rejects_missing_adapter_checkpoint(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "policy_identity_manifest.json"
    _write_manifest(manifest, _identity(adapter=True))

    with pytest.raises(FileNotFoundError, match="evaluation checkpoint not found"):
        configure_evaluation_policy(
            _config(tmp_path),
            checkpoint=tmp_path / "missing",
            identity_manifest=manifest,
        )


def test_validate_evaluation_policy_identity_accepts_matching_runtime(
    tmp_path: Path,
) -> None:
    expected = _identity(adapter=True)

    class _Policy:
        def identity(self) -> PolicyIdentity:
            return expected

    actual = validate_evaluation_policy_identity(
        _config(tmp_path),
        expected,
        policy_loader=lambda _config: _Policy(),
    )

    assert actual == expected


def test_validate_evaluation_policy_identity_rejects_runtime_mismatch(
    tmp_path: Path,
) -> None:
    expected = _identity(adapter=True)
    actual = PolicyIdentity(
        **{
            **expected.to_dict(),
            "adapter_hash": canonical_sha256("different-adapter"),
        }
    )

    class _Policy:
        def identity(self) -> PolicyIdentity:
            return actual

    with pytest.raises(ValueError, match="policy identity mismatch"):
        validate_evaluation_policy_identity(
            _config(tmp_path),
            expected,
            policy_loader=lambda _config: _Policy(),
        )
