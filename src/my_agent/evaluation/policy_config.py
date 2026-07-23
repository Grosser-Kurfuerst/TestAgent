"""Shared frozen-policy configuration for evaluation entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from my_agent.config import AgentConfig
from my_agent.policy.identity import (
    PolicyIdentity,
    load_policy_identity_manifest,
    require_matching_policy_identity,
)


def configure_evaluation_policy(
    config: AgentConfig,
    *,
    checkpoint: str | Path | None,
    identity_manifest: str | Path | None,
) -> AgentConfig:
    """Apply one policy identity exactly as eval-manifest historically did."""

    if not isinstance(config, AgentConfig):
        raise ValueError("evaluation policy configuration requires AgentConfig")
    if identity_manifest is None:
        return config
    identity_path = Path(identity_manifest).expanduser().resolve()
    identity = load_policy_identity_manifest(identity_path)
    checkpoint_path = (
        Path(checkpoint).expanduser().resolve()
        if checkpoint is not None
        else identity_path.parent
    )
    if identity.adapter_hash is not None and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"evaluation checkpoint not found: {checkpoint_path}"
        )
    return replace(
        config,
        policy_backend="transformers",
        policy_base_model=identity.base_model,
        policy_base_revision=identity.base_revision,
        policy_tokenizer_revision=identity.tokenizer_revision,
        policy_adapter_path=(
            checkpoint_path if identity.adapter_hash is not None else None
        ),
        policy_identity_manifest=identity_path,
    )


def validate_evaluation_policy_identity(
    config: AgentConfig,
    expected: PolicyIdentity,
    *,
    policy_loader: Callable[[AgentConfig], Any] | None = None,
) -> PolicyIdentity:
    """Load the configured policy once and verify its actual frozen identity."""

    if not isinstance(config, AgentConfig):
        raise ValueError("evaluation policy identity validation requires AgentConfig")
    if not isinstance(expected, PolicyIdentity):
        raise ValueError("expected evaluation policy identity must be PolicyIdentity")
    if policy_loader is None:
        from my_agent.policy.transformers_policy import TransformersPolicy

        policy_loader = TransformersPolicy.from_config
    policy = policy_loader(config)
    identity_method = getattr(policy, "identity", None)
    if not callable(identity_method):
        raise ValueError("evaluation policy loader must expose identity()")
    actual = identity_method()
    if not isinstance(actual, PolicyIdentity):
        raise ValueError("evaluation policy identity() must return PolicyIdentity")
    require_matching_policy_identity(expected, actual)
    return actual


__all__ = [
    "configure_evaluation_policy",
    "validate_evaluation_policy_identity",
]
