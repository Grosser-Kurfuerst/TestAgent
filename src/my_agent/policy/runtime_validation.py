"""Fail-closed validation for formal policy runtime entry points."""

from __future__ import annotations

from typing import Any

from my_agent.config import AgentConfig
from my_agent.policy.contracts import TrainablePolicy
from my_agent.policy.identity import (
    PolicyIdentity,
    load_policy_identity_manifest,
    require_matching_policy_identity,
)


def require_formal_policy(
    config: AgentConfig,
    policy: Any,
) -> PolicyIdentity | None:
    """Validate config, white-box capability, and versioned identity for formal runs."""

    config.require_valid_formal_evolver()
    if config.memory_evolver_mode != "formal":
        return None
    if not isinstance(policy, TrainablePolicy):
        raise ValueError("formal OPD runtime requires a TrainablePolicy")
    if config.policy_identity_manifest is None:
        raise ValueError("formal OPD runtime requires policy_identity_manifest")
    identity = policy.identity()
    if not isinstance(identity, PolicyIdentity):
        raise ValueError("formal OPD runtime requires policy.identity() to return PolicyIdentity")
    expected = load_policy_identity_manifest(config.policy_identity_manifest)
    require_matching_policy_identity(expected, identity)
    return identity


__all__ = ["require_formal_policy"]
