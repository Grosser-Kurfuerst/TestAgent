"""Fail-closed policy boundaries for the formal OPD runtime."""

from __future__ import annotations

from collections.abc import Iterable


FORMAL_MAINTENANCE_ACTIONS = frozenset({"lookup", "merge", "delete", "finish"})
FORMAL_FORBIDDEN_LEGACY_CONFIG_KEYS = frozenset({
    "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER",
    "MY_AGENT_MEMORY_EVOLVER_TOP_K_PER_TIER",
    "AGENTCLI_MEMORY_EVOLVER_MIN_SCORE",
    "MY_AGENT_MEMORY_EVOLVER_MIN_SCORE",
    "AGENTCLI_MEMORY_EVOLVER_TIER_CAPS",
    "MY_AGENT_MEMORY_EVOLVER_TIER_CAPS",
    "AGENTCLI_MEMORY_EVOLVER_TIER_WEIGHTS",
    "MY_AGENT_MEMORY_EVOLVER_TIER_WEIGHTS",
    "AGENTCLI_MEMORY_EVOLVER_WRITER_MODE",
    "MY_AGENT_MEMORY_EVOLVER_WRITER_MODE",
    "AGENTCLI_MEMORY_EVOLVER_WRITER",
    "MY_AGENT_MEMORY_EVOLVER_WRITER",
    "AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE",
    "MY_AGENT_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE",
    "AGENTCLI_MEMORY_EVOLVER_WRITER_DATASET_PATH",
    "MY_AGENT_MEMORY_EVOLVER_WRITER_DATASET_PATH",
})


def require_formal_maintenance_actions(actions: Iterable[str]) -> None:
    invalid = sorted({str(action) for action in actions} - FORMAL_MAINTENANCE_ACTIONS)
    if invalid:
        raise ValueError(
            "formal maintenance supports only lookup/merge/delete/finish; "
            f"unsupported actions: {', '.join(invalid)}"
        )


__all__ = [
    "FORMAL_FORBIDDEN_LEGACY_CONFIG_KEYS",
    "FORMAL_MAINTENANCE_ACTIONS",
    "require_formal_maintenance_actions",
]
