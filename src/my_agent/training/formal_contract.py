"""Fail-closed policy boundaries for the formal OPD runtime."""

from __future__ import annotations

from collections.abc import Iterable


FORMAL_MAINTENANCE_ACTIONS = frozenset({"lookup", "merge", "delete", "finish"})
FORMAL_LEGACY_CONFIG_DEFAULTS = {
    "memory_evolver_top_k_per_tier": 50,
    "memory_evolver_min_score": 0.0,
    "memory_evolver_min_experience_entries": 0,
    "memory_evolver_tier_caps": {
        "trajectory": 1,
        "tip": 2,
        "skill": 2,
        "tool": 2,
    },
    "memory_evolver_tier_weights": {
        "trajectory": 0.90,
        "tip": 1.00,
        "skill": 1.20,
        "tool": 1.10,
    },
    "memory_evolver_writer_enabled": False,
    "memory_evolver_writer_mode": "fallback",
    "memory_evolver_writer_min_confidence": 0.70,
    "memory_evolver_writer_max_records": 6,
    "memory_evolver_writer_max_input_chars": 12_000,
    "memory_evolver_writer_max_content_chars": 1_200,
    "memory_evolver_writer_dataset_path": None,
}
FORMAL_FORBIDDEN_LEGACY_CONFIG_KEYS = frozenset({
    "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER",
    "MY_AGENT_MEMORY_EVOLVER_TOP_K_PER_TIER",
    "AGENTCLI_MEMORY_EVOLVER_MIN_SCORE",
    "MY_AGENT_MEMORY_EVOLVER_MIN_SCORE",
    "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES",
    "MY_AGENT_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES",
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
    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS",
    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_RECORDS",
    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS",
    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS",
    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS",
    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS",
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
    "FORMAL_LEGACY_CONFIG_DEFAULTS",
    "FORMAL_MAINTENANCE_ACTIONS",
    "require_formal_maintenance_actions",
]
