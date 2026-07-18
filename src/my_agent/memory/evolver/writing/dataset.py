"""Legacy writer dataset JSONL output."""

from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from my_agent.memory.evolver.selection.contracts import SelectionResult
from my_agent.memory.evolver.writing.contracts import (
    ExperienceWriteProposal,
    ExperienceWriteRequest,
    ExperienceWriteResult,
    ExperienceWriteStep,
)
from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.experience.serialization import experience_payload_to_dict
from my_agent.text_safety import sanitize_json_value

WRITER_METADATA_STRING_CHARS = 1_000
WRITER_DATASET_TASK_CHARS = 2_000
WRITER_DATASET_CONTENT_CHARS = 1_200
WRITER_DATASET_OUTPUT_CHARS = 1_000
WRITER_DATASET_FORBIDDEN_MARKERS = (
    "hidden_test_output",
    "hidden_ok",
    "official_solution",
    "ground_truth",
    "expected_patch",
    "private_key",
    "api_key",
    "apikey",
    "access_key",
    "bearer",
    "cookie",
    "credential",
    "github_pat_",
    "ghp_",
    "glpat-",
    "password",
    "secret",
    "token",
)
WRITER_DATASET_SECRET_PREFIX_RE = re.compile(
    r"(?i)(?:github_pat_|ghp_|glpat-|sk-[A-Za-z0-9_-]{16,}|xox[baprs]-|AKIA|ASIA|AIza|ya29\.|eyJ[A-Za-z0-9_-]{8,})"
)


class MemoryWriterDatasetLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    sanitize_json_value(record),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def writer_dataset_context_payload(
    request: ExperienceWriteRequest,
) -> dict[str, Any]:
    return {
        "source_task": safe_dataset_join_text(request.source_task),
        "stream_id": safe_dataset_join_text(request.stream_id),
        "task_type": safe_dataset_join_text(request.task_type),
        "memory_project_key": safe_dataset_join_text(request.project_key),
        "memory_mode": safe_dataset_join_text(request.memory_mode),
    }


def writer_dataset_record(
    *,
    request: ExperienceWriteRequest,
    result: ExperienceWriteResult,
    selection: SelectionResult | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "task": safe_dataset_text(request.task, WRITER_DATASET_TASK_CHARS),
        "run_id": request.run_id,
        "trace_path": safe_dataset_join_text(str(request.trace_path or "")),
        "source_task": safe_dataset_join_text(request.source_task),
        "task_id": safe_dataset_join_text(request.source_task),
        "task_type": safe_dataset_join_text(request.task_type),
        "stream_id": safe_dataset_join_text(request.stream_id),
        "memory_project_key": safe_dataset_join_text(request.project_key),
        "memory_mode": safe_dataset_join_text(request.memory_mode),
        "outcome": request.outcome,
        "outcome_source": request.outcome_source,
        "stop_reason": request.stop_reason,
        "selected_memory_ids": list(request.selected_memory_ids),
        "candidate_memory_ids": list(request.candidate_memory_ids),
        "selected_memory_ids_by_tier": selection_selected_ids_by_tier(selection),
        "candidate_memory_ids_by_tier": selection_candidate_ids_by_tier(selection),
        "steps": [writer_dataset_step(step) for step in request.steps],
        "proposals": [writer_dataset_proposal(item) for item in result.proposals],
        "saved_ids": [entry.id for entry in result.saved],
        "saved_records": saved_records(result.saved),
        "duplicate_ids": list(result.duplicate_ids),
        "rejected": [safe_dataset_value(dict(item)) for item in result.rejected],
        "llm_used": result.llm_used,
        "fallback_used": result.fallback_used,
    }
    if result.error:
        record["error"] = safe_error_text(result.error)
    return record


def writer_dataset_step(step: ExperienceWriteStep) -> dict[str, Any]:
    output = str(step.output or "")
    redacted = unsafe_dataset_text(output)
    return {
        "step_num": int(step.step_num or 0),
        "tool": str(step.tool or ""),
        "arguments": safe_dataset_value(dict(step.arguments or {})),
        "ok": bool(step.ok),
        "output": ""
        if redacted
        else safe_dataset_text(output, WRITER_DATASET_OUTPUT_CHARS),
        "output_redacted": redacted,
        "blocked": bool(step.blocked),
        "error_code": str(step.error_code or ""),
    }


def writer_dataset_proposal(proposal: ExperienceWriteProposal) -> dict[str, Any]:
    try:
        serialized_payload = experience_payload_to_dict(proposal.payload)
    except (TypeError, ValueError):
        serialized_payload = {}
    tier = proposal.tier
    return {
        "tier": tier.value if isinstance(tier, ExperienceTier) else str(tier),
        "content": safe_dataset_text(
            str(proposal.content or ""),
            WRITER_DATASET_CONTENT_CHARS,
        ),
        "payload": safe_dataset_value(serialized_payload),
        "confidence": float(proposal.confidence or 0.0),
        "reason": safe_dataset_text(
            str(proposal.reason or ""),
            WRITER_METADATA_STRING_CHARS,
        ),
    }


def safe_dataset_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe_items: dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            if unsafe_dataset_text(raw_key):
                safe_items[safe_dataset_redaction(raw_key)] = ""
                continue
            safe_items[
                safe_dataset_text(raw_key, WRITER_METADATA_STRING_CHARS)
            ] = safe_dataset_value(item)
        return safe_items
    if isinstance(value, list):
        return [safe_dataset_value(item) for item in value]
    if isinstance(value, tuple):
        return [safe_dataset_value(item) for item in value]
    if isinstance(value, str):
        return safe_dataset_text(value, WRITER_METADATA_STRING_CHARS)
    return value


def safe_dataset_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if unsafe_dataset_text(text):
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3].rstrip() + "..."


def safe_dataset_join_text(value: str) -> str:
    text = str(value or "")
    if unsafe_dataset_text(text):
        return safe_dataset_redaction(text)
    return safe_dataset_text(text, WRITER_METADATA_STRING_CHARS)


def safe_error_text(value: str) -> str:
    text = str(value or "")
    if unsafe_dataset_text(text):
        return safe_dataset_redaction(text)
    return safe_dataset_text(text, WRITER_METADATA_STRING_CHARS)


def safe_dataset_redaction(value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:12]
    return f"redacted_{digest}"


def unsafe_dataset_text(value: str) -> bool:
    text = str(value or "")
    lower = text.casefold()
    if WRITER_DATASET_SECRET_PREFIX_RE.search(text):
        return True
    return any(marker in lower for marker in WRITER_DATASET_FORBIDDEN_MARKERS)


def selection_candidate_ids_by_tier(
    selection: SelectionResult | None,
) -> dict[str, list[str]]:
    if selection is None:
        return {}
    grouped: dict[str, list[str]] = {}
    for candidate in selection.candidates:
        grouped.setdefault(candidate.tier.value, []).append(candidate.id)
    return grouped


def selection_selected_ids_by_tier(
    selection: SelectionResult | None,
) -> dict[str, list[str]]:
    if selection is None:
        return {}
    grouped: dict[str, list[str]] = {}
    for item in selection.selected:
        grouped.setdefault(item.candidate.tier.value, []).append(item.candidate.id)
    return grouped


def saved_records(entries: tuple[ExperienceMemory, ...]) -> list[dict[str, str]]:
    return [
        {
            "id": entry.id,
            "tier": entry.tier.value,
            "source_task": entry.source_task,
        }
        for entry in entries
    ]


__all__ = [
    "MemoryWriterDatasetLogger",
    "safe_dataset_join_text",
    "safe_error_text",
    "saved_records",
    "writer_dataset_context_payload",
    "writer_dataset_record",
]
