from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from my_agent.memory.evolver.types import ExperienceTier, normalize_experience_tier
from my_agent.memory.types import MemoryEntry
from my_agent.text_safety import sanitize_json_value


DEFAULT_STEP_OUTPUT_CHARS = 1_000
DEFAULT_TASK_CHARS = 2_000
FAILURE_STOP_MARKERS = (
    "failed",
    "failure",
    "error",
    "context_over_budget",
    "llm_failed",
    "max_steps_reached",
    "budget",
    "timeout",
)


@dataclass(frozen=True)
class ExperienceWriteStep:
    step_num: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    output: str = ""
    blocked: bool = False
    error_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_json_value(
            {
                "step_num": int(self.step_num),
                "tool": self.tool,
                "arguments": dict(self.arguments),
                "ok": bool(self.ok),
                "output": self.output,
                "blocked": bool(self.blocked),
                "error_code": self.error_code,
            }
        )


@dataclass(frozen=True)
class ExperienceWriteRequest:
    task: str
    run_id: str
    trace_path: Path | None
    stop_reason: str
    outcome: str
    outcome_source: str
    final_answer: str = ""
    selected_memory_ids: tuple[str, ...] = ()
    candidate_memory_ids: tuple[str, ...] = ()
    steps: tuple[ExperienceWriteStep, ...] = ()
    source_task: str = ""
    stream_id: str = ""
    task_type: str = ""
    project_key: str = ""


@dataclass(frozen=True)
class ExperienceWriteProposal:
    tier: ExperienceTier
    content: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class ExperienceWriteResult:
    proposals: tuple[ExperienceWriteProposal, ...] = ()
    saved: tuple[MemoryEntry, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    llm_used: bool = False
    fallback_used: bool = False
    error: str = ""


class ExperienceWriter:
    """Builds reusable experience proposals from a completed run."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        min_confidence: float = 0.70,
        max_records: int = 6,
        max_input_chars: int = 12_000,
        max_content_chars: int = 1_200,
    ) -> None:
        self.llm = llm
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_records = max(0, int(max_records))
        self.max_input_chars = max(0, int(max_input_chars))
        self.max_content_chars = max(0, int(max_content_chars))

    def validate_proposals(
        self, proposals: Sequence[ExperienceWriteProposal]
    ) -> tuple[tuple[ExperienceWriteProposal, ...], tuple[dict[str, Any], ...]]:
        if self.max_records <= 0:
            return (), tuple(_rejection(proposal, "max_records_zero") for proposal in proposals)

        accepted: list[ExperienceWriteProposal] = []
        rejected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for proposal in proposals:
            valid, reason, normalized = self.validate_proposal(proposal)
            if not valid or normalized is None:
                rejected.append(_rejection(proposal, reason))
                continue
            fingerprint = _content_fingerprint(normalized.content)
            if fingerprint in seen:
                rejected.append(_rejection(proposal, "duplicate_proposal"))
                continue
            seen.add(fingerprint)
            accepted.append(normalized)
            if len(accepted) >= self.max_records:
                break
        return tuple(accepted), tuple(rejected)

    def validate_proposal(
        self, proposal: ExperienceWriteProposal
    ) -> tuple[bool, str, ExperienceWriteProposal | None]:
        tier = normalize_experience_tier(proposal.tier)
        if tier is None:
            return False, "invalid_tier", None

        content = str(proposal.content or "").strip()
        if not content:
            return False, "empty_content", None

        confidence = _safe_float(proposal.confidence)
        if confidence is None:
            return False, "invalid_confidence", None
        if confidence < self.min_confidence:
            return False, "low_confidence", None

        content = _truncate_text(content, self.max_content_chars).strip()
        if not content:
            return False, "empty_content", None

        normalized = ExperienceWriteProposal(
            tier=tier,
            content=content,
            confidence=confidence,
            metadata=sanitize_json_value(dict(proposal.metadata or {})),
            reason=str(proposal.reason or "").strip(),
        )
        return True, "", normalized


def build_write_steps_from_tool_history(
    tool_history: Sequence[Mapping[str, Any]] | None,
    *,
    max_output_chars: int = DEFAULT_STEP_OUTPUT_CHARS,
) -> tuple[ExperienceWriteStep, ...]:
    steps: list[ExperienceWriteStep] = []
    for index, raw_record in enumerate(tool_history or (), 1):
        if not isinstance(raw_record, Mapping):
            continue
        call = _mapping(raw_record.get("call"))
        result = _mapping(raw_record.get("result"))
        tool = str(call.get("tool") or raw_record.get("tool") or "").strip()
        if not tool:
            continue
        arguments = _mapping(call.get("arguments"))
        output = _truncate_text(str(result.get("output") or raw_record.get("output") or ""), max_output_chars)
        error_code = str(result.get("reason") or result.get("error_code") or raw_record.get("error_code") or "")
        steps.append(
            ExperienceWriteStep(
                step_num=len(steps) + 1,
                tool=tool,
                arguments=sanitize_json_value(dict(arguments)),
                ok=bool(result.get("ok", raw_record.get("ok", False))),
                output=output,
                blocked=bool(result.get("blocked", raw_record.get("blocked", False))),
                error_code=error_code,
            )
        )
    return tuple(steps)


def runtime_outcome_from_tool_records(
    stop_reason: str,
    tool_history: Sequence[Mapping[str, Any]] | None,
) -> str:
    steps = build_write_steps_from_tool_history(tool_history)
    latest_test = next((step for step in reversed(steps) if step.tool == "run_tests"), None)
    if latest_test is not None:
        return "success" if latest_test.ok else "failure"

    normalized_stop = str(stop_reason or "").casefold()
    if any(marker in normalized_stop for marker in FAILURE_STOP_MARKERS):
        return "failure"
    return "unknown"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _content_fingerprint(content: str) -> str:
    return " ".join(content.casefold().split())


def _rejection(proposal: ExperienceWriteProposal, reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "tier": proposal.tier.value if isinstance(proposal.tier, ExperienceTier) else str(proposal.tier),
    }


__all__ = [
    "ExperienceWriteProposal",
    "ExperienceWriteRequest",
    "ExperienceWriteResult",
    "ExperienceWriteStep",
    "ExperienceWriter",
    "build_write_steps_from_tool_history",
    "runtime_outcome_from_tool_records",
]
