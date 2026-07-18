"""Shared contracts for legacy and formal Experience writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from my_agent.memory.experience.models import (
    ExperienceMemory,
    ExperiencePayload,
    ExperienceTier,
)
from my_agent.text_safety import sanitize_json_value


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
        return sanitize_json_value({
            "step_num": int(self.step_num),
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "ok": bool(self.ok),
            "output": self.output,
            "blocked": bool(self.blocked),
            "error_code": self.error_code,
        })


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
    memory_mode: str = ""


@dataclass(frozen=True)
class ExperienceWriteProposal:
    tier: ExperienceTier
    content: str
    payload: ExperiencePayload
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class ExperienceWriteResult:
    proposals: tuple[ExperienceWriteProposal, ...] = ()
    saved: tuple[ExperienceMemory, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    llm_used: bool = False
    fallback_used: bool = False
    error: str = ""


class ProposalGenerator(Protocol):
    def generate(self, request: ExperienceWriteRequest) -> ExperienceWriteResult: ...


__all__ = [
    "ExperienceWriteProposal",
    "ExperienceWriteRequest",
    "ExperienceWriteResult",
    "ExperienceWriteStep",
    "ProposalGenerator",
]
