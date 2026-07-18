from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from my_agent.memory.experience.serialization import (
    experience_payload_from_dict,
    experience_payload_to_dict,
)
from my_agent.memory.experience.models import (
    ExperienceMemory,
    ExperiencePayload,
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
    normalize_experience_tier,
)
from my_agent.text_safety import sanitize_json_value


DEFAULT_STEP_OUTPUT_CHARS = 1_000
DEFAULT_TASK_CHARS = 2_000
MAX_REASON_CHARS = 500
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

# writer_policy records which runtime policy produced each saved experience, so
# downstream Phase 5/8 distillation and human audits can tell LLM-authored
# experiences apart from deterministic fallback outputs. Mirrors the result
# flags on ExperienceWriteResult.
WRITER_POLICY_LLM = "llm_json_v1"
WRITER_POLICY_LLM_THEN_FALLBACK = "llm_then_fallback_runtime_v1"
WRITER_POLICY_FALLBACK = "fallback_runtime_v1"
FORBIDDEN_MARKERS = (
    "hidden_test_output",
    "hidden_ok",
    "official_solution",
    "ground_truth",
    "expected_patch",
    "private_key",
    "api_key",
    "apikey",
    "bearer",
    "github_pat_",
    "ghp_",
    "glpat-",
    "sk-",
)
SECRET_PREFIX_RE = re.compile(
    r"(?i)(?:github_pat_|ghp_|glpat-|xox[baprs]-|AKIA|ASIA|AIza|ya29\.|eyJ[A-Za-z0-9_-]{8,})"
)
DESTRUCTIVE_COMMAND_RE = re.compile(r"(?i)(?:\brm\s+-rf\b|\bgit\s+reset\s+--hard\b)")


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


class MemoryWriterDatasetLogger:
    """Append-only JSONL logger for writer training/distillation records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitize_json_value(record), ensure_ascii=False, sort_keys=True) + "\n")


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

    def propose(self, request: ExperienceWriteRequest, *, mode: str = "fallback") -> ExperienceWriteResult:
        if str(mode or "fallback").strip().lower() == "llm" and self.llm is not None:
            return self._propose_with_llm(request)
        proposals, rejected = self.validate_proposals(self._fallback_proposals(request))
        return ExperienceWriteResult(proposals=proposals, rejected=rejected, fallback_used=True)

    def _propose_with_llm(self, request: ExperienceWriteRequest) -> ExperienceWriteResult:
        rejected: list[dict[str, Any]] = []
        try:
            llm_proposals = self._llm_proposals(request)
        except Exception as exc:  # noqa: BLE001 - invalid writer output should fallback
            rejected.append({"reason": "llm_parse_failed", "error": f"{type(exc).__name__}: {exc}"})
        else:
            proposals, proposal_rejections = self.validate_proposals(llm_proposals)
            rejected.extend(proposal_rejections)
            if proposals:
                return ExperienceWriteResult(proposals=proposals, rejected=tuple(rejected), llm_used=True)

        fallback_proposals, fallback_rejections = self.validate_proposals(self._fallback_proposals(request))
        rejected.extend(fallback_rejections)
        return ExperienceWriteResult(
            proposals=fallback_proposals,
            rejected=tuple(rejected),
            llm_used=True,
            fallback_used=True,
        )

    def _llm_proposals(self, request: ExperienceWriteRequest) -> tuple[ExperienceWriteProposal, ...]:
        prompt = _llm_prompt(request, max_input_chars=self.max_input_chars)
        response = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an experience writer. Return only a JSON array of reusable experience "
                        "records. Do not include prose, secrets, hidden tests, official answers, or full logs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        content = str(getattr(response, "content", response) or "")
        payload = _parse_json_array(content)
        proposals: list[ExperienceWriteProposal] = []
        for item in payload:
            if not isinstance(item, Mapping):
                proposals.append(
                    ExperienceWriteProposal(
                        tier="",  # type: ignore[arg-type]
                        content="",
                        payload={},  # type: ignore[arg-type]
                        confidence=0.0,
                        reason="non_object",
                    )
                )
                continue
            proposals.append(
                ExperienceWriteProposal(
                    tier=item.get("tier", ""),  # type: ignore[arg-type]
                    content=str(item.get("content") or ""),
                    payload=(
                        dict(item.get("payload") or {})
                        if isinstance(item.get("payload"), Mapping)
                        else item.get("payload")  # type: ignore[arg-type]
                    ),
                    confidence=item.get("confidence", 0.0),  # type: ignore[arg-type]
                    reason=str(item.get("reason") or ""),
                )
            )
        return tuple(proposals)

    def validate_proposals(
        self, proposals: Sequence[ExperienceWriteProposal]
    ) -> tuple[tuple[ExperienceWriteProposal, ...], tuple[dict[str, Any], ...]]:
        if self.max_records <= 0:
            return (), tuple(_rejection(proposal, "max_records_zero") for proposal in proposals)

        accepted: list[ExperienceWriteProposal] = []
        rejected: list[dict[str, Any]] = []
        seen: set[tuple[ExperienceTier, str]] = set()
        for proposal in proposals:
            valid, reason, normalized = self.validate_proposal(proposal)
            if not valid or normalized is None:
                rejected.append(_rejection(proposal, reason))
                continue
            identity = (normalized.tier, _content_fingerprint(normalized.content))
            if identity in seen:
                rejected.append(_rejection(proposal, "duplicate_proposal"))
                continue
            seen.add(identity)
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

        try:
            if isinstance(proposal.payload, Mapping):
                payload = experience_payload_from_dict(tier, proposal.payload)
            else:
                payload = proposal.payload
                # Canonical round-trip validates both the concrete type and its tier.
                payload = experience_payload_from_dict(
                    tier,
                    experience_payload_to_dict(payload),
                )
        except (TypeError, ValueError):
            return False, "invalid_payload", None

        raw_payload = experience_payload_to_dict(payload)
        raw_reason = str(proposal.reason or "").strip()
        if (
            _contains_forbidden_content(content)
            or _contains_forbidden_content(raw_reason)
            or _contains_forbidden_content(_json_text(raw_payload))
        ):
            return False, "unsafe_content", None

        if isinstance(payload, ToolPayload) and _contains_destructive_tool(payload):
            return False, "unsafe_tool_command", None
        reason = _truncate_text(raw_reason, MAX_REASON_CHARS).strip()

        normalized = ExperienceWriteProposal(
            tier=tier,
            content=content,
            payload=payload,
            confidence=confidence,
            reason=reason,
        )
        return True, "", normalized

    def _fallback_proposals(self, request: ExperienceWriteRequest) -> tuple[ExperienceWriteProposal, ...]:
        if request.outcome == "success":
            return self._success_fallback(request)
        if request.outcome == "failure":
            return self._failure_fallback(request)
        return self._unknown_fallback(request)

    def _success_fallback(self, request: ExperienceWriteRequest) -> tuple[ExperienceWriteProposal, ...]:
        proposals: list[ExperienceWriteProposal] = []
        latest_test = _latest_step(request.steps, "run_tests")
        if latest_test is not None and latest_test.ok:
            proposals.append(
                ExperienceWriteProposal(
                    tier=ExperienceTier.SKILL,
                    content=(
                        "When a coding task reaches a passing focused test, preserve the narrow loop: "
                        "inspect the failure, patch the smallest relevant path, rerun the exact test, "
                        "then finish only after the verification result is stable."
                    ),
                    payload=SkillPayload(
                        category="debugging",
                        technique="Focused verification loop",
                        preconditions=(
                            "A focused run_tests command exposes or verifies the behavior.",
                        ),
                        steps=(
                            "inspect the focused failure or target behavior",
                            "patch the smallest relevant code path",
                            "rerun the exact focused test",
                            "finish only after the focused verification passes",
                        ),
                    ),
                    confidence=0.80,
                    reason="successful run with passing run_tests signal",
                )
            )
        command = _step_command(latest_test) if latest_test is not None else ""
        if command:
            proposals.append(
                ExperienceWriteProposal(
                    tier=ExperienceTier.TOOL,
                    content=f"Run the focused project test from the repository root: {command}",
                    payload=ToolPayload(
                        name="run_focused_tests",
                        language="bash",
                        code=command,
                        command=command,
                        input_description="Run from the repository root after applying the targeted change.",
                        output_description="Focused test result for the changed behavior.",
                    ),
                    confidence=0.78,
                    reason="successful run_tests command is reusable",
                )
            )
        return tuple(proposals)

    def _failure_fallback(self, request: ExperienceWriteRequest) -> tuple[ExperienceWriteProposal, ...]:
        failing_test = _latest_failing_test(request.steps)
        failing_tool = _latest_failing_tool(request.steps)
        stopped_by = request.stop_reason or "failure"
        if failing_test is not None:
            trigger = failing_test.error_code or stopped_by
            tip_text = (
                f"This run stopped with {stopped_by} after the latest run_tests signal "
                "failed. For similar tasks, inspect the last failing assertion or blocked "
                "tool result before issuing another patch or broad test command."
            )
        elif failing_tool is not None:
            trigger = failing_tool.error_code or stopped_by
            verb = "was blocked" if failing_tool.blocked else "failed"
            tip_text = (
                f"This run stopped with {stopped_by} after the latest tool {verb} or produced "
                "an unusable result. For similar tasks, inspect the failed or blocked tool "
                "result before retrying the same approach or broadening the command."
            )
        else:
            # The failure was inferred purely from stop_reason (budget / max steps /
            # timeout) without any failing test or tool signal to attribute it to.
            trigger = stopped_by
            tip_text = (
                f"This run stopped with {stopped_by} before reaching a verified success "
                "(no failing run_tests or blocked tool signal observed). For similar "
                "tasks, watch the remaining budget or step budget closely and prefer a "
                "narrow verification over a broad exploratory action."
            )
        proposals: list[ExperienceWriteProposal] = [
            ExperienceWriteProposal(
                tier=ExperienceTier.TIP,
                content=tip_text,
                payload=TipPayload(
                    category="debugging",
                    severity="warning",
                    trigger=trigger,
                ),
                confidence=0.76,
                reason="failure outcome with actionable next-step guard",
            )
        ]
        if len(request.steps) >= 2:
            proposals.append(
                ExperienceWriteProposal(
                    tier=ExperienceTier.TRAJECTORY,
                    content=_trajectory_content(request),
                    payload=TrajectoryPayload(
                        task_description=_truncate_text(request.task, DEFAULT_TASK_CHARS),
                        steps=tuple(_trajectory_step_payload(step) for step in request.steps),
                        outcome=request.outcome,
                        total_reward=0.0,
                        key_learnings=(
                            "Inspect the latest failing signal before making the next change.",
                        ),
                        tags=("failure", "runtime"),
                    ),
                    confidence=0.72,
                    reason="failure run with multiple tool steps",
                )
            )
        return tuple(proposals)

    def _unknown_fallback(self, request: ExperienceWriteRequest) -> tuple[ExperienceWriteProposal, ...]:
        if not request.steps:
            return ()
        return (
            ExperienceWriteProposal(
                tier=ExperienceTier.TRAJECTORY,
                content=_trajectory_content(request),
                payload=TrajectoryPayload(
                    task_description=_truncate_text(request.task, DEFAULT_TASK_CHARS),
                    steps=tuple(_trajectory_step_payload(step) for step in request.steps),
                    outcome=request.outcome,
                    total_reward=0.0,
                    key_learnings=(),
                    tags=("unknown", "runtime"),
                ),
                confidence=0.60,
                reason="unknown outcome with tool trajectory only",
            ),
        )


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


def proposal_tier_counts(proposals: Sequence[ExperienceWriteProposal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for proposal in proposals:
        tier = normalize_experience_tier(proposal.tier)
        if tier is None:
            continue
        counts[tier.value] = counts.get(tier.value, 0) + 1
    return counts


def writer_policy_for_result(*, llm_used: bool, fallback_used: bool) -> str:
    """Resolve the ``writer_policy`` label for a saved experience / trace event.

    The label is task/run context kept in trace and writer datasets; it is not
    persisted on individual experience records.
    """
    if llm_used and not fallback_used:
        return WRITER_POLICY_LLM
    if llm_used and fallback_used:
        return WRITER_POLICY_LLM_THEN_FALLBACK
    return WRITER_POLICY_FALLBACK


def _llm_prompt(request: ExperienceWriteRequest, *, max_input_chars: int) -> str:
    payload = {
        "task": _truncate_text(request.task, DEFAULT_TASK_CHARS),
        "run_id": request.run_id,
        "trace_path": str(request.trace_path or ""),
        "stop_reason": request.stop_reason,
        "outcome": request.outcome,
        "outcome_source": request.outcome_source,
        "final_answer": _truncate_text(request.final_answer, 1_000),
        "selected_memory_ids": list(request.selected_memory_ids),
        "candidate_memory_ids": list(request.candidate_memory_ids),
        "source_task": request.source_task,
        "stream_id": request.stream_id,
        "task_type": request.task_type,
        "memory_project_key": request.project_key,
        "steps": [
            {
                "step_num": step.step_num,
                "tool": step.tool,
                "arguments": dict(step.arguments),
                "ok": step.ok,
                "output": _truncate_text(step.output, DEFAULT_STEP_OUTPUT_CHARS),
                "blocked": step.blocked,
                "error_code": step.error_code,
            }
            for step in request.steps
        ],
    }
    prompt = (
        "Return JSON array only. Each item must have tier, content, payload, confidence, and reason. "
        "Allowed tiers: trajectory, tip, skill, tool. Payload must use the tier's canonical schema; "
        "do not add task/run metadata to payload. Keep content reusable and independent.\n"
        f"Run summary:\n{json.dumps(sanitize_json_value(payload), ensure_ascii=False, sort_keys=True)}"
    )
    return _truncate_text(prompt, max_input_chars)


def _parse_json_array(text: str) -> list[Any]:
    raw = _extract_json_text(text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"writer JSON is invalid: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("writer JSON must be an array")
    return parsed


def _extract_json_text(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("writer response is empty")
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return stripped


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _latest_step(steps: Sequence[ExperienceWriteStep], tool: str) -> ExperienceWriteStep | None:
    return next((step for step in reversed(steps) if step.tool == tool), None)


def _latest_failing_test(steps: Sequence[ExperienceWriteStep]) -> ExperienceWriteStep | None:
    """Latest run_tests step that actually failed (ok=False)."""
    return next(
        (step for step in reversed(steps) if step.tool == "run_tests" and not step.ok),
        None,
    )


def _latest_failing_tool(steps: Sequence[ExperienceWriteStep]) -> ExperienceWriteStep | None:
    """Latest non-test tool step that failed or was blocked."""
    return next(
        (
            step
            for step in reversed(steps)
            if step.tool != "run_tests" and (not step.ok or step.blocked)
        ),
        None,
    )


def _step_command(step: ExperienceWriteStep | None) -> str:
    if step is None:
        return ""
    command = step.arguments.get("command")
    if isinstance(command, str) and command.strip():
        return " ".join(command.split())
    return ""


def _trajectory_content(request: ExperienceWriteRequest) -> str:
    lines = [
        f"Task: {_truncate_text(request.task, 200)}",
        f"Outcome: {request.outcome}",
        "Key steps:",
    ]
    for step in request.steps:
        status = "ok" if step.ok else "failed"
        lines.append(f"{step.step_num}. {step.tool} -> {status}")
    return "\n".join(lines)


def _trajectory_step_payload(step: ExperienceWriteStep) -> ExperienceTrajectoryStep:
    return ExperienceTrajectoryStep(
        step_num=step.step_num,
        observation="",
        action=step.tool,
        action_params=dict(step.arguments),
        result=_truncate_text(step.output, 240),
        reward=1.0 if step.ok else 0.0,
    )


def _contains_forbidden_content(value: str) -> bool:
    text = str(value or "")
    lower = text.casefold()
    if SECRET_PREFIX_RE.search(text):
        return True
    return any(marker in lower for marker in FORBIDDEN_MARKERS)


def _contains_destructive_tool(payload: ToolPayload) -> bool:
    for value in (payload.code, payload.command):
        if isinstance(value, str) and DESTRUCTIVE_COMMAND_RE.search(value):
            return True
    return False


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


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
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3].rstrip() + "..."


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
    "MemoryWriterDatasetLogger",
    "build_write_steps_from_tool_history",
    "proposal_tier_counts",
    "runtime_outcome_from_tool_records",
    "writer_policy_for_result",
]
