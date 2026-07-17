"""Decision-level event recording for formal OPD collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4
import json

from my_agent.policy.contracts import (
    DecisionOutputError,
    DecisionRequest,
    DecisionResponse,
    GenerationPolicy,
)
from my_agent.policy.identity import canonical_json_bytes, require_matching_policy_identity
from my_agent.training.contracts import DecisionEvent


TraceSink = Callable[[str, dict[str, Any]], None]
ResponseParser = Callable[[DecisionResponse], Mapping[str, Any]]


@dataclass(frozen=True)
class DecisionEventContext:
    trajectory_id: str
    turn_index: int
    step_index: int
    task_id: str
    task_group: str
    stream_id: str
    memory_project_key: str
    run_id: str
    repository_revision: str
    candidate_snapshot_hash: str


@dataclass(frozen=True)
class LoggedDecision:
    decision_id: str
    response: DecisionResponse


class DecisionAttemptError(RuntimeError):
    """Generation failure that retains the audited decision ID for retries."""

    def __init__(self, decision_id: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.decision_id = decision_id
        self.cause = cause


class DecisionEventWriter:
    """Thread-safe append-only writer for the versioned decision-event stream."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append(self, event: DecisionEvent) -> None:
        payload = canonical_json_bytes(event.to_dict()).decode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")


class DecisionEventRecorder:
    """Generate one policy decision and persist its exact reconstruction data."""

    def __init__(
        self,
        *,
        policy: GenerationPolicy,
        trace_sink: TraceSink | None = None,
        dataset_path: str | Path | None = None,
    ) -> None:
        self.policy = policy
        self.trace_sink = trace_sink
        self.writer = DecisionEventWriter(dataset_path) if dataset_path is not None else None
        self._events_lock = Lock()
        self._events = list(
            load_decision_events(dataset_path)
            if dataset_path is not None and Path(dataset_path).exists()
            else ()
        )

    def bind_dataset_path(self, path: str | Path) -> None:
        target = Path(path)
        if self.writer is not None:
            if self.writer.path != target:
                raise ValueError("decision recorder is already bound to another dataset path")
            return
        with self._events_lock:
            if self._events:
                raise ValueError("cannot bind a populated in-memory decision recorder")
            self._events = list(load_decision_events(target) if target.exists() else ())
            self.writer = DecisionEventWriter(target)

    def events_for(
        self,
        trajectory_id: str,
        *,
        role: str | None = None,
        status: str | None = None,
        purpose: str | None = None,
    ) -> tuple[DecisionEvent, ...]:
        with self._events_lock:
            return tuple(
                event
                for event in self._events
                if event.trajectory_id == trajectory_id
                and (role is None or event.role == role)
                and (status is None or event.status == status)
                and (purpose is None or event.purpose == purpose)
            )

    def generate(
        self,
        request: DecisionRequest,
        *,
        context: DecisionEventContext,
        retry_of: str | None = None,
        parse_response: ResponseParser | None = None,
    ) -> LoggedDecision:
        decision_id = f"dec-{uuid4().hex}"
        rendered_prompt_hash = self._rendered_prompt_hash(request)
        try:
            response = self.policy.generate_decision(request)
            require_matching_policy_identity(self.policy.identity(), response.identity)
        except DecisionOutputError as exc:
            require_matching_policy_identity(self.policy.identity(), exc.response.identity)
            event = self._event(
                request=request,
                context=context,
                decision_id=decision_id,
                retry_of=retry_of,
                rendered_prompt_hash=rendered_prompt_hash,
                response=exc.response,
                parsed_output={
                    "error_type": type(exc.cause).__name__,
                    "error": str(exc.cause),
                },
                status="invalid_output",
            )
            self._append(event)
            raise DecisionAttemptError(decision_id, exc.cause) from exc
        except Exception as exc:  # noqa: BLE001 - failed attempts are part of the audit stream
            event = self._event(
                request=request,
                context=context,
                decision_id=decision_id,
                retry_of=retry_of,
                rendered_prompt_hash=rendered_prompt_hash,
                response=None,
                parsed_output={"error_type": type(exc).__name__, "error": str(exc)},
                status="llm_error",
            )
            self._append(event)
            raise DecisionAttemptError(decision_id, exc) from exc

        try:
            parsed_output = (
                parse_response(response)
                if parse_response is not None
                else {"tool_calls": [item.to_dict() for item in response.parsed_tool_calls]}
            )
            if not isinstance(parsed_output, Mapping):
                raise ValueError("decision response parser must return a mapping")
        except Exception as exc:  # noqa: BLE001 - role schema failures retain exact generation data
            event = self._event(
                request=request,
                context=context,
                decision_id=decision_id,
                retry_of=retry_of,
                rendered_prompt_hash=rendered_prompt_hash,
                response=response,
                parsed_output={"error_type": type(exc).__name__, "error": str(exc)},
                status="invalid_output",
            )
            self._append(event)
            raise DecisionAttemptError(decision_id, exc) from exc

        event = self._event(
            request=request,
            context=context,
            decision_id=decision_id,
            retry_of=retry_of,
            rendered_prompt_hash=rendered_prompt_hash,
            response=response,
            parsed_output=parsed_output,
            status="success",
        )
        self._append(event)
        return LoggedDecision(decision_id=decision_id, response=response)

    def _rendered_prompt_hash(self, request: DecisionRequest) -> str:
        prompt_hash = self.policy.render_prompt_hash(request)
        if not isinstance(prompt_hash, str):
            raise ValueError("formal policy render_prompt_hash() must return a string")
        return prompt_hash

    def _event(
        self,
        *,
        request: DecisionRequest,
        context: DecisionEventContext,
        decision_id: str,
        retry_of: str | None,
        rendered_prompt_hash: str,
        response: DecisionResponse | None,
        parsed_output: Mapping[str, Any],
        status: str,
    ) -> DecisionEvent:
        identity = response.identity if response is not None else self.policy.identity()
        return DecisionEvent(
            role=request.role,
            purpose=request.purpose,
            decision_id=decision_id,
            trajectory_id=context.trajectory_id,
            turn_index=context.turn_index,
            step_index=context.step_index,
            task_id=context.task_id,
            task_group=context.task_group,
            stream_id=context.stream_id,
            memory_project_key=context.memory_project_key,
            run_id=context.run_id,
            policy_identity=identity,
            repository_revision=context.repository_revision,
            candidate_snapshot_hash=context.candidate_snapshot_hash,
            canonical_messages=request.messages,
            canonical_tools=request.tools,
            rendered_prompt_hash=rendered_prompt_hash,
            prompt_token_ids=response.prompt_token_ids if response is not None else (),
            raw_completion=response.raw_completion if response is not None else "",
            completion_token_ids=response.completion_token_ids if response is not None else (),
            assistant_loss_mask=response.assistant_loss_mask if response is not None else (),
            parsed_output=parsed_output,
            retry_of=retry_of,
            status=status,
        )

    def _append(self, event: DecisionEvent) -> None:
        if self.writer is not None:
            self.writer.append(event)
        with self._events_lock:
            self._events.append(event)
        if self.trace_sink is not None:
            self.trace_sink("opd.decision", event.to_dict())


def load_decision_events(path: str | Path) -> tuple[DecisionEvent, ...]:
    events: list[DecisionEvent] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid decision event JSON at line {line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"decision event line {line_number} must be a JSON object")
            events.append(DecisionEvent.from_dict(payload))
    return tuple(events)


__all__ = [
    "DecisionAttemptError",
    "DecisionEventContext",
    "DecisionEventRecorder",
    "DecisionEventWriter",
    "LoggedDecision",
    "load_decision_events",
]
