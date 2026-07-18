from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from my_agent.config import AgentConfig
from my_agent.cancellation import CancellationToken
from my_agent.hitl.handler import HitlHandler
from my_agent.memory.api import MemoryService
from my_agent.plan.types import TaskResult
from my_agent.react.agent import ReActAgent
from my_agent.schema import AgentState


@dataclass(frozen=True)
class ChildReActRequest:
    task_id: str
    repo_path: str | Path
    task: str
    test_command: str | None
    run_id: str
    trace_dir: str | Path
    max_steps: int
    role_prompt: str | None = None
    run_label: str = "native_tool_calls"
    memory_session_id: str = ""
    failure_prefix: str = "ReAct task failed"
    cancellation_token: CancellationToken | None = None


class ChildReActRunner:
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: Any,
        command_timeout: int,
        event_sink: Callable[[object], None] | None = None,
        memory_manager: MemoryService | None = None,
        hitl_handler: HitlHandler | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.command_timeout = command_timeout
        self.event_sink = event_sink
        self.memory_manager = memory_manager
        self.hitl_handler = hitl_handler

    def run(self, request: ChildReActRequest) -> TaskResult:
        task_memory = (
            self.memory_manager.fork_for_task(
                session_id=request.memory_session_id or request.run_id,
                run_id=request.run_id,
            )
            if self.memory_manager is not None
            else None
        )
        state = AgentState.initial(
            repo_path=request.repo_path,
            task=request.task,
            test_command=request.test_command,
            max_steps=request.max_steps,
            run_id=request.run_id,
            cancellation_token=request.cancellation_token,
        )
        final_state = ReActAgent(
            config=self.config,
            llm=self.llm,
            trace_dir=request.trace_dir,
            command_timeout=self.command_timeout,
            event_sink=self.event_sink,
            memory_manager=task_memory,
            hitl_handler=self.hitl_handler,
            role_prompt=request.role_prompt,
            run_label=request.run_label,
        ).run(state)

        output = final_state.final_answer or final_state.review
        trace_path = str(final_state.trace_path or "")
        if _react_state_succeeded(final_state):
            return TaskResult.success(
                request.task_id,
                output,
                trace_path=trace_path,
                stop_reason=final_state.stop_reason,
            )
        return TaskResult.failure(
            request.task_id,
            _react_failure_message(final_state, request.failure_prefix),
            output=output,
            trace_path=trace_path,
            stop_reason=final_state.stop_reason,
        )


def _react_state_succeeded(state: AgentState) -> bool:
    return state.stop_reason in {"finish_called", "assistant_final"}


def _react_failure_message(state: AgentState, failure_prefix: str) -> str:
    if state.review:
        return state.review
    return f"{failure_prefix} with stop_reason={state.stop_reason or 'unknown'}."
