from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from my_agent.config import AgentConfig
from my_agent.llm import AgentLLM
from my_agent.llm.types import Message, MessageLike
from my_agent.memory import MemoryManager
from my_agent.plan import TaskResult
from my_agent.react_runtime import ReActRuntime
from my_agent.schema import AgentState
from my_agent.team.prompts import (
    TEAM_REVIEWER_SYSTEM_PROMPT,
    TEAM_WORKER_SYSTEM_PROMPT,
    build_reviewer_prompt,
    build_worker_prompt,
)
from my_agent.team.reviewer import parse_review_decision
from my_agent.team.types import AgentRole, ExecutionStep, ReviewDecision, TeamState

EventSink = Callable[[Any], None]


class SubAgent:
    def __init__(
        self,
        *,
        name: str,
        role: AgentRole,
        config: AgentConfig,
        llm: AgentLLM,
        repo_path: str | Path,
        trace_dir: str | Path,
        command_timeout: int,
        memory_manager: MemoryManager | None = None,
        event_sink: EventSink | None = None,
        test_command: str | None = None,
        step_max_steps: int | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.config = config
        self.llm = llm
        self.repo_path = Path(repo_path).resolve()
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.memory_manager = memory_manager
        self.event_sink = event_sink
        self.test_command = test_command
        self.step_max_steps = _positive_or_default(step_max_steps, config.team_step_max_steps)
        self.history: list[MessageLike] = [Message(role="system", content=self._system_prompt())]

    def execute_step(
        self,
        state: TeamState,
        step: ExecutionStep,
        context: str,
        feedback: str = "",
    ) -> TaskResult:
        if self.role != AgentRole.WORKER:
            raise ValueError("Only worker sub-agents can execute steps.")

        prompt = build_worker_prompt(
            state.goal,
            step.id,
            step.type.value,
            step.title,
            step.description,
            step.acceptance,
            dependency_context=context,
            feedback=feedback,
            test_command=self.test_command,
        )
        attempt = max(1, step.attempts)
        run_id = f"{state.id}_{step.id}_{attempt}"
        task_memory = (
            self.memory_manager.fork_for_task(session_id=f"{state.id}:{step.id}:{attempt}", run_id=run_id)
            if self.memory_manager is not None
            else None
        )
        agent_state = AgentState.initial(
            repo_path=self.repo_path,
            task=prompt,
            test_command=self.test_command,
            max_steps=self.step_max_steps,
            run_id=run_id,
        )
        final_state = ReActRuntime(
            config=self.config,
            llm=self.llm,
            trace_dir=self.trace_dir / state.id,
            command_timeout=self.command_timeout,
            event_sink=self.event_sink,
            memory_manager=task_memory,
            role_prompt=TEAM_WORKER_SYSTEM_PROMPT,
            run_label="team_worker",
        ).run(agent_state)

        output = final_state.final_answer or final_state.review
        trace_path = str(final_state.trace_path or "")
        if _react_state_succeeded(final_state):
            return TaskResult.success(
                step.id,
                output,
                trace_path=trace_path,
                stop_reason=final_state.stop_reason,
            )
        return TaskResult.failure(
            step.id,
            _react_failure_message(final_state),
            output=output,
            trace_path=trace_path,
            stop_reason=final_state.stop_reason,
        )

    def review_step(self, goal: str, step: ExecutionStep, context: str, result: str) -> ReviewDecision:
        if self.role != AgentRole.REVIEWER:
            raise ValueError("Only reviewer sub-agents can review steps.")

        prompt = build_reviewer_prompt(
            goal,
            step.id,
            step.type.value,
            step.title,
            step.description,
            step.acceptance,
            dependency_context=context,
            result=result,
        )
        self.history.append(Message(role="user", content=prompt))
        response = self.llm.chat(self.history, tools=None)
        self.history.append(Message(role=response.role or "assistant", content=response.content))
        return parse_review_decision(response.content)

    def clear_history(self) -> None:
        self.history = [Message(role="system", content=self._system_prompt())]

    def _system_prompt(self) -> str:
        if self.role == AgentRole.WORKER:
            return TEAM_WORKER_SYSTEM_PROMPT
        if self.role == AgentRole.REVIEWER:
            return TEAM_REVIEWER_SYSTEM_PROMPT
        return "You are a planner sub-agent in a Multi-Agent coding team."


def _react_state_succeeded(state: AgentState) -> bool:
    return state.stop_reason in {"finish_called", "assistant_final"}


def _react_failure_message(state: AgentState) -> str:
    if state.review:
        return state.review
    return f"Team worker failed with stop_reason={state.stop_reason or 'unknown'}."


def _positive_or_default(value: int | None, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return max(1, default)
