from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from my_agent.react.child_runner import ChildReActRequest, ChildReActRunner
from my_agent.config import AgentConfig
from my_agent.llm import AgentLLM
from my_agent.llm.types import Message, MessageLike
from my_agent.memory import MemoryManager
from my_agent.plan import TaskResult
from my_agent.team.prompts import (
    TEAM_REVIEWER_SYSTEM_PROMPT,
    TEAM_WORKER_SYSTEM_PROMPT,
    build_reviewer_prompt,
    build_worker_prompt,
)
from my_agent.team.reviewer import parse_review_decision
from my_agent.team.types import AgentRole, ExecutionStep, ReviewDecision, TeamState
from my_agent.utils.numbers import positive_or_default

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
        self.step_max_steps = positive_or_default(step_max_steps, config.team_step_max_steps)
        self.child_runner = ChildReActRunner(
            config=config,
            llm=llm,
            command_timeout=command_timeout,
            event_sink=event_sink,
            memory_manager=memory_manager,
        )
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
        return self.child_runner.run(
            ChildReActRequest(
                task_id=step.id,
                repo_path=self.repo_path,
                task=prompt,
                test_command=self.test_command,
                run_id=run_id,
                trace_dir=self.trace_dir / state.id,
                max_steps=self.step_max_steps,
                memory_session_id=f"{state.id}:{step.id}:{attempt}",
                failure_prefix="Team worker failed",
                role_prompt=TEAM_WORKER_SYSTEM_PROMPT,
                run_label="team_worker",
            )
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
