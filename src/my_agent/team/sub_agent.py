from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from my_agent.cancellation import CancellationToken
from my_agent.react.child_runner import ChildReActRequest, ChildReActRunner
from my_agent.config import AgentConfig
from my_agent.hitl.handler import HitlHandler
from my_agent.llm import AgentLLM
from my_agent.llm.types import Message, MessageLike
from my_agent.memory import MemoryService
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
TraceSink = Callable[[str, dict[str, object]], None]


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
        memory_manager: MemoryService | None = None,
        event_sink: EventSink | None = None,
        hitl_handler: HitlHandler | None = None,
        test_command: str | None = None,
        step_max_steps: int | None = None,
        trace_sink: TraceSink | None = None,
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
        self.hitl_handler = hitl_handler
        self.test_command = test_command
        self.step_max_steps = positive_or_default(step_max_steps, config.team_step_max_steps)
        self.trace_sink = trace_sink
        self.child_runner = ChildReActRunner(
            config=config,
            llm=llm,
            command_timeout=command_timeout,
            event_sink=event_sink,
            memory_manager=memory_manager,
            hitl_handler=hitl_handler,
        )
        self.history: list[MessageLike] = [Message(role="system", content=self._system_prompt())]

    def set_trace_sink(self, trace_sink: TraceSink | None) -> TraceSink | None:
        previous = self.trace_sink
        self.trace_sink = trace_sink
        return previous

    def execute_step(
        self,
        state: TeamState,
        step: ExecutionStep,
        context: str,
        feedback: str = "",
        cancellation_token: CancellationToken | None = None,
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
                cancellation_token=cancellation_token,
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
        self._trace(
            "llm.requested",
            {
                "phase": "team_reviewer",
                "reviewer": self.name,
                "step_id": step.id,
                "message_count": len(self.history),
                "tool_count": 0,
            },
        )
        try:
            response = self.llm.chat(self.history, tools=None)
        except Exception as exc:
            self._trace(
                "llm.failed",
                {
                    "phase": "team_reviewer",
                    "reviewer": self.name,
                    "step_id": step.id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        self._trace(
            "llm.completed",
            {
                "phase": "team_reviewer",
                "reviewer": self.name,
                "step_id": step.id,
                "finish_reason": response.finish_reason,
                "content_chars": len(response.content),
                "tool_calls": [],
                "usage": response.usage.to_dict(),
            },
        )
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

    def _trace(self, event: str, payload: dict[str, object]) -> None:
        if self.trace_sink is None:
            return
        self.trace_sink(event, payload)
