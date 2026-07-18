from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable

from my_agent.config import AgentConfig
from my_agent.hitl.handler import HitlHandler, NonInteractiveHitlHandler, TerminalHitlHandler
from my_agent.llm import AgentLLM, build_llm
from my_agent.memory import MemoryService
from my_agent.observability.tracing import TraceWriter, append_agent_completed
from my_agent.plan import AgentMode, resolve_mode
from my_agent.policy.runtime_validation import require_formal_policy
from my_agent.runtime.cancellation import CancelledError, CancellationToken
from my_agent.runtime.factory import AgentFactory
from my_agent.schema import AgentState, TraceEvent


class CodingAgentRuntime:
    """Facade for the native ReAct + tool-calls runtime."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: AgentLLM | None = None,
        trace_dir: str | Path | None = None,
        command_timeout: int | None = None,
        event_sink: Callable[[Any], None] | None = None,
        memory_manager: MemoryService | None = None,
        hitl_handler: HitlHandler | None = None,
    ):
        self.config = config or AgentConfig.from_env()
        self.llm = llm or build_llm(self.config)
        policy_identity = require_formal_policy(self.config, self.llm)
        if policy_identity is not None and memory_manager is not None:
            validator = getattr(memory_manager, "require_formal_runtime_binding", None)
            if not callable(validator):
                raise ValueError("formal OPD runtime requires a formal MemoryManager binding")
            validator(
                config=self.config,
                policy_identity=policy_identity,
                repo_path=None,
            )
        self.trace_dir = Path(trace_dir) if trace_dir is not None else self.config.trace_dir
        self.command_timeout = command_timeout or self.config.command_timeout
        self.event_sink = event_sink
        self.memory_manager = memory_manager
        self.hitl_handler = hitl_handler if hitl_handler is not None else _default_hitl_handler(self.config)

    def run(self, state: AgentState, *, mode: AgentMode | str | None = None) -> AgentState:
        selected = resolve_mode(mode if mode is not None else self.config.agent_mode, state.task, default=AgentMode.AUTO)
        if not getattr(self.llm, "supports_tools", False):
            raise RuntimeError("The ReAct runtime requires an LLM client with native tool-call support.")
        agent = AgentFactory(
            config=self.config,
            llm=self.llm,
            trace_dir=self.trace_dir,
            command_timeout=self.command_timeout,
            event_sink=self.event_sink,
            memory_manager=self.memory_manager,
            hitl_handler=self.hitl_handler,
        ).create(selected)
        try:
            return agent.run(state)
        except CancelledError as exc:
            return _cancelled_state(state, self.trace_dir, reason=str(exc) or "cancelled")


def run_agent(
    repo_path: str | Path,
    task: str,
    test_command: str | None = None,
    config: AgentConfig | None = None,
    llm: AgentLLM | None = None,
    max_steps: int | None = None,
    trace_dir: str | Path | None = None,
    event_sink: Callable[[Any], None] | None = None,
    mode: AgentMode | str | None = None,
    memory_manager: MemoryService | None = None,
    hitl_handler: HitlHandler | None = None,
    cancellation_token: CancellationToken | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    resolved_config = config or AgentConfig.from_env()
    selected_mode = resolve_mode(mode if mode is not None else resolved_config.agent_mode, task, default=AgentMode.AUTO)
    default_max_steps = resolved_config.team_max_steps if selected_mode == AgentMode.TEAM else resolved_config.max_steps
    state = AgentState.initial(
        repo_path=Path(repo_path).resolve(),
        task=task,
        test_command=test_command,
        max_steps=default_max_steps if max_steps is None else max_steps,
        cancellation_token=cancellation_token,
        metadata=metadata,
    )
    runtime = CodingAgentRuntime(
        config=resolved_config,
        llm=llm,
        trace_dir=trace_dir,
        event_sink=event_sink,
        memory_manager=memory_manager,
        hitl_handler=hitl_handler,
    )
    return runtime.run(state, mode=selected_mode)


def _default_hitl_handler(config: AgentConfig) -> HitlHandler | None:
    if not config.hitl_enabled:
        return None
    if _stdin_is_interactive():
        return TerminalHitlHandler(enabled=True)
    return NonInteractiveHitlHandler(enabled=True)


def _stdin_is_interactive() -> bool:
    isatty = getattr(sys.stdin, "isatty", None)
    return bool(isatty is not None and isatty())


def _cancelled_state(state: AgentState, trace_dir: Path, *, reason: str) -> AgentState:
    state.done = True
    state.stop_reason = "cancelled"
    state.final_answer = "Cancelled."
    state.review = f"Run cancelled: {reason or 'cancelled'}."
    writer = TraceWriter(state.trace_path) if state.trace_path is not None else TraceWriter.create(trace_dir, state.run_id)
    state.trace_path = writer.path
    payload = {"reason": reason or "cancelled"}
    writer.append(TraceEvent(event="run.cancelled", payload=payload, run_id=state.run_id))
    writer.append(
        TraceEvent(
            event="run.completed",
            payload={
                "stop_reason": state.stop_reason,
                "done": state.done,
                "review": state.review,
                "final_answer": state.final_answer,
            },
            run_id=state.run_id,
        )
    )
    append_agent_completed(writer, state, mode="react", run_label="runtime_cancelled")
    return state
