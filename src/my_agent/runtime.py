from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from my_agent.config import AgentConfig
from my_agent.llm import AgentLLM, build_llm
from my_agent.memory import MemoryManager
from my_agent.plan import AgentMode, PlanExecuteAgent, resolve_mode
from my_agent.react_runtime import ReActRuntime
from my_agent.schema import AgentState


class CodingAgentRuntime:
    """Facade for the native ReAct + tool-calls runtime."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: AgentLLM | None = None,
        trace_dir: str | Path | None = None,
        command_timeout: int | None = None,
        event_sink: Callable[[Any], None] | None = None,
        memory_manager: MemoryManager | None = None,
    ):
        self.config = config or AgentConfig.from_env()
        self.llm = llm or build_llm(self.config)
        self.trace_dir = Path(trace_dir) if trace_dir is not None else self.config.trace_dir
        self.command_timeout = command_timeout or self.config.command_timeout
        self.event_sink = event_sink
        self.memory_manager = memory_manager

    def run(self, state: AgentState, *, mode: AgentMode | str | None = None) -> AgentState:
        selected = resolve_mode(mode, state.task, default=AgentMode.REACT)
        if selected == AgentMode.TEAM:
            raise RuntimeError("team mode not implemented")
        if not getattr(self.llm, "supports_tools", False):
            raise RuntimeError("The ReAct runtime requires an LLM client with native tool-call support.")
        if selected == AgentMode.PLAN:
            return PlanExecuteAgent(
                config=self.config,
                llm=self.llm,
                trace_dir=self.trace_dir,
                command_timeout=self.command_timeout,
                event_sink=self.event_sink,
            ).run(
                repo_path=state.repo_path,
                goal=state.task,
                test_command=state.test_command,
                max_steps=state.max_steps,
                memory_manager=self.memory_manager,
            )
        return ReActRuntime(
            config=self.config,
            llm=self.llm,
            trace_dir=self.trace_dir,
            command_timeout=self.command_timeout,
            event_sink=self.event_sink,
            memory_manager=self.memory_manager,
        ).run(state)


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
    memory_manager: MemoryManager | None = None,
) -> AgentState:
    resolved_config = config or AgentConfig.from_env()
    state = AgentState.initial(
        repo_path=Path(repo_path).resolve(),
        task=task,
        test_command=test_command,
        max_steps=resolved_config.max_steps if max_steps is None else max_steps,
    )
    runtime = CodingAgentRuntime(
        config=resolved_config,
        llm=llm,
        trace_dir=trace_dir,
        event_sink=event_sink,
        memory_manager=memory_manager,
    )
    return runtime.run(state, mode=mode)
