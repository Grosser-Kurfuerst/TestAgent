from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from my_agent.config import AgentConfig
from my_agent.hitl.handler import HitlHandler
from my_agent.llm import AgentLLM
from my_agent.memory import MemoryService
from my_agent.plan import AgentMode, PlanExecuteAgent
from my_agent.react import ReActAgent
from my_agent.runtime.base import AgentBase
from my_agent.team import TeamAgent

EventSink = Callable[[Any], None]


class AgentFactory:
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: AgentLLM,
        trace_dir: str | Path,
        command_timeout: int,
        event_sink: EventSink | None = None,
        memory_manager: MemoryService | None = None,
        hitl_handler: HitlHandler | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.event_sink = event_sink
        self.memory_manager = memory_manager
        self.hitl_handler = hitl_handler

    def create(self, mode: AgentMode) -> AgentBase:
        if mode == AgentMode.TEAM:
            return TeamAgent(
                config=self.config,
                llm=self.llm,
                trace_dir=self.trace_dir,
                command_timeout=self.command_timeout,
                event_sink=self.event_sink,
                memory_manager=self.memory_manager,
                hitl_handler=self.hitl_handler,
            )
        if mode == AgentMode.PLAN:
            return PlanExecuteAgent(
                config=self.config,
                llm=self.llm,
                trace_dir=self.trace_dir,
                command_timeout=self.command_timeout,
                event_sink=self.event_sink,
                memory_manager=self.memory_manager,
                hitl_handler=self.hitl_handler,
            )
        return ReActAgent(
            config=self.config,
            llm=self.llm,
            trace_dir=self.trace_dir,
            command_timeout=self.command_timeout,
            event_sink=self.event_sink,
            memory_manager=self.memory_manager,
            hitl_handler=self.hitl_handler,
        )
