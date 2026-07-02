from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager, ContextProfile
from my_agent.cancellation import CancellationToken
from my_agent.hitl import SwitchableHitlHandler, TerminalHitlHandler
from my_agent.llm import build_llm
from my_agent.memory import MemoryManager, NoopMemoryManager
from my_agent.mcp.manager import McpServerManager, McpServerManagerPool
from my_agent.mcp.observability import format_mcp_summary
from my_agent.plan import AgentMode, normalize_mode
from my_agent.runtime import run_agent as _default_run_agent
from my_agent.schema import AgentState
from my_agent.text_safety import repair_surrogates
from my_agent.tools import RepoTools
from my_agent.ui.renderer import PlainRenderer, Renderer, StartupInfo
from my_agent.ui.repl.commands import handle_repl_command
from my_agent.ui.repl.events import dispatch_repl_event
from my_agent.ui.repl.status import (
    _discover_test_command,
    _last_memory_prepared_from_trace,
    _tool_summary,
)


class AgentRepl:
    def __init__(
        self,
        *,
        repo_path: str | Path,
        config: AgentConfig,
        trace_dir: str | Path,
        renderer: Renderer | None = None,
        input_stream: TextIO | None = None,
        mode: AgentMode | str | None = AgentMode.AUTO,
        test_command: str | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.trace_dir = Path(trace_dir)
        self.renderer = renderer or PlainRenderer()
        self.input_stream = input_stream or sys.stdin
        self.mode = normalize_mode(mode, default=AgentMode.AUTO)
        self._next_mode: AgentMode | None = None
        self.test_command = test_command or _discover_test_command(self.repo_path)
        self._hitl_handler = SwitchableHitlHandler(
            TerminalHitlHandler(
                enabled=config.hitl_enabled,
                stdin=self.input_stream,
                stdout=_renderer_output(self.renderer),
                before_prompt=self.renderer.reset_between_iterations,
                require_tty=self.input_stream is sys.stdin,
            )
        )
        self._tools = self._load_tools()
        if config.memory_enabled:
            self._memory = MemoryManager.from_config(config=config, llm=build_llm(config), repo_path=self.repo_path)
        else:
            self._memory = NoopMemoryManager(config=config, repo_path=self.repo_path)
        self._profile = getattr(self._memory, "context_profile", ContextProfile.resolve(config, config.model))
        self._context_manager = AgentContextManager(self._profile)
        self._latest_trace: Path | None = None
        self._last_memory_prepared: dict[str, object] | None = None
        self._shutdown_complete = False
        self._run_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentcli-repl")
        self._current_future: Future[AgentState] | None = None
        self._current_token: CancellationToken | None = None

    def run(self, *, show_banner: bool = True) -> int:
        try:
            if show_banner:
                self.renderer.banner(self._startup_info())
            while True:
                try:
                    line = self._readline()
                except EOFError:
                    return 0
                if line is None:
                    return 0
                text = repair_surrogates(line.strip())
                if not text:
                    continue
                self._collect_current_task(wait=False)
                if text.startswith("/"):
                    if self._handle_command(text):
                        return 0
                    continue
                self._run_task(text)
        finally:
            self._shutdown()

    def _readline(self) -> str | None:
        if self.input_stream is sys.stdin:
            return input(self.renderer.user_prompt())
        return self.input_stream.readline() or None

    def _handle_command(self, command: str) -> bool:
        return handle_repl_command(self, command)

    def _shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        try:
            self._cancel_current_task(shutting_down=True)
            self._run_executor.shutdown(wait=False, cancel_futures=True)
            self._memory.extract_facts(reason="session_end")
        finally:
            McpServerManagerPool.close_all()

    def _run_task(self, text: str, *, mode: AgentMode | None = None) -> None:
        self._collect_current_task(wait=False)
        if self._current_future is not None and not self._current_future.done():
            self.renderer.status("Current task is still running; use /cancel or wait.")
            return
        selected_mode = mode or self._next_mode or self.mode
        self._next_mode = None
        token = CancellationToken()
        self._current_token = token
        self._current_future = self._run_executor.submit(self._run_task_worker, text, selected_mode, token)

    def _run_task_worker(self, text: str, selected_mode: AgentMode, token: CancellationToken) -> AgentState:
        return _run_agent(
            repo_path=self.repo_path,
            task=text,
            test_command=self.test_command,
            config=self.config,
            trace_dir=self.trace_dir,
            event_sink=self._handle_event,
            mode=selected_mode,
            memory_manager=self._memory,
            hitl_handler=self._hitl_handler,
            cancellation_token=token,
        )

    def _collect_current_task(self, *, wait: bool) -> None:
        future = self._current_future
        if future is None:
            return
        if not wait and not future.done():
            return
        try:
            state = future.result()
        except Exception as exc:  # noqa: BLE001 - interactive shell should report and continue
            self.renderer.error(f"Error: {exc}")
            self._current_future = None
            self._current_token = None
            return
        self._latest_trace = state.trace_path
        self._last_memory_prepared = _last_memory_prepared_from_trace(state.trace_path)
        self.renderer.assistant_delta(state.final_answer)
        self._current_future = None
        self._current_token = None

    def _cancel_current_task(self, *, shutting_down: bool = False) -> None:
        future = self._current_future
        token = self._current_token
        if future is None or future.done():
            self._collect_current_task(wait=False)
            if not shutting_down:
                self.renderer.status("No running task.")
            return
        if token is not None:
            token.cancel("user_cancelled")
        future.cancel()
        if not shutting_down:
            self.renderer.status("Cancellation requested.")

    def _handle_event(self, event: object) -> None:
        prepared = dispatch_repl_event(event, self.renderer)
        if prepared is not None:
            self._last_memory_prepared = prepared

    def _startup_info(self) -> StartupInfo:
        return StartupInfo(
            version="0.1.0",
            repo_path=self.repo_path,
            provider=self.config.provider,
            model=self.config.model,
            tool_summary=_tool_summary(self._tools),
            trace_dir=self.trace_dir,
            limits=(
                f"max_steps={self.config.max_steps}, timeout={self.config.command_timeout}s, "
                f"context={self._profile.max_context_tokens}"
            ),
        )

    def _load_tools(self) -> RepoTools:
        return RepoTools(
            self.repo_path,
            timeout=self.config.command_timeout,
            config=self.config,
            hitl_handler=self._hitl_handler,
        )

    def _mcp_manager(self) -> McpServerManager:
        if not self.config.mcp_enabled:
            raise RuntimeError("MCP is disabled.")
        return McpServerManagerPool.get(self.repo_path, self.config)

    def _mcp_summary(self) -> str:
        if not self.config.mcp_enabled:
            return "disabled"
        return format_mcp_summary(self._mcp_manager().status_rows())

    def _approval_label(self, *, source: str, risk: str) -> str:
        if risk == "read":
            return "none"
        if not self._hitl_handler.is_enabled():
            return "off"
        if source.startswith("mcp:"):
            return "ask" if self.config.mcp_require_approval else "none"
        if risk == "execute":
            return "ask"
        return self.config.hitl_medium_risk_mode


def _renderer_output(renderer: Renderer) -> TextIO:
    output = getattr(renderer, "output", None)
    if output is not None:
        return output
    return sys.stdout


def _run_agent(**kwargs: object) -> AgentState:
    repl_module = sys.modules.get("my_agent.ui.repl")
    runner = getattr(repl_module, "run_agent", _default_run_agent)
    return runner(**kwargs)
