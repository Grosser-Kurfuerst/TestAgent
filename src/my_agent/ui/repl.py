from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import TextIO

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile, ConversationCompactor
from my_agent.llm import build_llm
from my_agent.llm.types import Message
from my_agent.runtime import run_agent
from my_agent.tools import RepoTools, ToolExecutionResult, ToolInvocation
from my_agent.ui.renderer import PlainRenderer, Renderer, StartupInfo


HELP_TEXT = """Commands:
/help             Show this help.
/tools            List enabled tools with source and risk.
/tools reload     Reload tools from configuration and plugins.
/context          Show context budget estimates.
/compact [focus]  Compact session context for the supplied focus.
/clear            Clear session context.
/trace            Show the latest trace path.
/quit             Exit.
"""


class AgentRepl:
    def __init__(
        self,
        *,
        repo_path: str | Path,
        config: AgentConfig,
        trace_dir: str | Path,
        renderer: Renderer | None = None,
        input_stream: TextIO | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.trace_dir = Path(trace_dir)
        self.renderer = renderer or PlainRenderer()
        self.input_stream = input_stream or sys.stdin
        self._tools = RepoTools(self.repo_path, timeout=config.command_timeout, config=config)
        self._profile = ContextProfile.from_config(config)
        self._compactor = ConversationCompactor(self._profile, llm=build_llm(config))
        self._session_messages: list[object] = []
        self._latest_trace: Path | None = None

    def run(self, *, show_banner: bool = True) -> int:
        if show_banner:
            self.renderer.banner(self._startup_info())
        while True:
            try:
                line = self._readline()
            except EOFError:
                return 0
            if line is None:
                return 0
            text = line.strip()
            if not text:
                continue
            if text.startswith("/"):
                if self._handle_command(text):
                    return 0
                continue
            try:
                self._session_messages.append(Message(role="user", content=text))
                state = run_agent(
                    repo_path=self.repo_path,
                    task=text,
                    config=self.config,
                    trace_dir=self.trace_dir,
                    event_sink=self._handle_event,
                )
            except Exception as exc:  # noqa: BLE001 - interactive shell should report and continue
                self.renderer.error(f"Error: {exc}")
                continue
            self._latest_trace = state.trace_path
            self._session_messages.append(Message(role="assistant", content=state.final_answer))
            self.renderer.assistant_delta(state.final_answer)

    def _readline(self) -> str | None:
        if self.input_stream is sys.stdin:
            return input(self.renderer.user_prompt())
        return self.input_stream.readline() or None

    def _handle_command(self, command: str) -> bool:
        if command in {"/quit", "/exit"}:
            return True
        if command == "/help":
            self.renderer.status(HELP_TEXT)
            return False
        if command == "/tools":
            self.renderer.status(self._tools_text())
            return False
        if command == "/tools reload":
            self._tools = RepoTools(self.repo_path, timeout=self.config.command_timeout, config=self.config)
            self.renderer.status(f"Reloaded {len(self._tools.registry.tools)} tools.")
            return False
        if command == "/context":
            self.renderer.status(self._context_text())
            return False
        if command.startswith("/compact"):
            focus = command.removeprefix("/compact").strip()
            result = self._compactor.compact_now(self._session_messages, self._tools.tool_definitions(), focus=focus)
            if result.compacted:
                self.renderer.status(
                    f"Compacted context: {result.before_tokens} -> {result.after_tokens} estimated tokens."
                )
            else:
                self.renderer.status("No conversation history was compacted.")
            return False
        if command == "/clear":
            self._session_messages.clear()
            self.renderer.status("Conversation context cleared.")
            return False
        if command == "/trace":
            self.renderer.status(f"Latest trace: {self._latest_trace or 'none'}")
            return False
        self.renderer.status("Unknown command. Type /help for commands.")
        return False

    def _handle_event(self, event: object) -> None:
        event_name = getattr(event, "event", "")
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            return
        if event_name == "tool.started":
            invocation = ToolInvocation(
                id=str(payload.get("id", "")),
                name=str(payload.get("name", "")),
                arguments_json=str(payload.get("arguments", "{}")),
            )
            self.renderer.tool_call_started(invocation)
        elif event_name == "tool.completed":
            result = ToolExecutionResult(
                id=str(payload.get("id", "")),
                name=str(payload.get("name", "")),
                ok=bool(payload.get("ok")),
                content=str(payload.get("content", "")),
                elapsed_ms=int(payload.get("elapsed_ms", 0) or 0),
                error_code=str(payload.get("error_code", "") or ""),
                retryable=bool(payload.get("retryable")),
                blocked=bool(payload.get("blocked")),
                timed_out=bool(payload.get("timed_out")),
            )
            self.renderer.tool_call_completed(result)
            self._session_messages.append(Message(role="tool", tool_call_id=result.id, content=result.content))

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

    def _tools_text(self) -> str:
        lines = ["name\tsource\trisk\tdescription"]
        for tool in self._tools.registry.tools:
            if not tool.spec.enabled:
                continue
            lines.append(
                "\t".join(
                    [
                        tool.spec.name,
                        tool.spec.source,
                        tool.spec.risk.value,
                        tool.spec.description,
                    ]
                )
            )
        return "\n".join(lines)

    def _context_text(self) -> str:
        definitions = self._tools.tool_definitions()
        estimate = self._compactor.estimate_tokens(self._session_messages, definitions)
        return "\n".join(
            [
                f"system/project: rebuilt per run",
                f"conversation: {estimate} estimated tokens",
                f"tools: {len(definitions)} definitions",
                f"compression trigger: {self._profile.compression_trigger_tokens}",
                f"max tool result chars: {self._profile.max_tool_result_chars}",
            ]
        )


def _tool_summary(tools: RepoTools) -> str:
    counts = Counter(tool.spec.source for tool in tools.registry.tools if tool.spec.enabled)
    if not counts:
        return "0 enabled"
    return ", ".join(f"{count} {source}" for source, count in sorted(counts.items()))
