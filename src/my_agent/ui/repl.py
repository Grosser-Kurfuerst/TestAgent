from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import TextIO

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile, ConversationCompactor
from my_agent.llm import build_llm
from my_agent.llm.types import Message
from my_agent.plan import AgentMode, PlanState, PlanTask, normalize_mode
from my_agent.runtime import run_agent
from my_agent.text_safety import repair_surrogates
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
/plan <task>      Run a task with Plan-and-Execute.
/plan             Run the next task with Plan-and-Execute.
/mode <mode>      Set mode: react, plan, or auto.
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
            text = repair_surrogates(line.strip())
            if not text:
                continue
            if text.startswith("/"):
                if self._handle_command(text):
                    return 0
                continue
            self._run_task(text)

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
        if command.startswith("/mode"):
            value = command.removeprefix("/mode").strip()
            if not value:
                self.renderer.status(f"Current mode: {self.mode.value}")
                return False
            try:
                self.mode = normalize_mode(value, default=self.mode)
            except ValueError as exc:
                self.renderer.error(f"Error: {exc}")
                return False
            self.renderer.status(f"Mode set to {self.mode.value}.")
            return False
        if command.startswith("/plan"):
            task = command.removeprefix("/plan").strip()
            if task:
                self._run_task(task, mode=AgentMode.PLAN)
            else:
                self._next_mode = AgentMode.PLAN
                self.renderer.status("Next task will use Plan-and-Execute.")
            return False
        self.renderer.status("Unknown command. Type /help for commands.")
        return False

    def _run_task(self, text: str, *, mode: AgentMode | None = None) -> None:
        selected_mode = mode or self._next_mode or self.mode
        self._next_mode = None
        try:
            self._session_messages.append(Message(role="user", content=text))
            state = run_agent(
                repo_path=self.repo_path,
                task=text,
                test_command=self.test_command,
                config=self.config,
                trace_dir=self.trace_dir,
                event_sink=self._handle_event,
                mode=selected_mode,
            )
        except Exception as exc:  # noqa: BLE001 - interactive shell should report and continue
            self.renderer.error(f"Error: {exc}")
            return
        self._latest_trace = state.trace_path
        self._session_messages.append(Message(role="assistant", content=state.final_answer))
        self.renderer.assistant_delta(state.final_answer)

    def _handle_event(self, event: object) -> None:
        event_name = getattr(event, "event", "")
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            return
        if event_name == "plan.started":
            plan = _plan_from_payload(payload)
            if plan is not None:
                self.renderer.plan_started(plan)
        elif event_name.startswith("plan.task."):
            task = _task_from_payload(payload)
            if task is not None:
                self.renderer.plan_task_updated(task, plan_id=str(payload.get("plan_id", "")))
        elif event_name in {"plan.completed", "plan.failed", "plan.cancelled", "plan.validation_failed"}:
            plan = _plan_from_payload(payload)
            if plan is not None:
                self.renderer.plan_completed(plan)
        elif event_name == "tool.started":
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
                f"default test command: {self.test_command or 'not configured'}",
                f"compression trigger: {self._profile.compression_trigger_tokens}",
                f"max tool result chars: {self._profile.max_tool_result_chars}",
            ]
        )


def _plan_from_payload(payload: dict[str, object]) -> PlanState | None:
    raw = payload.get("plan")
    if not isinstance(raw, dict):
        return None
    return PlanState.from_dict(raw)


def _task_from_payload(payload: dict[str, object]) -> PlanTask | None:
    raw = payload.get("task")
    if not isinstance(raw, dict):
        return None
    return PlanTask.from_dict(raw)


def _discover_test_command(repo_path: Path) -> str | None:
    for filename in ("AGENT.md", "AGENTS.md"):
        path = repo_path / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        command = _first_backticked_test_command(text)
        if command:
            return command
    tests_dir = repo_path / "tests"
    if tests_dir.exists() and any(tests_dir.glob("test*.py")):
        return "python -m unittest discover -s tests -q"
    return None


def _first_backticked_test_command(text: str) -> str | None:
    parts = text.split("`")
    for index in range(1, len(parts), 2):
        command = parts[index].strip()
        if _looks_like_test_command(command):
            return command
    return None


def _looks_like_test_command(command: str) -> bool:
    normalized = " ".join(command.split())
    return (
        normalized.startswith("pytest")
        or normalized.startswith("python -m pytest")
        or normalized.startswith("python -m unittest")
        or normalized in {"npm test", "pnpm test", "yarn test"}
        or normalized.startswith("npm run test")
    )


def _tool_summary(tools: RepoTools) -> str:
    counts = Counter(tool.spec.source for tool in tools.registry.tools if tool.spec.enabled)
    if not counts:
        return "0 enabled"
    return ", ".join(f"{count} {source}" for source, count in sorted(counts.items()))
