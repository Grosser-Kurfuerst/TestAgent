from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.cancellation import CancellationToken
from my_agent.hitl import SwitchableHitlHandler, TerminalHitlHandler
from my_agent.llm import build_llm
from my_agent.llm.types import Message
from my_agent.memory import MemoryManager, MemoryScope
from my_agent.plan import AgentMode, PlanState, PlanTask, normalize_mode
from my_agent.runtime import run_agent
from my_agent.schema import AgentState
from my_agent.team import ExecutionStep, TeamState
from my_agent.text_safety import repair_surrogates
from my_agent.tools import RepoTools, ToolExecutionResult, ToolInvocation
from my_agent.ui.renderer import PlainRenderer, Renderer, StartupInfo


HELP_TEXT = """Commands:
/help             Show this help.
/tools            List enabled tools with source and risk.
/tools reload     Reload tools from configuration and plugins.
/context          Show memory and context budget estimates.
/memory           Show memory system status and long-term entries.
/save <fact>      Save a durable fact to long-term memory.
/compact [focus]  Compact session context for the supplied focus.
/clear            Extract facts, then clear short-term memory.
/trace            Show the latest trace path.
/plan <task>      Run a task with Plan-and-Execute.
/plan             Run the next task with Plan-and-Execute.
/team <task>      Run a task with Multi-Agent team orchestration.
/team             Run the next task with Multi-Agent team orchestration.
/cancel           Request cancellation of the current task.
/hitl             Show HITL approval status.
/hitl on          Enable HITL approvals for this session.
/hitl off         Disable HITL approvals and clear approve-all grants.
/mode <mode>      Set mode: react, plan, team, or auto.
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
        self._profile = ContextProfile.from_config(config)
        self._memory = MemoryManager.from_config(config=config, llm=build_llm(config), repo_path=self.repo_path)
        self._latest_trace: Path | None = None
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
        if command != "/cancel" and self.input_stream is not sys.stdin:
            self._collect_current_task(wait=True)
        if command in {"/quit", "/exit"}:
            self._cancel_current_task(shutting_down=True)
            return True
        if command == "/help":
            self.renderer.status(HELP_TEXT)
            return False
        if command == "/tools":
            self.renderer.status(self._tools_text())
            return False
        if command == "/tools reload":
            self._tools = self._load_tools()
            self.renderer.status(f"Reloaded {len(self._tools.registry.tools)} tools.")
            return False
        if command == "/context":
            self.renderer.status(self._context_text())
            return False
        if command == "/memory":
            self.renderer.status(self._memory_text())
            return False
        if command.startswith("/save"):
            self._handle_save(command)
            return False
        if command.startswith("/compact"):
            focus = command.removeprefix("/compact").strip()
            _, _, result = self._memory.prepare_messages(
                base_messages=[Message(role="system", content="Memory maintenance request.")],
                query=focus or "session context",
                tools=self._tools.tool_definitions(),
                force_compact=True,
                focus=focus,
            )
            if result and result.compacted:
                self.renderer.status(
                    f"Compacted context: {result.before_tokens} -> {result.after_tokens} estimated tokens."
                )
            else:
                self.renderer.status("No conversation history was compacted.")
            return False
        if command == "/clear":
            removed, extracted = self._memory.clear_short_term(extract_first=True, reason="clear_command")
            self._hitl_handler.clear_approved_all()
            if self._memory.last_fact_extraction_error:
                self.renderer.status(
                    f"Fact extraction failed; cleared {removed} short-term entries.\n"
                    "Cleared HITL approve-all grants."
                )
            else:
                self.renderer.status(
                    f"Extracted {len(extracted)} facts, cleared {removed} short-term entries.\n"
                    "Cleared HITL approve-all grants."
                )
            return False
        if command == "/trace":
            self.renderer.status(f"Latest trace: {self._latest_trace or 'none'}")
            return False
        if command == "/cancel":
            self._cancel_current_task()
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
        if command.startswith("/hitl"):
            self._handle_hitl(command)
            return False
        if command.startswith("/plan"):
            task = command.removeprefix("/plan").strip()
            if task:
                self._run_task(task, mode=AgentMode.PLAN)
            else:
                self._next_mode = AgentMode.PLAN
                self.renderer.status("Next task will use Plan-and-Execute.")
            return False
        if command.startswith("/team"):
            task = command.removeprefix("/team").strip()
            if task:
                self._run_task(task, mode=AgentMode.TEAM)
            else:
                self._next_mode = AgentMode.TEAM
                self.renderer.status("Next task will use Multi-Agent team orchestration.")
            return False
        self.renderer.status("Unknown command. Type /help for commands.")
        return False

    def _shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self._cancel_current_task(shutting_down=True)
        self._run_executor.shutdown(wait=False, cancel_futures=True)
        self._memory.extract_facts(reason="session_end")

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
        return run_agent(
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
        elif event_name == "team.started":
            team = _team_from_payload(payload)
            if team is not None:
                self.renderer.team_started(team)
        elif event_name.startswith("team.step."):
            step = _team_step_from_payload(payload)
            if step is not None:
                self.renderer.team_step_updated(step, team_id=str(payload.get("team_id", "")))
        elif event_name in {"team.completed", "team.failed", "team.cancelled", "team.validation_failed"}:
            team = _team_from_payload(payload)
            if team is not None:
                self.renderer.team_completed(team)
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
        elif event_name == "render.flush_requested":
            self.renderer.reset_between_iterations()
        elif event_name == "approval.requested":
            tool_name = str(payload.get("tool_name", ""))
            risk_level = str(payload.get("risk_level", ""))
            self.renderer.status(f"approval requested: {tool_name} {risk_level}")
        elif event_name == "approval.completed":
            tool_name = str(payload.get("tool_name", ""))
            decision = str(payload.get("decision", ""))
            self.renderer.status(f"approval completed: {tool_name} {decision}")
        elif event_name == "approval.audit_failed":
            tool_name = str(payload.get("tool_name", ""))
            error = str(payload.get("error", ""))
            self.renderer.status(f"approval audit failed: {tool_name} {error}")

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
        lines = ["name\tsource\trisk\tapproval\tdescription"]
        for tool in self._tools.registry.tools:
            if not tool.spec.enabled:
                continue
            lines.append(
                "\t".join(
                    [
                        tool.spec.name,
                        tool.spec.source,
                        tool.spec.risk.value,
                        self._approval_label(tool.spec.risk.value),
                        tool.spec.description,
                    ]
                )
            )
        return "\n".join(lines)

    def _context_text(self) -> str:
        status = self._memory.status(include_entries=False)
        return "\n".join(
            [
                f"system/project: rebuilt per run",
                f"short-term: {status.short_term_entries} entries, {status.short_term_tokens} tokens",
                f"short-term limit: {status.short_term_token_limit}",
                f"long-term: {status.long_term_entries} entries, {status.long_term_tokens} tokens",
                f"tools: {len(self._tools.tool_definitions())} definitions",
                f"default test command: {self.test_command or 'not configured'}",
                f"compression trigger: {int(status.short_term_token_limit * status.compression_trigger_ratio)}",
                f"retain recent turns: {status.retain_recent_turns}",
                f"max tool result chars: {self.config.memory_tool_result_chars}",
            ]
        )

    def _memory_text(self) -> str:
        status = self._memory.status(include_entries=True)
        lines = [
            "Memory",
            f"project: {status.project_key}",
            f"storage: {status.storage_path}",
            (
                f"short-term: {status.short_term_entries} entries, {status.short_term_tokens} tokens, "
                f"limit {status.short_term_token_limit}"
            ),
            f"long-term: {status.long_term_entries} entries, {status.long_term_tokens} tokens",
            (
                f"compression: trigger {int(status.compression_trigger_ratio * 100)}%, "
                f"retain {status.retain_recent_turns} turns, map chunk {status.map_chunk_size}"
            ),
            "",
            "Long-term entries:",
        ]
        if not status.long_term_entries_detail:
            lines.append("- none")
            return "\n".join(lines)
        for entry in status.long_term_entries_detail:
            timestamp = entry.created_at.isoformat()
            lines.append(
                f"- {entry.id} [{entry.type.value} {entry.scope.value} {entry.source} {timestamp}] {entry.content}"
            )
        return "\n".join(lines)

    def _handle_save(self, command: str) -> None:
        content = command.removeprefix("/save").strip()
        scope = MemoryScope.PROJECT
        if content.startswith("--global"):
            scope = MemoryScope.GLOBAL
            content = content.removeprefix("--global").strip()
        if not content:
            self.renderer.status("Usage: /save <fact>")
            return
        try:
            entry, created = self._memory.save_fact(content, scope=scope)
        except Exception as exc:  # noqa: BLE001 - interactive shell should report and continue
            self.renderer.error(f"Error: {exc}")
            return
        if created:
            self.renderer.status(f"Saved memory: {entry.id}")
        else:
            self.renderer.status(f"Memory already exists: {entry.id}")

    def _handle_hitl(self, command: str) -> None:
        value = command.removeprefix("/hitl").strip().lower()
        if not value:
            self.renderer.status(
                "HITL approval is "
                f"{'on' if self._hitl_handler.is_enabled() else 'off'} "
                f"(medium={self.config.hitl_medium_risk_mode}, audit={self.config.hitl_audit_dir})."
            )
            return
        if value == "on":
            self.config = replace(self.config, hitl_enabled=True)
            self._hitl_handler.set_enabled(True)
            self.renderer.status("HITL approval enabled.")
            return
        if value == "off":
            self.config = replace(self.config, hitl_enabled=False)
            self._hitl_handler.set_enabled(False)
            self._hitl_handler.clear_approved_all()
            self.renderer.status("HITL approval disabled. Cleared HITL approve-all grants.")
            return
        self.renderer.status("Usage: /hitl [on|off]")

    def _load_tools(self) -> RepoTools:
        return RepoTools(
            self.repo_path,
            timeout=self.config.command_timeout,
            config=self.config,
            hitl_handler=self._hitl_handler,
        )

    def _approval_label(self, risk: str) -> str:
        if risk == "read":
            return "none"
        if not self._hitl_handler.is_enabled():
            return "off"
        if risk == "execute":
            return "ask"
        return self.config.hitl_medium_risk_mode


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


def _team_from_payload(payload: dict[str, object]) -> TeamState | None:
    raw = payload.get("team")
    if not isinstance(raw, dict):
        return None
    return TeamState.from_dict(raw)


def _team_step_from_payload(payload: dict[str, object]) -> ExecutionStep | None:
    raw = payload.get("step")
    if not isinstance(raw, dict):
        return None
    return ExecutionStep.from_dict(raw)


def _renderer_output(renderer: Renderer) -> TextIO:
    output = getattr(renderer, "output", None)
    if output is not None:
        return output
    return sys.stdout


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
