from __future__ import annotations

import shlex
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from my_agent.context import budget_tool_definitions
from my_agent.llm.types import Message
from my_agent.mcp.observability import format_mcp_disabled, format_mcp_logs, format_mcp_status
from my_agent.memory import ExperienceTier, MemoryScope, SkillPayload, TipPayload
from my_agent.plan import AgentMode, normalize_mode
from my_agent.ui.repl.status import format_context_text, format_memory_text, format_tools_text

if TYPE_CHECKING:
    from my_agent.ui.repl.session import AgentRepl


HELP_TEXT = """Commands:
/help             Show this help.
/tools            List enabled tools with source and risk.
/tools reload     Reload tools from configuration and plugins.
/mcp              Show MCP server status.
/mcp status       Show MCP server status.
/mcp logs <name>  Show recent MCP server stderr.
/mcp reload       Reload MCP servers and tools.
/context          Show memory and context budget estimates.
/memory           Show memory system status and long-term entries.
/save ...         Save a typed tip or skill to long-term memory; use /save --tier ... for syntax.
/compact [focus]  Compact session context for the supplied focus.
/clear            Clear short-term memory.
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

SAVE_USAGE = """Usage:
  /save [--global] --tier tip --category <category> --severity <info|warning|critical> --trigger <trigger> <content>
  /save [--global] --tier skill --category <category> --technique <technique> --step <step> [--step <step> ...] <content>

Quote option values that contain spaces. Only tip and skill can be saved from the text command.
"""


def handle_repl_command(repl: "AgentRepl", command: str) -> bool:
    if command != "/cancel" and repl.input_stream is not sys.stdin:
        repl._collect_current_task(wait=True)
    if command in {"/quit", "/exit"}:
        repl._cancel_current_task(shutting_down=True)
        return True
    if command == "/help":
        repl.renderer.status(HELP_TEXT)
        return False
    if command == "/tools":
        repl.renderer.status(format_tools_text(repl._tools, repl._approval_label))
        return False
    if command == "/tools reload":
        repl._tools = repl._load_tools()
        repl.renderer.status(f"Reloaded {len(repl._tools.registry.tools)} tools.")
        return False
    if command == "/mcp" or command.startswith("/mcp "):
        _handle_mcp(repl, command)
        return False
    if command == "/context":
        repl.renderer.status(
            format_context_text(
                memory=repl._memory,
                profile=repl._profile,
                tools=repl._tools,
                latest_trace=repl._latest_trace,
                last_memory_prepared=repl._last_memory_prepared,
                mcp_summary=repl._mcp_summary(),
                test_command=repl.test_command,
                last_evolver_candidates=repl._last_evolver_candidates,
                last_evolver_selected=repl._last_evolver_selected,
            )
        )
        return False
    if command == "/memory":
        repl.renderer.status(format_memory_text(repl._memory))
        return False
    if command.startswith("/save"):
        _handle_save(repl, command)
        return False
    if command.startswith("/compact"):
        _handle_compact(repl, command)
        return False
    if command == "/clear":
        removed, _ = repl._memory.clear_short_term(extract_first=True, reason="clear_command")
        repl._hitl_handler.clear_approved_all()
        repl.renderer.status(
            f"Cleared {removed} short-term entries.\n"
            "Cleared HITL approve-all grants."
        )
        return False
    if command == "/trace":
        repl.renderer.status(f"Latest trace: {repl._latest_trace or 'none'}")
        return False
    if command == "/cancel":
        repl._cancel_current_task()
        return False
    if command.startswith("/mode"):
        _handle_mode(repl, command)
        return False
    if command.startswith("/hitl"):
        _handle_hitl(repl, command)
        return False
    if command.startswith("/plan"):
        task = command.removeprefix("/plan").strip()
        if task:
            repl._run_task(task, mode=AgentMode.PLAN)
        else:
            repl._next_mode = AgentMode.PLAN
            repl.renderer.status("Next task will use Plan-and-Execute.")
        return False
    if command.startswith("/team"):
        task = command.removeprefix("/team").strip()
        if task:
            repl._run_task(task, mode=AgentMode.TEAM)
        else:
            repl._next_mode = AgentMode.TEAM
            repl.renderer.status("Next task will use Multi-Agent team orchestration.")
        return False
    repl.renderer.status("Unknown command. Type /help for commands.")
    return False


def _handle_compact(repl: "AgentRepl", command: str) -> None:
    focus = command.removeprefix("/compact").strip()
    tool_budget = budget_tool_definitions(repl._tools.tool_definitions(), repl._profile)
    _, _, result = repl._context_manager.prepare_messages(
        base_messages=[Message(role="system", content="Memory maintenance request.")],
        query=focus or "session context",
        tools=tool_budget.definitions,
        memory=repl._memory,
        force_compact=True,
        focus=focus,
        tool_budget=tool_budget,
    )
    if result and result.compacted:
        repl.renderer.status(f"Compacted context: {result.before_tokens} -> {result.after_tokens} estimated tokens.")
    else:
        repl.renderer.status("No conversation history was compacted.")


def _handle_mcp(repl: "AgentRepl", command: str) -> None:
    parts = command.split(maxsplit=2)
    action = parts[1].lower() if len(parts) > 1 else "status"
    if not repl.config.mcp_enabled:
        if action == "logs" and len(parts) >= 3 and parts[2].strip():
            repl.renderer.status(format_mcp_logs(parts[2].strip(), ["MCP is disabled."]))
            return
        repl.renderer.status(format_mcp_disabled())
        return
    if action == "status":
        repl.renderer.status(format_mcp_status(repl._mcp_manager().status_rows()))
        return
    if action == "logs":
        if len(parts) < 3 or not parts[2].strip():
            repl.renderer.status("Usage: /mcp logs <server>")
            return
        server_name = parts[2].strip()
        repl.renderer.status(format_mcp_logs(server_name, repl._mcp_manager().logs(server_name)))
        return
    if action == "reload":
        manager = repl._mcp_manager()
        manager.reload(max_wait_seconds=repl.config.mcp_startup_wait_seconds)
        repl._hitl_handler.clear_approved_all()
        repl._tools = repl._load_tools()
        repl.renderer.status(
            "Reloaded MCP servers and tools. Cleared HITL approve-all grants.\n"
            f"{format_mcp_status(manager.status_rows())}"
        )
        return
    repl.renderer.status("Usage: /mcp [status|logs <server>|reload]")


def _handle_save(repl: "AgentRepl", command: str) -> None:
    try:
        tokens = shlex.split(command.removeprefix("/save"))
    except ValueError as exc:
        repl.renderer.error(f"Error: {exc}\n{SAVE_USAGE}")
        return

    scope = MemoryScope.PROJECT
    options: dict[str, list[str]] = {}
    content_parts: list[str] = []
    option_names = {"--tier", "--category", "--severity", "--trigger", "--technique", "--step", "--precondition"}
    single_value_options = option_names - {"--step", "--precondition"}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--global":
            scope = MemoryScope.GLOBAL
            index += 1
            continue
        if token.startswith("--"):
            if token not in option_names or index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                repl.renderer.status(SAVE_USAGE)
                return
            if token in single_value_options and token in options:
                repl.renderer.status(SAVE_USAGE)
                return
            options.setdefault(token, []).append(tokens[index + 1])
            index += 2
            continue
        content_parts.append(token)
        index += 1

    tier_values = options.get("--tier", [])
    if not tier_values:
        repl.renderer.status("Usage: /save requires --tier.\n" + SAVE_USAGE)
        return
    tier_name = tier_values[0].casefold()
    content = " ".join(content_parts).strip()
    if not content:
        repl.renderer.status(SAVE_USAGE)
        return

    try:
        if tier_name == ExperienceTier.TIP.value:
            required = ("--category", "--severity", "--trigger")
            if any(not options.get(name) for name in required):
                repl.renderer.status(SAVE_USAGE)
                return
            tier = ExperienceTier.TIP
            payload = TipPayload(
                category=options["--category"][0],
                severity=options["--severity"][0],
                trigger=options["--trigger"][0],
            )
        elif tier_name == ExperienceTier.SKILL.value:
            required = ("--category", "--technique", "--step")
            if any(not options.get(name) for name in required):
                repl.renderer.status(SAVE_USAGE)
                return
            tier = ExperienceTier.SKILL
            payload = SkillPayload(
                category=options["--category"][0],
                technique=options["--technique"][0],
                preconditions=tuple(options.get("--precondition", [])),
                steps=tuple(options["--step"]),
            )
        else:
            repl.renderer.status("Usage: /save supports only --tier tip or --tier skill.\n" + SAVE_USAGE)
            return
        entry, created = repl._memory.save_experience(
            tier=tier,
            content=content,
            payload=payload,
            scope=scope,
        )
    except Exception as exc:  # noqa: BLE001 - interactive shell should report and continue
        repl.renderer.error(f"Error: {exc}")
        return
    if created:
        repl.renderer.status(f"Saved {entry.tier.value} experience: {entry.id}")
    else:
        repl.renderer.status(f"Experience already exists: {entry.id}")


def _handle_hitl(repl: "AgentRepl", command: str) -> None:
    value = command.removeprefix("/hitl").strip().lower()
    if not value:
        repl.renderer.status(
            "HITL approval is "
            f"{'on' if repl._hitl_handler.is_enabled() else 'off'} "
            f"(medium={repl.config.hitl_medium_risk_mode}, audit={repl.config.hitl_audit_dir})."
        )
        return
    if value == "on":
        repl.config = replace(repl.config, hitl_enabled=True)
        repl._hitl_handler.set_enabled(True)
        repl.renderer.status("HITL approval enabled.")
        return
    if value == "off":
        repl.config = replace(repl.config, hitl_enabled=False)
        repl._hitl_handler.set_enabled(False)
        repl._hitl_handler.clear_approved_all()
        repl.renderer.status("HITL approval disabled. Cleared HITL approve-all grants.")
        return
    repl.renderer.status("Usage: /hitl [on|off]")


def _handle_mode(repl: "AgentRepl", command: str) -> None:
    value = command.removeprefix("/mode").strip()
    if not value:
        repl.renderer.status(f"Current mode: {repl.mode.value}")
        return
    try:
        repl.mode = normalize_mode(value, default=repl.mode)
    except ValueError as exc:
        repl.renderer.error(f"Error: {exc}")
        return
    repl.renderer.status(f"Mode set to {repl.mode.value}.")
