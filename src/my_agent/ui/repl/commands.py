from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from my_agent.context import budget_tool_definitions
from my_agent.llm.types import Message
from my_agent.mcp.observability import format_mcp_disabled, format_mcp_logs, format_mcp_status
from my_agent.memory import MemoryScope
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
        removed, extracted = repl._memory.clear_short_term(extract_first=True, reason="clear_command")
        repl._hitl_handler.clear_approved_all()
        if repl._memory.last_fact_extraction_error:
            repl.renderer.status(
                f"Fact extraction failed; cleared {removed} short-term entries.\n"
                "Cleared HITL approve-all grants."
            )
        else:
            repl.renderer.status(
                f"Extracted {len(extracted)} facts, cleared {removed} short-term entries.\n"
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
    content = command.removeprefix("/save").strip()
    scope = MemoryScope.PROJECT
    if content.startswith("--global"):
        scope = MemoryScope.GLOBAL
        content = content.removeprefix("--global").strip()
    if not content:
        repl.renderer.status("Usage: /save <fact>")
        return
    try:
        entry, created = repl._memory.save_fact(content, scope=scope)
    except Exception as exc:  # noqa: BLE001 - interactive shell should report and continue
        repl.renderer.error(f"Error: {exc}")
        return
    if created:
        repl.renderer.status(f"Saved memory: {entry.id}")
    else:
        repl.renderer.status(f"Memory already exists: {entry.id}")


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
