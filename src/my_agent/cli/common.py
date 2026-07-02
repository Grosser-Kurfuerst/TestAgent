from __future__ import annotations

import json
import os
import sys
import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from my_agent.config import AgentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_FILE = PROJECT_ROOT / "examples" / "tasks" / "sample_task.json"


@dataclass(frozen=True)
class CliContext:
    run_agent: Callable[..., Any]
    agent_repl_cls: type[Any]
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)

    def config_from_env(self, *, require_env_file: bool = False) -> AgentConfig:
        return config_from_env(env=self.env, require_env_file=require_env_file)

    def tool_environment_overrides(self) -> dict[str, str]:
        return tool_environment_overrides(self.env)

    def parse_env_overrides(self, items: Sequence[str]) -> dict[str, str]:
        return parse_env_overrides(items)

    def run_event_sink(self, event: object) -> None:
        run_event_sink(event)

    def close_mcp_servers(self) -> None:
        close_mcp_servers()

    def with_hitl_flag(self, config: AgentConfig, enabled: bool | None) -> AgentConfig:
        return with_hitl_flag(config, enabled)

    def resolve_repo_path(self, value: str | Path) -> Path:
        return resolve_repo_path(value)

    def resolve_trace_dir(self, value: str | None, default: Path) -> Path:
        return resolve_trace_dir(value, default)


def load_task(path: str | Path = DEFAULT_TASK_FILE) -> dict[str, Any]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Task file must contain one JSON object.")
    for key in ("repo", "task"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Task file requires a non-empty {key!r} field.")
    return payload


def format_task(task: dict[str, Any]) -> str:
    lines = [
        f"id: {task.get('id', 'unknown')}",
        f"source: {task.get('source', 'local')}",
        f"repo: {task['repo']}",
        f"task: {task['task']}",
        f"test_command: {task.get('test_command') or 'not configured'}",
    ]
    return "\n".join(lines)


def section(title: str, body: str) -> str:
    return f"# {title}\n{body}"


def positive_top_k(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top_k must be >= 1.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("top_k must be >= 1.")
    return parsed


def positive_max_steps(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max_steps must be >= 1.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("max_steps must be >= 1.")
    return parsed


def config_from_env(*, env: Mapping[str, str], require_env_file: bool = False) -> AgentConfig:
    return AgentConfig.from_env(env=tool_environment_overrides(env), require_env_file=require_env_file)


def tool_environment_overrides(env: Mapping[str, str]) -> dict[str, str]:
    keys = {
        "MY_AGENT_LLM_PROVIDER",
        "MY_AGENT_USE_FAKE_LLM",
        "MY_AGENT_API_KEY",
        "OPENAI_API_KEY",
        "MY_AGENT_BASE_URL",
        "OPENAI_BASE_URL",
        "MY_AGENT_MODEL",
        "MY_AGENT_TEMPERATURE",
        "MY_AGENT_MAX_STEPS",
        "MY_AGENT_COMMAND_TIMEOUT",
        "MY_AGENT_TRACE_DIR",
        "AGENTCLI_ENABLE_PROJECT_TOOLS",
        "AGENTCLI_ENABLE_PROJECT_PLUGINS",
        "AGENTCLI_TOOL_CONFIGS",
        "MY_AGENT_ENABLE_PROJECT_TOOLS",
        "MY_AGENT_ENABLE_PROJECT_PLUGINS",
        "MY_AGENT_TOOL_CONFIGS",
        "MY_AGENT_MAX_ITERATIONS",
        "MY_AGENT_MAX_TOOL_CALLS",
        "MY_AGENT_MAX_ELAPSED_SECONDS",
        "MY_AGENT_TOKEN_BUDGET",
        "MY_AGENT_STAGNATION_WINDOW",
        "MY_AGENT_REPEATED_FAILURE_WINDOW",
        "MY_AGENT_CONTEXT_WINDOW",
        "MY_AGENT_RESPONSE_RESERVE_TOKENS",
        "MY_AGENT_COMPRESSION_BUFFER_TOKENS",
        "AGENTCLI_REPO_CONTEXT_BUDGET_TOKENS",
        "MY_AGENT_REPO_CONTEXT_BUDGET_TOKENS",
        "AGENTCLI_TOOL_SCHEMA_BUDGET_TOKENS",
        "MY_AGENT_TOOL_SCHEMA_BUDGET_TOKENS",
        "MY_AGENT_RETAIN_RECENT_TURNS",
        "MY_AGENT_MAX_TOOL_RESULT_CHARS",
        "MY_AGENT_MAX_SUMMARY_INPUT_CHARS",
        "AGENTCLI_PLAN_TASK_MAX_STEPS",
        "AGENTCLI_PLAN_MAX_TASKS",
        "AGENTCLI_PLAN_MAX_REPLANS",
        "AGENTCLI_AGENT_MODE",
        "MY_AGENT_PLAN_TASK_MAX_STEPS",
        "MY_AGENT_PLAN_MAX_TASKS",
        "MY_AGENT_PLAN_MAX_REPLANS",
        "MY_AGENT_AGENT_MODE",
        "AGENTCLI_TEAM_WORKERS",
        "AGENTCLI_TEAM_MAX_STEPS",
        "AGENTCLI_TEAM_MAX_RETRIES",
        "AGENTCLI_TEAM_STEP_MAX_STEPS",
        "AGENTCLI_TEAM_DEPENDENCY_CONTEXT_CHARS",
        "AGENTCLI_TEAM_PARALLEL",
        "AGENTCLI_TEAM_ALLOW_UNAPPROVED_RESULTS",
        "MY_AGENT_TEAM_WORKERS",
        "MY_AGENT_TEAM_MAX_STEPS",
        "MY_AGENT_TEAM_MAX_RETRIES",
        "MY_AGENT_TEAM_STEP_MAX_STEPS",
        "MY_AGENT_TEAM_DEPENDENCY_CONTEXT_CHARS",
        "MY_AGENT_TEAM_PARALLEL",
        "MY_AGENT_TEAM_ALLOW_UNAPPROVED_RESULTS",
        "AGENTCLI_MEMORY",
        "MY_AGENT_MEMORY",
        "AGENTCLI_MEMORY_DIR",
        "MY_AGENT_MEMORY_DIR",
        "AGENTCLI_MEMORY_SHORT_TERM_TOKENS",
        "MY_AGENT_MEMORY_SHORT_TERM_TOKENS",
        "AGENTCLI_MEMORY_SHORT_TERM_ENTRIES",
        "MY_AGENT_MEMORY_SHORT_TERM_ENTRIES",
        "AGENTCLI_MEMORY_CONTEXT_TOKENS",
        "MY_AGENT_MEMORY_CONTEXT_TOKENS",
        "AGENTCLI_MEMORY_RETRIEVAL_LIMIT",
        "MY_AGENT_MEMORY_RETRIEVAL_LIMIT",
        "AGENTCLI_MEMORY_COMPRESSION_TRIGGER_RATIO",
        "MY_AGENT_MEMORY_COMPRESSION_TRIGGER_RATIO",
        "AGENTCLI_MEMORY_RETAIN_RECENT_TURNS",
        "MY_AGENT_MEMORY_RETAIN_RECENT_TURNS",
        "AGENTCLI_MEMORY_MAP_CHUNK_SIZE",
        "MY_AGENT_MEMORY_MAP_CHUNK_SIZE",
        "AGENTCLI_MEMORY_TOOL_RESULT_CHARS",
        "MY_AGENT_MEMORY_TOOL_RESULT_CHARS",
        "AGENTCLI_MEMORY_AUTO_EXTRACT",
        "MY_AGENT_MEMORY_AUTO_EXTRACT",
        "AGENTCLI_HITL",
        "MY_AGENT_HITL",
        "AGENTCLI_HITL_AUDIT_DIR",
        "MY_AGENT_HITL_AUDIT_DIR",
        "AGENTCLI_HITL_NON_INTERACTIVE",
        "MY_AGENT_HITL_NON_INTERACTIVE",
        "AGENTCLI_HITL_MEDIUM_RISK_MODE",
        "MY_AGENT_HITL_MEDIUM_RISK_MODE",
        "AGENTCLI_HITL_LLM_JUDGE",
        "MY_AGENT_HITL_LLM_JUDGE",
        "AGENTCLI_MAX_PARALLEL_TOOLS",
        "MY_AGENT_MAX_PARALLEL_TOOLS",
        "AGENTCLI_TOOL_BATCH_TIMEOUT_SECONDS",
        "MY_AGENT_TOOL_BATCH_TIMEOUT_SECONDS",
        "AGENTCLI_TOOL_SHUTDOWN_GRACE_SECONDS",
        "MY_AGENT_TOOL_SHUTDOWN_GRACE_SECONDS",
        "AGENTCLI_MAX_PROCESS_OUTPUT_CHARS",
        "MY_AGENT_MAX_PROCESS_OUTPUT_CHARS",
        "AGENTCLI_PLAN_PARALLEL",
        "MY_AGENT_PLAN_PARALLEL",
        "AGENTCLI_PLAN_MAX_PARALLEL_TASKS",
        "MY_AGENT_PLAN_MAX_PARALLEL_TASKS",
        "AGENTCLI_PLAN_TASK_BATCH_TIMEOUT_SECONDS",
        "MY_AGENT_PLAN_TASK_BATCH_TIMEOUT_SECONDS",
        "AGENTCLI_TEAM_STEP_BATCH_TIMEOUT_SECONDS",
        "MY_AGENT_TEAM_STEP_BATCH_TIMEOUT_SECONDS",
        "AGENTCLI_MCP",
        "MY_AGENT_MCP",
        "AGENTCLI_MCP_STARTUP_WAIT_SECONDS",
        "MY_AGENT_MCP_STARTUP_WAIT_SECONDS",
        "AGENTCLI_MCP_INITIALIZE_TIMEOUT_SECONDS",
        "MY_AGENT_MCP_INITIALIZE_TIMEOUT_SECONDS",
        "AGENTCLI_MCP_CALL_TIMEOUT_SECONDS",
        "MY_AGENT_MCP_CALL_TIMEOUT_SECONDS",
        "AGENTCLI_MCP_MAX_STARTUP_WORKERS",
        "MY_AGENT_MCP_MAX_STARTUP_WORKERS",
        "AGENTCLI_MCP_REQUIRE_APPROVAL",
        "MY_AGENT_MCP_REQUIRE_APPROVAL",
        "AGENTCLI_MCP_ENABLE_PROJECT_SERVERS",
        "MY_AGENT_MCP_ENABLE_PROJECT_SERVERS",
    }
    return {key: env[key] for key in keys if key in env}


def parse_env_overrides(items: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        if not key.strip():
            raise ValueError("--env key must not be empty.")
        overrides[key.strip()] = value
    return overrides


def run_event_sink(event: object) -> None:
    event_name = getattr(event, "event", "")
    payload = getattr(event, "payload", {})
    if event_name != "tools.schema_capped" or not isinstance(payload, dict):
        return
    print(format_tool_schema_capped_status(payload), file=sys.stderr)


def format_tool_schema_capped_status(payload: dict[str, object]) -> str:
    included = safe_int(payload.get("included_count"))
    omitted = safe_int(payload.get("omitted_count"))
    omitted_names = payload.get("omitted")
    names: list[str] = []
    if isinstance(omitted_names, list):
        names = [str(name) for name in omitted_names[:5]]
    suffix = f": {', '.join(names)}" if names else ""
    return f"Tool schema budget applied: {included} exposed, {omitted} omitted{suffix}."


def safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def close_mcp_servers() -> None:
    from my_agent.mcp.manager import McpServerManagerPool

    McpServerManagerPool.close_all()


def with_hitl_flag(config: AgentConfig, enabled: bool | None) -> AgentConfig:
    if enabled is None:
        return config
    return replace(config, hitl_enabled=enabled)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_trace_dir(value: str | None, default: Path) -> Path:
    path = Path(value) if value is not None else default
    return path if path.is_absolute() else PROJECT_ROOT / path
