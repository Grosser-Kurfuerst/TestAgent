from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_agent.schema import ToolResult
from my_agent.tools.hooks import HookViolation, ensure_inside_repo
from my_agent.tools.spec import ToolContext, ToolRegistration, ToolRisk, ToolSource, ToolSpec


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SAFE_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True)
class ConfigToolSource(ToolSource):
    path: Path
    source_name: str = "config"

    @property
    def name(self) -> str:
        return self.source_name

    @classmethod
    def sources_for(cls, repo_root: Path, config: Any | None) -> list["ConfigToolSource"]:
        sources: list[ConfigToolSource] = []
        if bool(getattr(config, "enable_project_tools", False)):
            sources.append(cls(repo_root / ".agentcli" / "tools.json", source_name="config:project"))
        for path in getattr(config, "tool_config_paths", ()):
            sources.append(cls(Path(path).expanduser(), source_name="config:user"))
        return sources

    def load(self, context: ToolContext) -> list[ToolRegistration]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool config is not valid JSON: {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Tool config must contain one JSON object: {self.path}")
        if payload.get("version") != 1:
            raise ValueError(f"Tool config version must be 1: {self.path}")

        tools = payload.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError(f"Tool config 'tools' must be an array: {self.path}")
        registrations: list[ToolRegistration] = []
        for index, raw_tool in enumerate(tools):
            if not isinstance(raw_tool, dict):
                raise ValueError(f"Tool entry #{index + 1} must be an object: {self.path}")
            if raw_tool.get("enabled", True) is False:
                continue
            registrations.append(self._load_command_tool(raw_tool, context))
        return registrations

    def _load_command_tool(self, raw_tool: dict[str, Any], context: ToolContext) -> ToolRegistration:
        if raw_tool.get("kind") != "command":
            raise ValueError(f"Unsupported tool kind {raw_tool.get('kind')!r} in {self.path}")
        name = _required_string(raw_tool, "name", self.path)
        description = _required_string(raw_tool, "description", self.path)
        parameters = raw_tool.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(f"Tool {name} parameters must be an object JSON schema.")
        if parameters.get("additionalProperties") is not False:
            raise ValueError(f"Tool {name} parameters.additionalProperties must be false for command tools.")
        declared_properties = parameters.get("properties")
        if not isinstance(declared_properties, dict):
            raise ValueError(f"Tool {name} parameters.properties must be an object.")
        risk = ToolRisk(raw_tool.get("risk", ToolRisk.EXECUTE.value))

        command = raw_tool.get("command")
        if not isinstance(command, dict):
            raise ValueError(f"Tool {name} command must be an object.")
        argv = command.get("argv")
        if isinstance(argv, str):
            raise ValueError(f"Tool {name} command.argv must be an array; shell strings are not allowed.")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) and part for part in argv):
            raise ValueError(f"Tool {name} command.argv must be a non-empty string array.")
        cwd = command.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ValueError(f"Tool {name} command.cwd must be a non-empty string.")
        timeout_seconds = command.get("timeout_seconds", context.timeout_seconds)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            raise ValueError(f"Tool {name} command.timeout_seconds must be a positive integer.")
        allowed_path_args = command.get("allowed_path_args", [])
        if not isinstance(allowed_path_args, list) or not all(isinstance(item, str) for item in allowed_path_args):
            raise ValueError(f"Tool {name} command.allowed_path_args must be a string array.")
        unknown_path_args = sorted(set(allowed_path_args) - set(declared_properties))
        if unknown_path_args:
            raise ValueError(f"Tool {name} command.allowed_path_args references unknown parameters: {unknown_path_args}")
        template_args = _extract_template_args([*argv, cwd])
        unknown_template_args = sorted(template_args - set(declared_properties))
        if unknown_template_args:
            raise ValueError(f"Tool {name} templates reference unknown parameters: {unknown_template_args}")
        path_args = set(allowed_path_args)
        path_args.update(key for key, schema in declared_properties.items() if _is_path_parameter(key, schema))
        env = command.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError(f"Tool {name} command.env must be a string object.")
        for key in env:
            if not _SAFE_ENV_RE.match(key):
                raise ValueError(f"Tool {name} command.env contains invalid key: {key}")

        spec = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            risk=risk,
            source=self.name,
            timeout_seconds=timeout_seconds,
        )

        def handler(arguments: dict[str, Any], tool_context: ToolContext) -> ToolResult:
            return _run_configured_command(
                tool_name=name,
                repo_root=tool_context.repo_root,
                argv_template=argv,
                cwd_template=cwd,
                timeout_seconds=timeout_seconds,
                path_args=path_args,
                configured_env=env,
                declared_properties=set(declared_properties),
                arguments=arguments,
            )

        def preflight(arguments: dict[str, Any], tool_context: ToolContext) -> None:
            _prepare_configured_command(
                tool_name=name,
                repo_root=tool_context.repo_root,
                argv_template=argv,
                cwd_template=cwd,
                path_args=path_args,
                declared_properties=set(declared_properties),
                arguments=arguments,
            )

        return ToolRegistration(spec=spec, handler=handler, preflight=preflight)


def _required_string(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tool config {path} requires non-empty {key}.")
    return value.strip()


def _run_configured_command(
    *,
    tool_name: str,
    repo_root: Path,
    argv_template: list[str],
    cwd_template: str,
    timeout_seconds: int,
    path_args: set[str],
    configured_env: dict[str, str],
    declared_properties: set[str],
    arguments: dict[str, Any],
) -> ToolResult:
    try:
        sanitized_arguments, argv, cwd = _prepare_configured_command(
            tool_name=tool_name,
            repo_root=repo_root,
            argv_template=argv_template,
            cwd_template=cwd_template,
            path_args=path_args,
            declared_properties=declared_properties,
            arguments=arguments,
        )
    except HookViolation as exc:
        return ToolResult(ok=False, output=str(exc), blocked=True, reason=str(exc))
    except ValueError as exc:
        return ToolResult(ok=False, output=str(exc))

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONUNBUFFERED": "1",
        **configured_env,
    }
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            input=json.dumps(sanitized_arguments, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return ToolResult(
            ok=False,
            output=_format_command_output("timeout", stdout, f"{stderr}\nCommand timed out after {timeout_seconds}s."),
            reason="timeout",
        )
    except OSError as exc:
        return ToolResult(ok=False, output=f"Tool {tool_name} failed to start: {exc}", reason="start_failed")

    return ToolResult(
        ok=completed.returncode == 0,
        output=_format_command_output(completed.returncode, completed.stdout, completed.stderr),
    )


def _prepare_configured_command(
    *,
    tool_name: str,
    repo_root: Path,
    argv_template: list[str],
    cwd_template: str,
    path_args: set[str],
    declared_properties: set[str],
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[str], Path]:
    sanitized_arguments = _sanitize_arguments(
        tool_name=tool_name,
        repo_root=repo_root,
        arguments=arguments,
        declared_properties=declared_properties,
        path_args=path_args,
    )
    argv = [_render_template(part, sanitized_arguments, tool_name) for part in argv_template]
    cwd = ensure_inside_repo(repo_root, _render_template(cwd_template, sanitized_arguments, tool_name))
    return sanitized_arguments, argv, cwd


def _render_template(template: str, arguments: dict[str, Any], tool_name: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in arguments:
            raise ValueError(f"Tool {tool_name} missing template argument: {key}")
        value = arguments[key]
        if isinstance(value, (dict, list)):
            raise ValueError(f"Tool {tool_name} template argument must be scalar: {key}")
        return str(value)

    rendered = _PLACEHOLDER_RE.sub(replace, template)
    if "{" in rendered or "}" in rendered:
        raise ValueError(f"Tool {tool_name} contains unsupported template syntax: {template}")
    return rendered


def _extract_template_args(templates: list[str]) -> set[str]:
    names: set[str] = set()
    for template in templates:
        names.update(_PLACEHOLDER_RE.findall(template))
    return names


def _sanitize_arguments(
    *,
    tool_name: str,
    repo_root: Path,
    arguments: dict[str, Any],
    declared_properties: set[str],
    path_args: set[str],
) -> dict[str, Any]:
    unknown = sorted(set(arguments) - declared_properties)
    if unknown:
        raise ValueError(f"Tool {tool_name} received undeclared arguments: {unknown}")

    sanitized: dict[str, Any] = {}
    for name, value in arguments.items():
        if name in path_args:
            if isinstance(value, (dict, list)):
                raise HookViolation(f"Path argument must be scalar: {name}")
            safe_path = ensure_inside_repo(repo_root, value)
            sanitized[name] = safe_path.relative_to(repo_root).as_posix()
        elif isinstance(value, str) and _looks_like_path_escape(value):
            raise HookViolation(f"Potential path argument escapes repository root: {name}")
        else:
            sanitized[name] = value
    return sanitized


def _is_path_parameter(name: str, schema: Any) -> bool:
    lowered = name.lower()
    if lowered in {"path", "file", "dir", "directory", "cwd", "source", "target"}:
        return True
    if lowered.endswith(("_path", "_file", "_dir", "_directory")):
        return True
    if isinstance(schema, dict):
        description = str(schema.get("description", "")).lower()
        return "path" in description or "file path" in description or "directory" in description
    return False


def _looks_like_path_escape(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if Path(stripped).is_absolute():
        return True
    return stripped == ".." or stripped.startswith("../") or "/../" in stripped or stripped.endswith("/..")


def _format_command_output(exit_status: int | str, stdout: str, stderr: str, limit: int = 12000) -> str:
    output = f"exit_status: {exit_status}\nstdout:\n{stdout.strip()}\nstderr:\n{stderr.strip()}"
    if len(output) <= limit:
        return output
    head = output[: limit // 2]
    tail = output[-limit // 2 :]
    return f"{head}\n... output truncated ...\n{tail}"
