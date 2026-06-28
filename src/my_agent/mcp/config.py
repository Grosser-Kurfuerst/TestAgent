from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from dotenv import dotenv_values


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE_KEY_RE = re.compile(r"(authorization|token|api[_-]?key|password|secret|credential)", re.IGNORECASE)


@dataclass(frozen=True)
class McpServerConfig:
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    url: str = ""
    headers: dict[str, str] | None = None
    disabled: bool = False
    parse_error: str = ""

    @property
    def is_stdio(self) -> bool:
        return bool(self.command.strip())

    @property
    def is_http(self) -> bool:
        return bool(self.url.strip())

    @property
    def transport_name(self) -> str:
        return "http" if self.is_http else "stdio"

    def masked(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": list(self.args),
            "env": mask_sensitive_values(self.env or {}),
            "url": self.url,
            "headers": mask_sensitive_values(self.headers or {}),
            "disabled": self.disabled,
        }


class McpConfigLoader:
    def __init__(
        self,
        project_dir: str | Path,
        user_config: str | Path | None = None,
        project_config: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.user_config = Path(user_config).expanduser() if user_config is not None else Path.home() / ".paicli" / "mcp.json"
        self.project_config = (
            Path(project_config).expanduser() if project_config is not None else self.project_dir / ".paicli" / "mcp.json"
        )
        self.environ = dict(os.environ if environ is None else environ)

    def load(self) -> dict[str, McpServerConfig]:
        merged: dict[str, McpServerConfig] = {}
        for path in (self.user_config, self.project_config):
            if not path.exists():
                continue
            merged.update(self._read(path))
        return merged

    def prepare(self, config: McpServerConfig) -> McpServerConfig:
        if config.parse_error:
            raise ValueError(config.parse_error)
        prepared = replace(
            config,
            command=self._expand(config.command),
            args=tuple(self._expand(arg) for arg in config.args),
            env={key: self._expand(value) for key, value in (config.env or {}).items()},
            url=self._expand(config.url),
            headers={key: self._expand(value) for key, value in (config.headers or {}).items()},
        )
        if prepared.is_stdio == prepared.is_http:
            raise ValueError("MCP server must configure exactly one of command or url.")
        return prepared

    def _read(self, path: Path) -> dict[str, McpServerConfig]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"MCP config is not valid JSON: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"MCP config must contain one JSON object: {path}")
        raw_servers = payload.get("mcpServers", {})
        if raw_servers is None:
            return {}
        if not isinstance(raw_servers, dict):
            raise ValueError(f"MCP config mcpServers must be an object: {path}")
        servers: dict[str, McpServerConfig] = {}
        for name, raw_config in raw_servers.items():
            server_name = str(name).strip()
            if not server_name:
                continue
            servers[server_name] = _parse_server_config(raw_config, path=path, server_name=server_name)
        return servers

    def _expand(self, raw: str) -> str:
        if not raw:
            return raw

        def replace_var(match: re.Match[str]) -> str:
            name = match.group(1)
            if name == "PROJECT_DIR":
                return str(self.project_dir)
            if name == "HOME":
                return str(Path.home())
            value = self._lookup_variable(name)
            if value is None or not str(value).strip():
                raise ValueError(f"MCP config references unset variable: {name}")
            return str(value)

        return _VAR_RE.sub(replace_var, raw)

    def _lookup_variable(self, name: str) -> str | None:
        if name in self.environ:
            return self.environ[name]
        for path in (self.project_dir / ".env", Path.home() / ".env"):
            if not path.exists():
                continue
            values = dotenv_values(path)
            value = values.get(name)
            if value:
                return value
        return None


def mask_sensitive_values(values: Mapping[str, str]) -> dict[str, str]:
    return {key: "***" if _SENSITIVE_KEY_RE.search(key) else value for key, value in values.items()}


def _parse_server_config(raw_config: object, *, path: Path, server_name: str) -> McpServerConfig:
    if not isinstance(raw_config, dict):
        return McpServerConfig(parse_error=f"MCP server {server_name} config must be an object: {path}")
    errors: list[str] = []
    command = _optional_string(raw_config.get("command"), "command", errors)
    url = _optional_string(raw_config.get("url"), "url", errors)
    args = _string_tuple(raw_config.get("args", []), "args", errors)
    env = _string_mapping(raw_config.get("env", {}), "env", errors)
    headers = _string_mapping(raw_config.get("headers", {}), "headers", errors)
    disabled = raw_config.get("disabled", False)
    if not isinstance(disabled, bool):
        errors.append("disabled must be a boolean")
        disabled = False
    return McpServerConfig(
        command=command,
        args=args,
        env=env,
        url=url,
        headers=headers,
        disabled=disabled,
        parse_error=f"MCP server {server_name} config is invalid: {', '.join(errors)}" if errors else "",
    )


def _optional_string(value: object, field: str, errors: list[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(f"{field} must be a string")
        return ""
    return value


def _string_tuple(value: object, field: str, errors: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{field} must be a string array")
        return ()
    return tuple(value)


def _string_mapping(value: object, field: str, errors: list[str]) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        errors.append(f"{field} must be a string object")
        return {}
    return dict(value)
