from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any

from my_agent.mcp.client import McpClient
from my_agent.mcp.config import McpConfigLoader, McpServerConfig
from my_agent.mcp.protocol import McpToolDescriptor
from my_agent.mcp.server import McpServer, McpServerStatus
from my_agent.mcp.transport import McpTransport, StdioTransport
from my_agent.schema import ToolResult


class McpServerManager:
    def __init__(
        self,
        repo_root: str | Path,
        config: Any | None = None,
        *,
        loader: McpConfigLoader | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        user_config = getattr(config, "mcp_user_config_path", None)
        project_config = getattr(config, "mcp_project_config_path", None)
        if project_config is None and not bool(getattr(config, "mcp_enable_project_servers", True)):
            project_config = self.repo_root / ".paicli" / "__project_mcp_disabled__.json"
        self.loader = loader or McpConfigLoader(
            self.repo_root,
            user_config=user_config,
            project_config=project_config,
        )
        self.servers: dict[str, McpServer] = {}
        self._tool_index: dict[str, tuple[McpServer, McpToolDescriptor]] = {}
        self._lock = threading.RLock()
        self._started = False
        self._closed = False

    def load_configured_servers(self) -> None:
        with self._lock:
            self.servers = {
                name: McpServer(name=name, config=server_config)
                for name, server_config in self.loader.load().items()
            }
            self._tool_index = {}
            self._started = False

    def start_all(self, max_wait_seconds: int | float | None = None) -> None:
        with self._lock:
            if self._started:
                return
            self._closed = False
            if not self.servers:
                self.load_configured_servers()
            targets = list(self.servers.values())
            self._started = True
        if max_wait_seconds is None:
            for server in targets:
                self.start(server.name)
            with self._lock:
                self._rebuild_tool_index()
            return

        threads = self._start_background_workers(targets)
        deadline = time.monotonic() + max(0.0, float(max_wait_seconds))
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        with self._lock:
            self._rebuild_tool_index()

    def _start_background_workers(self, targets: list[McpServer]) -> list[threading.Thread]:
        runnable = [server for server in targets if not server.config.disabled]
        if not runnable:
            return []
        queue: Queue[str] = Queue()
        for server in runnable:
            queue.put(server.name)
        max_workers = _startup_worker_count(len(runnable), self.config)
        threads: list[threading.Thread] = []

        def worker() -> None:
            while True:
                with self._lock:
                    if self._closed:
                        return
                try:
                    name = queue.get_nowait()
                except Empty:
                    return
                try:
                    self.start(name)
                finally:
                    queue.task_done()

        for index in range(max_workers):
            thread = threading.Thread(
                target=worker,
                name=f"agentcli-mcp-startup-{index + 1}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        return threads

    def start(self, name: str) -> None:
        with self._lock:
            if self._closed:
                return
            server = self.servers.get(name)
        if server is None:
            return
        with self._lock:
            if self._closed:
                return
            server.close()
            server.tools = []
            server.error = ""
            if server.config.disabled:
                server.status = McpServerStatus.DISABLED
                return
            server.status = McpServerStatus.STARTING
        try:
            prepared = self.loader.prepare(server.config)
            transport = self._create_transport(prepared)
            client = McpClient(
                name,
                transport,
                initialize_timeout_seconds=int(getattr(self.config, "mcp_initialize_timeout_seconds", 60) or 60),
                call_timeout_seconds=int(getattr(self.config, "mcp_call_timeout_seconds", 60) or 60),
            )
            with self._lock:
                server.client = client
            client.initialize()
            tools = client.list_tools()
            _validate_no_duplicate_tools(name, tools)
            with self._lock:
                if self._closed:
                    client.close()
                    return
                server.tools = tools
                server.started_at = datetime.now()
                server.status = McpServerStatus.READY
                self._rebuild_tool_index()
        except Exception as exc:  # noqa: BLE001 - one MCP server must not break the rest
            with self._lock:
                server.close()
                server.error = str(exc)
                server.status = McpServerStatus.ERROR
                self._rebuild_tool_index()

    def tool_descriptors(self) -> list[McpToolDescriptor]:
        with self._lock:
            return [
                descriptor
                for server in self.servers.values()
                if server.status == McpServerStatus.READY
                for descriptor in server.tools
            ]

    def ready_tools(self) -> list[McpToolDescriptor]:
        return self.tool_descriptors()

    def status_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": server.name,
                    "status": server.status.value,
                    "transport": server.transport_name(),
                    "tools": len(server.tools),
                    "error": server.error,
                    "pid": server.process_id(),
                }
                for server in sorted(self.servers.values(), key=lambda item: item.name)
            ]

    def logs(self, server_name: str) -> list[str]:
        with self._lock:
            server = self.servers.get(server_name)
        if server is None:
            return []
        return server.logs()

    def reload(self, max_wait_seconds: int | float | None = None) -> None:
        self.close()
        self.load_configured_servers()
        self.start_all(max_wait_seconds=max_wait_seconds)

    def call_tool(self, namespaced_name: str, arguments: dict[str, Any]) -> ToolResult:
        with self._lock:
            entry = self._tool_index.get(namespaced_name)
        if entry is None:
            return ToolResult(ok=False, output=f"MCP tool is not available: {namespaced_name}")
        server, descriptor = entry
        if server.status != McpServerStatus.READY or server.client is None:
            return ToolResult(ok=False, output=f"MCP server is not ready: {server.name} ({server.status.value})")
        try:
            return server.client.call_tool(descriptor.name, arguments)
        except TimeoutError as exc:
            return ToolResult(ok=False, output=f"MCP tool call timed out ({server.name}/{descriptor.name}): {exc}", reason="timeout")
        except Exception as exc:  # noqa: BLE001 - tool boundary must return an observation
            return ToolResult(ok=False, output=f"MCP tool call failed ({server.name}/{descriptor.name}): {exc}")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            servers = list(self.servers.values())
            self._tool_index = {}
            self._started = False
        for server in servers:
            server.close()

    def _create_transport(self, config: McpServerConfig) -> McpTransport:
        if config.is_http:
            raise NotImplementedError("Streamable HTTP MCP transport is not implemented in phase 9.1.")
        return StdioTransport(config.command, config.args, config.env or {}, cwd=self.repo_root)

    def _rebuild_tool_index(self) -> None:
        self._tool_index = {
            descriptor.namespaced_name: (server, descriptor)
            for server in self.servers.values()
            if server.status == McpServerStatus.READY
            for descriptor in server.tools
        }


class McpServerManagerPool:
    _lock = threading.Lock()
    _managers: dict[tuple[Path, tuple[object, ...]], McpServerManager] = {}

    @classmethod
    def get(cls, repo_root: str | Path, config: Any | None = None) -> McpServerManager:
        key = (Path(repo_root).resolve(), _config_signature(config))
        should_start = False
        with cls._lock:
            manager = cls._managers.get(key)
            if manager is None:
                manager = McpServerManager(key[0], config)
                cls._managers[key] = manager
                should_start = True
        if should_start:
            manager.start_all(max_wait_seconds=_startup_wait_seconds(config))
        return manager

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            managers = list(cls._managers.values())
            cls._managers = {}
        for manager in managers:
            manager.close()

    @classmethod
    def close_all(cls) -> None:
        cls.clear()


def _validate_no_duplicate_tools(server_name: str, tools: list[McpToolDescriptor]) -> None:
    counts: dict[str, int] = {}
    for tool in tools:
        counts[tool.name] = counts.get(tool.name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"MCP server {server_name} returned duplicate tools: {duplicates}")


def _config_signature(config: Any | None) -> tuple[object, ...]:
    return (
        bool(getattr(config, "mcp_enabled", False)),
        _path_signature(getattr(config, "mcp_user_config_path", None)),
        _path_signature(getattr(config, "mcp_project_config_path", None)),
        bool(getattr(config, "mcp_enable_project_servers", True)),
        int(getattr(config, "mcp_startup_wait_seconds", 8) or 0),
        int(getattr(config, "mcp_max_startup_workers", 8) or 8),
        int(getattr(config, "mcp_initialize_timeout_seconds", 60) or 60),
        int(getattr(config, "mcp_call_timeout_seconds", 60) or 60),
    )


def _path_signature(value: object) -> str:
    if value is None:
        return ""
    return str(Path(value).expanduser())


def _startup_wait_seconds(config: Any | None) -> int:
    value = getattr(config, "mcp_startup_wait_seconds", 8)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 8


def _startup_worker_count(target_count: int, config: Any | None) -> int:
    configured = getattr(config, "mcp_max_startup_workers", 8)
    try:
        parsed = max(1, int(configured))
    except (TypeError, ValueError):
        parsed = 8
    return min(target_count, parsed, 8)


atexit.register(McpServerManagerPool.close_all)
