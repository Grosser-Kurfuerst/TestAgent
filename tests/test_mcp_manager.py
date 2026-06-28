from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import tempfile
import threading
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.mcp.manager import McpServerManager, McpServerManagerPool, _startup_worker_count
from my_agent.mcp.server import McpServerStatus
from my_agent.tools import RepoTools, ToolInvocation, ToolRisk


class McpServerManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        McpServerManagerPool.clear()

    def tearDown(self) -> None:
        McpServerManagerPool.clear()

    def test_fake_stdio_server_initializes_lists_and_calls_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = _write_fake_mcp_server(repo)
            config_path = _write_mcp_config(repo, {"fake": _server_config(script)})
            config = _test_config(repo, config_path)
            manager = McpServerManager(repo, config)

            try:
                manager.start_all()
                result = manager.call_tool("mcp__fake__echo", {"message": "hello"})
            finally:
                manager.close()

        self.assertTrue(result.ok, result.output)
        self.assertEqual(result.output, "hello")

    def test_single_server_failure_does_not_block_other_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = _write_fake_mcp_server(repo)
            config_path = _write_mcp_config(
                repo,
                {
                    "fake": _server_config(script),
                    "bad": {"command": "${MISSING_MCP_VAR}"},
                },
            )
            config = _test_config(repo, config_path)
            manager = McpServerManager(repo, config)

            try:
                manager.start_all()
                ready_tools = [tool.namespaced_name for tool in manager.tool_descriptors()]
                fake_status = manager.servers["fake"].status
                bad_status = manager.servers["bad"].status
            finally:
                manager.close()

        self.assertEqual(fake_status, McpServerStatus.READY)
        self.assertEqual(bad_status, McpServerStatus.ERROR)
        self.assertEqual(ready_tools, ["mcp__fake__echo"])

    def test_start_all_starts_multiple_servers_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = _write_fake_mcp_server(repo)
            config_path = _write_mcp_config(
                repo,
                {
                    "one": _server_config(script, env={"STARTUP_DELAY": "1"}),
                    "two": _server_config(script, env={"STARTUP_DELAY": "1"}),
                },
            )
            config = _test_config(repo, config_path)
            manager = McpServerManager(repo, config)

            started_at = time.monotonic()
            try:
                manager.start_all()
                elapsed = time.monotonic() - started_at
                rows = manager.status_rows()
            finally:
                manager.close()

        self.assertLess(elapsed, 1.8)
        self.assertEqual({row["name"]: row["status"] for row in rows}, {"one": "ready", "two": "ready"})

    def test_http_server_initializes_lists_calls_and_closes(self) -> None:
        server, thread, url = _start_http_mcp_server()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config_path = _write_mcp_config(
                repo,
                {
                    "http": {
                        "url": url,
                        "headers": {"Authorization": "Bearer test-token"},
                    }
                },
            )
            config = _test_config(repo, config_path)
            manager = McpServerManager(repo, config)
            try:
                manager.start_all()
                result = manager.call_tool("mcp__http__echo", {"message": "via-http"})
                rows = manager.status_rows()
            finally:
                manager.close()
                _stop_http_mcp_server(server, thread)

        self.assertTrue(result.ok, result.output)
        self.assertEqual(result.output, "via-http")
        self.assertEqual(rows[0]["transport"], "http")
        self.assertEqual(server.session_headers[-1], "session-http")  # type: ignore[attr-defined]
        self.assertTrue(server.delete_seen.is_set())  # type: ignore[attr-defined]

    def test_initialize_failure_closes_started_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "pid.txt"
            script = _write_nonresponsive_mcp_server(repo, marker)
            config_path = _write_mcp_config(repo, {"bad": _server_config(script)})
            config = _test_config(repo, config_path)
            config.mcp_initialize_timeout_seconds = 1
            manager = McpServerManager(repo, config)

            manager.start_all()

            pid = int(marker.read_text(encoding="utf-8"))
            self.assertEqual(manager.servers["bad"].status, McpServerStatus.ERROR)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_repo_tools_registers_mcp_tool_with_external_risk_and_executes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = _write_fake_mcp_server(repo)
            config_path = _write_mcp_config(repo, {"fake": _server_config(script)})
            config = _test_config(repo, config_path)
            tools = RepoTools(repo, config=config)

            result = tools.execute([ToolInvocation.from_arguments("mcp__fake__echo", {"message": "hi"})])[0]
            registered = tools.registry.get_registered("mcp__fake__echo")

        self.assertIsNotNone(registered)
        self.assertEqual(registered.spec.source, "mcp:fake")
        self.assertEqual(registered.spec.risk, ToolRisk.EXTERNAL)
        self.assertTrue(result.ok, result.content)
        self.assertEqual(result.content, "hi")

    def test_repo_tools_reuses_manager_for_same_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "starts.txt"
            script = _write_fake_mcp_server(repo)
            config_path = _write_mcp_config(repo, {"fake": _server_config(script, env={"START_MARKER": str(marker)})})
            config = _test_config(repo, config_path)

            first = RepoTools(repo, config=config)
            second = RepoTools(repo, config=config)

            self.assertIn("mcp__fake__echo", first.tool_names)
            self.assertIn("mcp__fake__echo", second.tool_names)
            self.assertEqual(marker.read_text(encoding="utf-8"), "1")

    def test_concurrent_repo_tools_reuse_and_wait_for_same_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "starts.txt"
            script = _write_fake_mcp_server(repo)
            config_path = _write_mcp_config(
                repo,
                {"fake": _server_config(script, env={"START_MARKER": str(marker), "STARTUP_DELAY": "0.5"})},
            )
            config = _test_config(repo, config_path)
            results: list[bool] = []
            lock = threading.Lock()

            def load_tools() -> None:
                tools = RepoTools(repo, config=config)
                with lock:
                    results.append("mcp__fake__echo" in tools.tool_names)

            threads = [threading.Thread(target=load_tools) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(results, [True, True, True])
            self.assertEqual(marker.read_text(encoding="utf-8"), "1")

    def test_public_status_logs_reload_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "starts.txt"
            script = _write_fake_mcp_server(repo)
            config_path = _write_mcp_config(repo, {"fake": _server_config(script, env={"START_MARKER": str(marker)})})
            config = _test_config(repo, config_path)
            manager = McpServerManager(repo, config)

            try:
                manager.start_all()
                ready = manager.ready_tools()
                rows = manager.status_rows()
                logs = manager.logs("fake")
                manager.reload()
                reloaded_ready = manager.ready_tools()
                marker_text = marker.read_text(encoding="utf-8")
            finally:
                manager.close()

        self.assertEqual([tool.namespaced_name for tool in ready], ["mcp__fake__echo"])
        self.assertEqual([tool.namespaced_name for tool in reloaded_ready], ["mcp__fake__echo"])
        self.assertEqual(rows[0]["name"], "fake")
        self.assertEqual(rows[0]["status"], McpServerStatus.READY.value)
        self.assertEqual(rows[0]["tools"], 1)
        self.assertIsInstance(logs, list)
        self.assertEqual(marker_text, "2")

    def test_startup_workers_are_capped_at_eight(self) -> None:
        config = SimpleNamespace(mcp_max_startup_workers=100)

        self.assertEqual(_startup_worker_count(20, config), 8)


def _test_config(repo: Path, project_config: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mcp_enabled=True,
        mcp_user_config_path=repo / "missing-user-mcp.json",
        mcp_project_config_path=project_config,
        mcp_startup_wait_seconds=8,
        mcp_max_startup_workers=8,
        mcp_enable_project_servers=True,
        mcp_initialize_timeout_seconds=2,
        mcp_call_timeout_seconds=2,
        max_parallel_tools=4,
        tool_batch_timeout_seconds=60,
        tool_shutdown_grace_seconds=2,
        max_process_output_chars=8_000,
    )


def _server_config(script: Path, *, env: dict[str, str] | None = None) -> dict[str, object]:
    return {
        "command": sys.executable,
        "args": [str(script)],
        "env": env or {},
    }


def _write_mcp_config(repo: Path, servers: dict[str, object]) -> Path:
    config_path = repo / ".paicli" / "mcp.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return config_path


def _write_fake_mcp_server(repo: Path) -> Path:
    script = repo / "fake_mcp_server.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path
            import sys
            import time

            marker = os.environ.get("START_MARKER")
            if marker:
                path = Path(marker)
                current = int(path.read_text(encoding="utf-8")) if path.exists() else 0
                path.write_text(str(current + 1), encoding="utf-8")
            startup_delay = float(os.environ.get("STARTUP_DELAY", "0") or "0")
            if startup_delay:
                time.sleep(startup_delay)

            tools = [
                {
                    "name": "echo",
                    "description": "Echo a message.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                }
            ]

            for line in sys.stdin:
                message = json.loads(line)
                method = message.get("method")
                request_id = message.get("id")
                if method == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1"},
                    }
                elif method == "tools/list":
                    result = {"tools": tools}
                elif method == "tools/call":
                    params = message.get("params") or {}
                    args = params.get("arguments") or {}
                    result = {"content": [{"type": "text", "text": str(args.get("message", ""))}]}
                elif method == "notifications/initialized":
                    continue
                else:
                    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "missing"}}), flush=True)
                    continue
                print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    return script


class _HttpMcpHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - http.server callback name
        length = int(self.headers.get("Content-Length", "0") or "0")
        message = json.loads(self.rfile.read(length).decode("utf-8"))
        if "id" not in message:
            self.send_response(202)
            self.end_headers()
            return
        method = message.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "http", "version": "1"},
            }
            self._send_response(message.get("id"), result, session_id="session-http")
            return
        self.server.session_headers.append(self.headers.get("Mcp-Session-Id", ""))  # type: ignore[attr-defined]
        if method == "tools/list":
            self._send_response(
                message.get("id"),
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo over HTTP.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                                "additionalProperties": False,
                            },
                        }
                    ]
                },
            )
            return
        if method == "tools/call":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            self._send_response(
                message.get("id"),
                {"content": [{"type": "text", "text": str(arguments.get("message", ""))}]},
            )
            return
        self._send_response(message.get("id"), {})

    def do_DELETE(self) -> None:  # noqa: N802 - http.server callback name
        self.server.delete_seen.set()  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()

    def _send_response(self, request_id: object, result: dict[str, Any], *, session_id: str = "") -> None:
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_http_mcp_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HttpMcpHandler)
    server.session_headers = []  # type: ignore[attr-defined]
    server.delete_seen = threading.Event()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, name="mcp-manager-http-test", daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}/mcp"


def _stop_http_mcp_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)


def _write_nonresponsive_mcp_server(repo: Path, marker: Path) -> Path:
    script = repo / "nonresponsive_mcp_server.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            import sys

            Path({str(marker)!r}).write_text(str(os.getpid()), encoding="utf-8")
            for _line in sys.stdin:
                pass
            """
        ),
        encoding="utf-8",
    )
    return script


if __name__ == "__main__":
    unittest.main()
