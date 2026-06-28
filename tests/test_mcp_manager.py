from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

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

            marker = os.environ.get("START_MARKER")
            if marker:
                path = Path(marker)
                current = int(path.read_text(encoding="utf-8")) if path.exists() else 0
                path.write_text(str(current + 1), encoding="utf-8")

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
