from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tests._path import add_src_to_path
from tests.ui.test_ui import fake_config

add_src_to_path()

from my_agent.mcp.manager import McpServerManagerPool
from my_agent.ui import AgentRepl, PlainRenderer


class McpReplTests(unittest.TestCase):
    def setUp(self) -> None:
        McpServerManagerPool.clear()

    def tearDown(self) -> None:
        McpServerManagerPool.clear()

    def test_repl_mcp_status_logs_reload_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            marker = repo / "starts.txt"
            script = write_fake_mcp_server_with_logs(repo)
            write_mcp_config(repo, {"fake": mcp_server_config(script, env={"START_MARKER": str(marker)})})
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(
                    base / "traces",
                    mcp_initialize_timeout_seconds=2,
                    mcp_call_timeout_seconds=2,
                    hitl_enabled=True,
                    hitl_medium_risk_mode="allow",
                ),
                trace_dir=base / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO(
                    "/mcp\n"
                    "/mcp logs fake\n"
                    "/tools\n"
                    "/context\n"
                    "/mcp reload\n"
                    "/mcp status\n"
                    "/quit\n"
                ),
            )

            exit_code = repl.run(show_banner=False)
            marker_text = marker.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("MCP servers", text)
        self.assertIn("fake\tready\tstdio\t1", text)
        self.assertIn("MCP logs: fake", text)
        self.assertIn("fake stderr ready", text)
        self.assertIn("mcp__fake__echo", text)
        self.assertIn("mcp__fake__echo\tmcp:fake\texternal\task", text)
        self.assertIn("mcp: 1/1 ready, 1 tools", text)
        self.assertIn("Reloaded MCP servers and tools. Cleared HITL approve-all grants.", text)
        self.assertEqual(marker_text, "2")
        self.assertEqual(errors.getvalue(), "")

    def test_repl_mcp_usage_for_missing_log_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(repo / "traces"),
                trace_dir=repo / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO("/mcp logs\n/quit\n"),
            )

            exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        self.assertIn("Usage: /mcp logs <server>", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_repl_mcp_disabled_does_not_start_servers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            marker = repo / "starts.txt"
            script = write_fake_mcp_server_with_logs(repo)
            write_mcp_config(repo, {"fake": mcp_server_config(script, env={"START_MARKER": str(marker)})})
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(base / "traces", mcp_enabled=False),
                trace_dir=base / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO("/context\n/mcp\n/mcp logs fake\n/quit\n"),
            )

            exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("mcp: disabled", text)
        self.assertIn("MCP servers\n- disabled", text)
        self.assertIn("MCP logs: fake\nMCP is disabled.", text)
        self.assertFalse(marker.exists())
        self.assertEqual(errors.getvalue(), "")


def write_fake_mcp_server_with_logs(repo: Path) -> Path:
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
            print("fake stderr ready", file=sys.stderr, flush=True)

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
                    result = {}
                print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    return script


def write_mcp_config(repo: Path, servers: dict[str, object]) -> Path:
    config_path = repo / ".paicli" / "mcp.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return config_path


def mcp_server_config(script: Path, *, env: dict[str, str] | None = None) -> dict[str, object]:
    return {"command": sys.executable, "args": [str(script)], "env": env or {}}


if __name__ == "__main__":
    unittest.main()
