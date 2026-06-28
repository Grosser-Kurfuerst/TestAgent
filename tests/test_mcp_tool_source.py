from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from ._path import add_src_to_path
    from . import test_mcp_manager as mcp_helpers
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path
    import test_mcp_manager as mcp_helpers

add_src_to_path()

from my_agent.mcp.manager import McpServerManagerPool
from my_agent.tools import RepoTools, ToolInvocation, ToolRisk


class McpToolSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        McpServerManagerPool.clear()

    def tearDown(self) -> None:
        McpServerManagerPool.clear()

    def test_repo_tools_tool_definitions_include_mcp_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = mcp_helpers._write_fake_mcp_server(repo)
            config_path = mcp_helpers._write_mcp_config(repo, {"fake": mcp_helpers._server_config(script)})
            config = mcp_helpers._test_config(repo, config_path)

            tools = RepoTools(repo, config=config)
            definitions = tools.tool_definitions()
            registered = tools.registry.get_registered("mcp__fake__echo")

        self.assertIn("mcp__fake__echo", [item["function"]["name"] for item in definitions])
        self.assertIsNotNone(registered)
        self.assertEqual(registered.spec.source, "mcp:fake")
        self.assertEqual(registered.spec.risk, ToolRisk.EXTERNAL)

    def test_mcp_tool_call_returns_server_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = mcp_helpers._write_fake_mcp_server(repo)
            config_path = mcp_helpers._write_mcp_config(repo, {"fake": mcp_helpers._server_config(script)})
            config = mcp_helpers._test_config(repo, config_path)

            result = RepoTools(repo, config=config).execute(
                [ToolInvocation.from_arguments("mcp__fake__echo", {"message": "from-source"})]
            )[0]

        self.assertTrue(result.ok, result.content)
        self.assertEqual(result.content, "from-source")

    def test_disabled_mcp_config_does_not_load_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = mcp_helpers._write_fake_mcp_server(repo)
            config_path = mcp_helpers._write_mcp_config(repo, {"fake": mcp_helpers._server_config(script)})
            config = SimpleNamespace(
                mcp_enabled=False,
                mcp_user_config_path=repo / "missing-user.json",
                mcp_project_config_path=config_path,
            )

            tools = RepoTools(repo, config=config)

        self.assertNotIn("mcp__fake__echo", tools.tool_names)

    def test_project_mcp_config_loads_by_default_and_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = mcp_helpers._write_fake_mcp_server(repo)
            mcp_helpers._write_mcp_config(repo, {"fake": mcp_helpers._server_config(script)})
            default_config = SimpleNamespace(
                mcp_enabled=True,
                mcp_user_config_path=repo / "missing-user.json",
                mcp_initialize_timeout_seconds=2,
                mcp_call_timeout_seconds=2,
            )
            disabled = SimpleNamespace(
                mcp_enabled=True,
                mcp_user_config_path=repo / "missing-user.json",
                mcp_enable_project_servers=False,
            )

            default_tools = RepoTools(repo, config=default_config)
            disabled_tools = RepoTools(repo, config=disabled)

        self.assertIn("mcp__fake__echo", default_tools.tool_names)
        self.assertNotIn("mcp__fake__echo", disabled_tools.tool_names)

    def test_startup_wait_does_not_block_until_initialize_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "pid.txt"
            script = mcp_helpers._write_nonresponsive_mcp_server(repo, marker)
            config_path = mcp_helpers._write_mcp_config(repo, {"slow": mcp_helpers._server_config(script)})
            config = mcp_helpers._test_config(repo, config_path)
            config.mcp_startup_wait_seconds = 0
            config.mcp_initialize_timeout_seconds = 3

            started_at = time.monotonic()
            tools = RepoTools(repo, config=config)
            elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 1.0)
        self.assertNotIn("mcp__slow__echo", tools.tool_names)


if __name__ == "__main__":
    unittest.main()
