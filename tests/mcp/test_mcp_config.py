from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.mcp.config import McpConfigLoader, mask_sensitive_values


class McpConfigLoaderTests(unittest.TestCase):
    def test_load_merges_user_and_project_with_project_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            user_config = base / "user.json"
            project_config = base / "project" / ".paicli" / "mcp.json"
            project_config.parent.mkdir(parents=True)
            user_config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "shared": {"command": "user-cmd"},
                            "user_only": {"command": "user-only"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            project_config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "shared": {"command": "project-cmd"},
                            "project_only": {"command": "project-only"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            loaded = McpConfigLoader(
                base / "project",
                user_config=user_config,
                project_config=project_config,
            ).load()

        self.assertEqual(list(loaded), ["shared", "user_only", "project_only"])
        self.assertEqual(loaded["shared"].command, "project-cmd")
        self.assertEqual(loaded["user_only"].command, "user-only")
        self.assertEqual(loaded["project_only"].command, "project-only")

    def test_prepare_expands_project_home_env_and_dotenv_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "repo"
            project.mkdir()
            (project / ".env").write_text("PROJECT_TOKEN=dotenv-token\n", encoding="utf-8")
            config_path = project / ".paicli" / "mcp.json"
            config_path.parent.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "fake": {
                                "command": "${COMMAND}",
                                "args": ["${PROJECT_DIR}", "${HOME}", "${PROJECT_TOKEN}"],
                                "env": {"AUTH_TOKEN": "${TOKEN}"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            loader = McpConfigLoader(
                project,
                user_config=project / "missing-user.json",
                project_config=config_path,
                environ={"COMMAND": "python", "TOKEN": "secret"},
            )

            prepared = loader.prepare(loader.load()["fake"])

        self.assertEqual(prepared.command, "python")
        self.assertEqual(prepared.args[0], str(project.resolve()))
        self.assertEqual(prepared.args[1], str(Path.home()))
        self.assertEqual(prepared.args[2], "dotenv-token")
        self.assertEqual(prepared.env, {"AUTH_TOKEN": "secret"})

    def test_prepare_missing_variable_only_fails_that_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            config_path = project / "mcp.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "ok": {"command": "python"},
                            "bad": {"command": "${MISSING_MCP_VAR}"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            loader = McpConfigLoader(project, user_config=project / "none.json", project_config=config_path, environ={})
            loaded = loader.load()

            ok = loader.prepare(loaded["ok"])

        self.assertEqual(ok.command, "python")
        with self.assertRaisesRegex(ValueError, "MISSING_MCP_VAR"):
            loader.prepare(loaded["bad"])

    def test_server_shape_errors_are_deferred_to_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            config_path = project / "mcp.json"
            config_path.write_text(
                json.dumps({"mcpServers": {"bad": {"command": 1}, "ok": {"command": "python"}}}),
                encoding="utf-8",
            )
            loader = McpConfigLoader(project, user_config=project / "none.json", project_config=config_path)
            loaded = loader.load()

        self.assertEqual(loaded["ok"].command, "python")
        with self.assertRaisesRegex(ValueError, "command must be a string"):
            loader.prepare(loaded["bad"])

    def test_prepare_requires_exactly_one_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            config_path = project / "mcp.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "both": {"command": "python", "url": "http://127.0.0.1/mcp"},
                            "neither": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            loader = McpConfigLoader(project, user_config=project / "none.json", project_config=config_path)
            loaded = loader.load()

        with self.assertRaisesRegex(ValueError, "exactly one of command or url"):
            loader.prepare(loaded["both"])
        with self.assertRaisesRegex(ValueError, "exactly one of command or url"):
            loader.prepare(loaded["neither"])

    def test_mask_sensitive_values(self) -> None:
        masked = mask_sensitive_values({"Authorization": "Bearer x", "NODE_OPTIONS": "--max=1", "api_key": "secret"})

        self.assertEqual(masked["Authorization"], "***")
        self.assertEqual(masked["api_key"], "***")
        self.assertEqual(masked["NODE_OPTIONS"], "--max=1")


if __name__ == "__main__":
    unittest.main()
