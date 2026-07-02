from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import run_manifest_benchmark
from my_agent.tools import RepoTools


def fake_config(trace_dir: Path | None = None, **overrides: object) -> AgentConfig:
    resolved_trace_dir = trace_dir or Path("traces")
    values = {
        "provider": "fake",
        "api_key": "",
        "base_url": None,
        "model": "fake",
        "temperature": 0.0,
        "max_steps": 4,
        "command_timeout": 20,
        "trace_dir": resolved_trace_dir,
        "memory_dir": resolved_trace_dir.parent / "memory",
        "use_fake_llm": True,
        "memory_enabled": False,
    }
    values.update(overrides)
    return AgentConfig(**values)


def write_repo(repo: Path, value: int) -> None:
    repo.mkdir()
    (repo / "solution.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
    (repo / "visible_check.py").write_text(
        "import sys\nfrom solution import VALUE\nsys.exit(0 if VALUE >= 1 else 1)\n",
        encoding="utf-8",
    )
    (repo / "hidden_check.py").write_text(
        "import sys\nfrom solution import VALUE\nsys.exit(0 if VALUE == 2 else 1)\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_agent_trace(trace_dir: Path, run_id: str) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"agent_trace_{run_id}.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "event": "agent.completed",
                "payload": {
                    "mode": "react",
                    "run_label": "fake",
                    "stop_reason": "finish_called",
                    "steps": 1,
                    "done": True,
                    "status": "completed",
                    "trace_path": str(trace_path),
                    "child_trace_paths": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return trace_path


class ManifestBenchmarkTests(unittest.TestCase):
    def test_initial_visible_and_hidden_pass_marks_invalid_without_agent_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=2)
            manifest = base / "tasks.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "already-fixed",
                        "repo": str(repo),
                        "task": "Fix VALUE.",
                        "visible_test_command": [sys.executable, "visible_check.py"],
                        "hidden_test_command": [sys.executable, "hidden_check.py"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            calls: list[dict[str, object]] = []

            def fake_agent_runner(**kwargs: object) -> object:
                calls.append(dict(kwargs))
                raise AssertionError("agent should not run for invalid initial pass")

            result = run_manifest_benchmark(
                tasks_path=manifest,
                output_dir=base / "out",
                config=fake_config(base / "traces"),
                agent_runner=fake_agent_runner,
            )

        self.assertEqual(calls, [])
        self.assertEqual(result.results[0].failure_type, "invalid_initial_pass")
        self.assertFalse(result.results[0].task_valid)
        self.assertEqual(result.summary["invalid_initial_pass"], 1)

    def test_visible_pass_hidden_fail_records_clean_copy_result_and_trace_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "hidden-case",
                        "repo": str(repo),
                        "task": "Set VALUE to satisfy checks.",
                        "source": "unit",
                        "tags": ["hidden"],
                        "expected_changed_files": ["solution.py"],
                        "agent_test_command": [sys.executable, "visible_check.py"],
                        "visible_test_command": f"{sys.executable} visible_check.py",
                        "hidden_test_command": [sys.executable, "hidden_check.py"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runner_calls: list[dict[str, object]] = []

            def fake_agent_runner(**kwargs: object) -> object:
                runner_calls.append(dict(kwargs))
                work_repo = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / "agent_trace_fake.jsonl"
                (work_repo / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
                trace_path.write_text(
                    json.dumps(
                        {
                            "run_id": "run-hidden",
                            "event": "agent.completed",
                            "payload": {
                                "mode": "react",
                                "run_label": "fake",
                                "stop_reason": "finish_called",
                                "steps": 1,
                                "done": True,
                                "status": "completed",
                                "trace_path": str(trace_path),
                                "child_trace_paths": [],
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    trace_path=trace_path,
                    run_id="run-hidden",
                    steps=1,
                    done=True,
                    stop_reason="finish_called",
                )

            result = run_manifest_benchmark(
                tasks_path=manifest,
                output_dir=base / "out",
                config=fake_config(base / "traces"),
                agent_runner=fake_agent_runner,
            )

            trace_events = read_jsonl(Path(result.results[0].trace_path))
            clean_solution = base / "out" / "work" / "hidden-case" / "clean" / "solution.py"
            clean_solution_text = clean_solution.read_text(encoding="utf-8")
            patch_path = Path(result.results[0].patch_path)
            patch_exists = patch_path.exists()
            patch_text = patch_path.read_text(encoding="utf-8")

        self.assertEqual(len(runner_calls), 1)
        self.assertNotIn("hidden_check", str(runner_calls[0]["test_command"]))
        self.assertTrue(result.results[0].final_visible.ok)
        self.assertFalse(result.results[0].final_hidden.ok)
        self.assertFalse(result.results[0].resolved)
        self.assertEqual(result.results[0].failure_type, "hidden_test_failed")
        self.assertEqual(result.results[0].source, "unit")
        self.assertEqual(result.results[0].tags, ["hidden"])
        self.assertTrue(result.results[0].expected_changed_files_ok)
        self.assertTrue(result.results[0].patch_apply_ok)
        self.assertEqual(result.results[0].changed_files, ["solution.py"])
        self.assertEqual(clean_solution_text, "VALUE = 1\n")
        benchmark_payload = [event["payload"] for event in trace_events if event["event"] == "benchmark_result"][-1]
        self.assertFalse(benchmark_payload["hidden_ok"])
        self.assertEqual(benchmark_payload["failure_type"], "hidden_test_failed")
        self.assertNotIn("hidden_test_command", benchmark_payload)
        self.assertNotIn("hidden_test_output", benchmark_payload)
        self.assertNotIn("initial_hidden_output", benchmark_payload)
        self.assertEqual(result.results[0].mode, "auto")
        self.assertEqual(result.results[0].env_overrides, {})
        self.assertEqual(result.results[0].resolved_config["memory_enabled"], False)
        self.assertTrue(patch_exists)
        self.assertIn("solution.py", patch_text)

    def test_manifest_work_repo_has_git_baseline_for_agent_git_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "git-case",
                        "repo": str(repo),
                        "task": "Set VALUE to pass.",
                        "visible_test_command": [sys.executable, "visible_check.py"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            diff_outputs: list[str] = []

            def fake_agent_runner(**kwargs: object) -> object:
                work_repo = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                config = kwargs["config"]
                tools = RepoTools(work_repo, config=config)
                diff = tools._git_diff({})
                diff_outputs.append(diff.output)
                self.assertTrue(diff.ok, diff.output)
                self.assertEqual(diff.output, "No git diff.")
                (work_repo / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
                trace_path = write_agent_trace(Path(kwargs["trace_dir"]), "run-git")  # type: ignore[arg-type]
                return SimpleNamespace(
                    trace_path=trace_path,
                    run_id="run-git",
                    steps=1,
                    done=True,
                    stop_reason="finish_called",
                )

            result = run_manifest_benchmark(
                tasks_path=manifest,
                output_dir=base / "out",
                config=fake_config(base / "traces"),
                agent_runner=fake_agent_runner,
            )
            work_git_dir_exists = (base / "out" / "work" / "git-case" / "repo" / ".git").is_dir()

        self.assertEqual(diff_outputs, ["No git diff."])
        self.assertTrue(work_git_dir_exists)
        self.assertTrue(result.results[0].resolved)

    def test_manifest_uses_per_task_memory_dir_by_default_and_respects_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            custom_memory = base / "custom-memory"
            manifest = base / "tasks.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "memory-default",
                                "repo": str(repo),
                                "task": "default memory",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "memory-custom",
                                "repo": str(repo),
                                "task": "custom memory",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                                "env_overrides": {"AGENTCLI_MEMORY_DIR": str(custom_memory)},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            seen_memory_dirs: dict[str, Path] = {}

            def fake_agent_runner(**kwargs: object) -> object:
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                config = kwargs["config"]
                self.assertIsInstance(config, AgentConfig)
                seen_memory_dirs[trace_dir.name] = config.memory_dir
                work_repo = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                (work_repo / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
                trace_path = write_agent_trace(trace_dir, f"run-{trace_dir.name}")
                return SimpleNamespace(
                    trace_path=trace_path,
                    run_id=f"run-{trace_dir.name}",
                    steps=1,
                    done=True,
                    stop_reason="finish_called",
                )

            result = run_manifest_benchmark(
                tasks_path=manifest,
                output_dir=base / "out",
                config=fake_config(base / "traces", memory_enabled=True),
                agent_runner=fake_agent_runner,
            )
            expected_default = base / "out" / "memory" / "memory-default"
            default_memory_dir_exists = expected_default.is_dir()
            snapshots = {item.task_id: item.resolved_config["memory_dir"] for item in result.results}

        self.assertTrue(all(item.resolved for item in result.results))
        self.assertEqual(seen_memory_dirs["memory-default"], expected_default)
        self.assertEqual(seen_memory_dirs["memory-custom"], custom_memory)
        self.assertTrue(default_memory_dir_exists)
        self.assertEqual(snapshots["memory-default"], str(expected_default))
        self.assertEqual(snapshots["memory-custom"], str(custom_memory))

    def test_env_overrides_apply_to_agent_config_and_internal_run_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            tests_dir = repo / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_env.py").write_text(
                "import os\nimport unittest\n\n"
                "class EnvTests(unittest.TestCase):\n"
                "    def test_eval_flag(self):\n"
                "        self.assertEqual(os.environ.get('EVAL_FLAG'), 'yes')\n",
                encoding="utf-8",
            )
            manifest = base / "tasks.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "env-case",
                        "repo": str(repo),
                        "task": "Set VALUE and run tests.",
                        "visible_test_command": [sys.executable, "visible_check.py"],
                        "env_overrides": {
                            "AGENTCLI_MEMORY": "0",
                            "AGENTCLI_MCP": "0",
                            "AGENTCLI_TEAM_PARALLEL": "0",
                            "EVAL_FLAG": "yes",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            seen_config: AgentConfig | None = None

            def fake_agent_runner(**kwargs: object) -> object:
                nonlocal seen_config
                seen_config = kwargs["config"]  # type: ignore[assignment]
                self.assertIsInstance(seen_config, AgentConfig)
                self.assertFalse(seen_config.memory_enabled)
                self.assertFalse(seen_config.context_window_explicit)
                self.assertFalse(seen_config.response_reserve_tokens_explicit)
                self.assertFalse(seen_config.compression_buffer_tokens_explicit)
                self.assertFalse(seen_config.repo_context_budget_tokens_explicit)
                self.assertFalse(seen_config.tool_schema_budget_tokens_explicit)
                self.assertFalse(seen_config.memory_short_term_tokens_explicit)
                self.assertFalse(seen_config.memory_context_tokens_explicit)
                self.assertFalse(seen_config.memory_tool_result_chars_explicit)
                self.assertFalse(seen_config.mcp_enabled)
                self.assertFalse(seen_config.team_parallel_enabled)
                tools = RepoTools(Path(kwargs["repo_path"]), config=seen_config)  # type: ignore[arg-type]
                env_result = tools._run_tests({"command": "python -m unittest discover -s tests -q"})
                self.assertTrue(env_result.ok, env_result.output)
                work_repo = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / "agent_trace_env.jsonl"
                (work_repo / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
                trace_path.write_text(
                    json.dumps(
                        {
                            "run_id": "run-env",
                            "event": "agent.completed",
                            "payload": {
                                "mode": "react",
                                "run_label": "fake",
                                "stop_reason": "finish_called",
                                "steps": 1,
                                "done": True,
                                "status": "completed",
                                "trace_path": str(trace_path),
                                "child_trace_paths": [],
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    trace_path=trace_path,
                    run_id="run-env",
                    steps=1,
                    done=True,
                    stop_reason="finish_called",
                )

            result = run_manifest_benchmark(
                tasks_path=manifest,
                output_dir=base / "out",
                config=fake_config(base / "traces", memory_enabled=True, mcp_enabled=True, team_parallel_enabled=True),
                env={"EVAL_FLAG": "no"},
                agent_runner=fake_agent_runner,
            )

        self.assertIsNotNone(seen_config)
        self.assertTrue(result.results[0].resolved)
        self.assertEqual(result.results[0].env_overrides["EVAL_FLAG"], "yes")
        self.assertFalse(result.results[0].resolved_config["memory_enabled"])
        self.assertFalse(result.results[0].resolved_config["mcp_enabled"])
        self.assertFalse(result.results[0].resolved_config["team_parallel_enabled"])


if __name__ == "__main__":
    unittest.main()
