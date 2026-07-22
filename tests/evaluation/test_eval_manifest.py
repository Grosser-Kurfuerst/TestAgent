from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.evaluation import manifest_benchmark as manifest_benchmark_module
from my_agent.evaluation.manifest_benchmark import (
    CommandResult,
    ManifestEvalResult,
    _config_for_eval_env,
    _config_env_values,
    _load_manifest,
    load_manifest_tasks,
    run_manifest_benchmark,
    summarize_manifest_results,
)
from my_agent.memory.manager import MemoryManager
from my_agent.tools import RepoTools
from tests.memory.experience.fixtures import save_typed_experience


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
    def test_formal_manifest_builds_shared_policy_and_embedding_once(self) -> None:
        class Encoder:
            model_revision = "embed-revision"
            tokenizer_revision = "embed-revision"

            def encode_queries(self, texts):
                return ((1.0, 0.0),) * len(texts)

            def encode_documents(self, texts):
                return ((1.0, 0.0),) * len(texts)

        policy = object()
        encoder = Encoder()
        config = fake_config(
            memory_evolver_mode="formal",
            memory_evolver_retrieval_backend="embedding_cosine",
        )
        with (
            patch.object(manifest_benchmark_module, "build_llm", return_value=policy) as build_policy,
            patch.object(
                manifest_benchmark_module.TransformersEmbeddingEncoder,
                "from_config",
                return_value=encoder,
            ) as build_encoder,
        ):
            shared_policy, shared_retriever = (
                manifest_benchmark_module._build_shared_runtime_resources(
                    config,
                    agent_runner=manifest_benchmark_module.run_agent,
                )
            )

        self.assertIs(shared_policy, policy)
        self.assertIsNotNone(shared_retriever)
        self.assertIs(shared_retriever.encoder, encoder)
        build_policy.assert_called_once_with(config)
        build_encoder.assert_called_once_with(config)

    def test_manifest_passes_same_shared_resources_to_each_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "id": "task-one",
                            "repo": str(repo),
                            "task": "Fix VALUE.",
                            "visible_test_command": [sys.executable, "visible_check.py"],
                        },
                        {
                            "id": "task-two",
                            "repo": str(repo),
                            "task": "Fix VALUE.",
                            "visible_test_command": [sys.executable, "visible_check.py"],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            policy = object()
            retriever = object()
            seen: list[tuple[object, object]] = []

            def fake_agent_runner(**kwargs: object) -> object:
                seen.append((kwargs["llm"], kwargs["memory_embedding_retriever"]))
                work_repo = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                (work_repo / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                run_id = f"run-{len(seen)}"
                trace_path = write_agent_trace(trace_dir, run_id)
                return SimpleNamespace(
                    trace_path=trace_path,
                    run_id=run_id,
                    steps=1,
                    done=True,
                    stop_reason="finish_called",
                )

            with patch.object(
                manifest_benchmark_module,
                "_build_shared_runtime_resources",
                return_value=(policy, retriever),
            ) as build_resources:
                result = run_manifest_benchmark(
                    tasks_path=manifest,
                    output_dir=base / "out",
                    config=fake_config(base / "traces"),
                    agent_runner=fake_agent_runner,
                )

        self.assertEqual(len(result.results), 2)
        self.assertEqual(seen, [(policy, retriever), (policy, retriever)])
        build_resources.assert_called_once()

    def test_load_manifest_tasks_keeps_public_shape_and_loads_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            jsonl_manifest = base / "tasks.jsonl"
            jsonl_manifest.write_text(
                json.dumps({"id": "jsonl-task", "repo": str(repo), "task": "Fix VALUE."}) + "\n",
                encoding="utf-8",
            )
            array_manifest = base / "tasks-array.json"
            array_manifest.write_text(
                json.dumps([{"id": "array-task", "repo": str(repo), "task": "Fix VALUE."}]),
                encoding="utf-8",
            )
            object_manifest = base / "tasks-object.json"
            object_manifest.write_text(
                json.dumps(
                    {
                        "memory_mode": "shared_stream",
                        "stream_id": "python-stream",
                        "tasks": [{"id": "object-task", "repo": str(repo), "task": "Fix VALUE."}],
                    }
                ),
                encoding="utf-8",
            )
            single_manifest = base / "single-task.json"
            single_manifest.write_text(
                json.dumps({"id": "single-task", "repo": str(repo), "task": "Fix VALUE."}),
                encoding="utf-8",
            )

            object_tasks, object_settings = _load_manifest(object_manifest)
            jsonl_task_id = load_manifest_tasks(jsonl_manifest)[0]["id"]
            array_task_id = load_manifest_tasks(array_manifest)[0]["id"]
            single_task_id = load_manifest_tasks(single_manifest)[0]["id"]
            object_public_tasks = load_manifest_tasks(object_manifest)

        self.assertEqual(jsonl_task_id, "jsonl-task")
        self.assertEqual(array_task_id, "array-task")
        self.assertEqual(single_task_id, "single-task")
        self.assertEqual(object_tasks[0]["id"], "object-task")
        self.assertEqual(object_public_tasks, object_tasks)
        self.assertEqual(object_settings.memory_mode, "shared_stream")
        self.assertEqual(object_settings.stream_id, "python-stream")

    def test_load_manifest_rejects_invalid_top_level_memory_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "memory_mode": "shared",
                        "tasks": [{"id": "bad-mode", "repo": str(repo), "task": "Fix VALUE."}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported memory_mode"):
                _load_manifest(manifest)

    def test_config_env_values_preserves_memory_project_key(self) -> None:
        config = fake_config(memory_project_key="stream:alpha")

        values = _config_env_values(config)

        self.assertEqual(values["AGENTCLI_MEMORY_PROJECT_KEY"], "stream:alpha")

    def test_config_env_values_preserves_reasoning_effort(self) -> None:
        config = fake_config(reasoning_effort="high")

        values = _config_env_values(config)

        self.assertEqual(values["MY_AGENT_REASONING_EFFORT"], "high")

    def test_config_env_values_preserves_evolver_config(self) -> None:
        config = fake_config(
            memory_evolver_mode="retrieve_select",
            memory_evolver_top_k_per_tier=7,
            memory_evolver_selected_max_items=6,
            memory_evolver_min_score=0.25,
            memory_evolver_min_experience_entries=3,
            memory_evolver_writer_enabled=True,
            memory_evolver_writer_mode="llm",
            memory_evolver_writer_min_confidence=0.8,
            memory_evolver_writer_max_records=4,
            memory_evolver_writer_max_input_chars=5000,
            memory_evolver_writer_max_content_chars=700,
            memory_evolver_writer_dataset_path=Path("/tmp/writer.jsonl"),
        )

        values = _config_env_values(config)

        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_MODE"], "retrieve_select")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER"], "7")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS"], "6")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_MIN_SCORE"], "0.25")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES"], "3")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_WRITER"], "1")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_WRITER_MODE"], "llm")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE"], "0.8")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS"], "4")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS"], "5000")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS"], "700")
        self.assertEqual(values["AGENTCLI_MEMORY_EVOLVER_WRITER_DATASET_PATH"], "/tmp/writer.jsonl")

    def test_config_for_eval_env_preserves_and_overrides_evolver_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = fake_config(
                base / "traces",
                memory_evolver_mode="retrieve_select",
                memory_evolver_top_k_per_tier=7,
                memory_evolver_selected_max_items=6,
                memory_evolver_min_score=0.25,
                memory_evolver_min_experience_entries=3,
                memory_evolver_writer_enabled=True,
                memory_evolver_writer_mode="fallback",
                memory_evolver_writer_min_confidence=0.8,
                memory_evolver_writer_max_records=4,
                memory_evolver_writer_max_input_chars=5000,
                memory_evolver_writer_max_content_chars=700,
                memory_evolver_writer_dataset_path=base / "writer.jsonl",
                memory_evolver_tier_caps={"trajectory": 2, "tip": 1, "skill": 3, "tool": 4},
                memory_evolver_tier_weights={"trajectory": 1.3, "tip": 0.8, "skill": 1.7, "tool": 1.1},
            )

            preserved = _config_for_eval_env(
                config,
                {},
                trace_dir=base / "task-traces",
                memory_dir=base / "memory",
                command_timeout=11,
            )
            overridden = _config_for_eval_env(
                config,
                {
                    "AGENTCLI_MEMORY_EVOLVER_MODE": "off",
                    "MY_AGENT_MEMORY_EVOLVER_TOP_K_PER_TIER": "2",
                    "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": "4",
                    "MY_AGENT_MEMORY_EVOLVER_MIN_SCORE": "0.5",
                    "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": "9",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER": "0",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MODE": "llm",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE": "0.6",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_RECORDS": "2",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS": "3000",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS": "400",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_DATASET_PATH": str(base / "override-writer.jsonl"),
                },
                trace_dir=base / "task-traces-override",
                memory_dir=base / "memory-override",
                command_timeout=13,
            )

        self.assertEqual(preserved.memory_evolver_mode, "retrieve_select")
        self.assertEqual(preserved.memory_evolver_top_k_per_tier, 7)
        self.assertEqual(preserved.memory_evolver_selected_max_items, 6)
        self.assertEqual(preserved.memory_evolver_min_score, 0.25)
        self.assertEqual(preserved.memory_evolver_min_experience_entries, 3)
        self.assertTrue(preserved.memory_evolver_writer_enabled)
        self.assertEqual(preserved.memory_evolver_writer_mode, "fallback")
        self.assertEqual(preserved.memory_evolver_writer_min_confidence, 0.8)
        self.assertEqual(preserved.memory_evolver_writer_max_records, 4)
        self.assertEqual(preserved.memory_evolver_writer_max_input_chars, 5000)
        self.assertEqual(preserved.memory_evolver_writer_max_content_chars, 700)
        self.assertEqual(preserved.memory_evolver_writer_dataset_path, base / "writer.jsonl")
        self.assertEqual(preserved.memory_evolver_tier_caps, {"trajectory": 2, "tip": 1, "skill": 3, "tool": 4})
        self.assertEqual(preserved.memory_evolver_tier_weights, {"trajectory": 1.3, "tip": 0.8, "skill": 1.7, "tool": 1.1})
        self.assertEqual(overridden.memory_evolver_mode, "off")
        self.assertEqual(overridden.memory_evolver_top_k_per_tier, 2)
        self.assertEqual(overridden.memory_evolver_selected_max_items, 4)
        self.assertEqual(overridden.memory_evolver_min_score, 0.5)
        self.assertEqual(overridden.memory_evolver_min_experience_entries, 9)
        self.assertFalse(overridden.memory_evolver_writer_enabled)
        self.assertEqual(overridden.memory_evolver_writer_mode, "llm")
        self.assertEqual(overridden.memory_evolver_writer_min_confidence, 0.6)
        self.assertEqual(overridden.memory_evolver_writer_max_records, 2)
        self.assertEqual(overridden.memory_evolver_writer_max_input_chars, 3000)
        self.assertEqual(overridden.memory_evolver_writer_max_content_chars, 400)
        self.assertEqual(overridden.memory_evolver_writer_dataset_path, base / "override-writer.jsonl")
        self.assertEqual(overridden.memory_evolver_tier_caps, {"trajectory": 2, "tip": 1, "skill": 3, "tool": 4})
        self.assertEqual(overridden.memory_evolver_tier_weights, {"trajectory": 1.3, "tip": 0.8, "skill": 1.7, "tool": 1.1})

    def test_config_for_eval_env_applies_evolver_bool_alias_after_mode_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = fake_config(base / "traces")

            agentcli_bool = _config_for_eval_env(
                config,
                {"AGENTCLI_MEMORY_EVOLVER": "1"},
                trace_dir=base / "trace-agentcli-bool",
                memory_dir=base / "memory-agentcli-bool",
                command_timeout=11,
            )
            mode_beats_bool = _config_for_eval_env(
                config,
                {
                    "MY_AGENT_MEMORY_EVOLVER_MODE": "retrieve_select",
                    "MY_AGENT_MEMORY_EVOLVER": "0",
                },
                trace_dir=base / "trace-mode",
                memory_dir=base / "memory-mode",
                command_timeout=11,
            )
            agentcli_mode_beats_my_agent_mode = _config_for_eval_env(
                config,
                {
                    "AGENTCLI_MEMORY_EVOLVER_MODE": "off",
                    "MY_AGENT_MEMORY_EVOLVER_MODE": "retrieve_select",
                    "AGENTCLI_MEMORY_EVOLVER": "1",
                },
                trace_dir=base / "trace-agentcli-mode",
                memory_dir=base / "memory-agentcli-mode",
                command_timeout=11,
            )

        self.assertEqual(agentcli_bool.memory_evolver_mode, "retrieve_select")
        self.assertEqual(mode_beats_bool.memory_evolver_mode, "retrieve_select")
        self.assertEqual(agentcli_mode_beats_my_agent_mode.memory_evolver_mode, "off")

    def test_manifest_summary_separates_same_stream_id_across_shared_modes(self) -> None:
        initial_visible = CommandResult(command="", ok=False, returncode=1)
        results = [
            ManifestEvalResult(
                task_id="stream-task",
                status="passed",
                resolved=True,
                task_valid=True,
                failure_type="",
                initial_visible=initial_visible,
                memory_mode="shared_stream",
                stream_id="python",
                memory_dir="/tmp/streams/python",
                memory_project_key="stream-key",
                memory_entries_before=0,
                memory_entries_after=2,
                memory_growth=2,
                memory_entries_total_before=0,
                memory_entries_total_after=2,
                memory_total_growth=2,
            ),
            ManifestEvalResult(
                task_id="group-task",
                status="failed",
                resolved=False,
                task_valid=True,
                failure_type="visible_test_failed",
                initial_visible=initial_visible,
                memory_mode="shared_by_group",
                stream_id="python",
                memory_dir="/tmp/groups/python",
                memory_project_key="group-key",
                memory_entries_before=0,
                memory_entries_after=3,
                memory_growth=3,
                memory_entries_total_before=0,
                memory_entries_total_after=3,
                memory_total_growth=3,
            ),
        ]

        summary = summarize_manifest_results(results)

        self.assertNotIn("python", summary["streams"])
        self.assertIn("shared_stream:python", summary["streams"])
        self.assertIn("shared_by_group:python", summary["streams"])
        self.assertEqual(summary["streams"]["shared_stream:python"]["memory_dir"], "/tmp/streams/python")
        self.assertEqual(summary["streams"]["shared_by_group:python"]["memory_dir"], "/tmp/groups/python")
        self.assertEqual(summary["streams"]["shared_stream:python"]["solve_rate"], 100.0)
        self.assertEqual(summary["streams"]["shared_by_group:python"]["solve_rate"], 0.0)
        self.assertEqual(summary["memory"]["visible_growth"], 5)

    def test_manifest_summary_separates_same_stream_id_with_memory_dir_override(self) -> None:
        initial_visible = CommandResult(command="", ok=False, returncode=1)
        results = [
            ManifestEvalResult(
                task_id="one",
                status="passed",
                resolved=True,
                task_valid=True,
                failure_type="",
                initial_visible=initial_visible,
                memory_mode="shared_stream",
                stream_id="python",
                memory_dir="/tmp/one",
                memory_project_key="same-key",
                memory_entries_before=0,
                memory_entries_after=1,
                memory_growth=1,
                memory_entries_total_before=0,
                memory_entries_total_after=1,
                memory_total_growth=1,
            ),
            ManifestEvalResult(
                task_id="two",
                status="passed",
                resolved=True,
                task_valid=True,
                failure_type="",
                initial_visible=initial_visible,
                memory_mode="shared_stream",
                stream_id="python",
                memory_dir="/tmp/two",
                memory_project_key="same-key",
                memory_entries_before=0,
                memory_entries_after=2,
                memory_growth=2,
                memory_entries_total_before=0,
                memory_entries_total_after=2,
                memory_total_growth=2,
            ),
        ]

        summary = summarize_manifest_results(results)
        stream_items = {
            payload["memory_dir"]: payload
            for key, payload in summary["streams"].items()
            if key.startswith("shared_stream:python:")
        }

        self.assertEqual(set(stream_items), {"/tmp/one", "/tmp/two"})
        self.assertEqual(stream_items["/tmp/one"]["memory_growth"], 1)
        self.assertEqual(stream_items["/tmp/two"]["memory_growth"], 2)
        self.assertEqual(summary["memory"]["visible_growth"], 3)

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
        record = result.results[0].to_dict()
        self.assertEqual(record["memory_mode"], "per_task")
        self.assertEqual(record["stream_id"], "already-fixed")
        self.assertEqual(record["memory_project_key"], "")
        self.assertTrue(str(record["memory_dir"]).endswith("out/memory/already-fixed"))
        self.assertEqual(record["memory_entries_before"], 0)
        self.assertEqual(record["memory_entries_after"], 0)
        self.assertEqual(record["memory_growth"], 0)
        self.assertEqual(record["memory_entries_total_before"], 0)
        self.assertEqual(record["memory_entries_total_after"], 0)
        self.assertEqual(record["memory_total_growth"], 0)

    def test_manifest_runner_passes_writer_context_metadata_to_agent_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "memory_mode": "shared_stream",
                        "stream_id": "python-stream",
                        "tasks": [
                            {
                                "id": "task-one",
                                "source": "manifest",
                                "repo": str(repo),
                                "task": "Fix VALUE.",
                                "tags": ["python", "writer"],
                                "env_overrides": {"MY_AGENT_MEMORY_PROJECT_KEY": "override:key"},
                                "visible_test_command": [sys.executable, "visible_check.py"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_metadata: list[dict[str, object]] = []
            seen_project_key = ""
            seen_writer_enabled = False
            seen_writer_mode = ""

            def fake_agent_runner(**kwargs: object) -> object:
                nonlocal seen_project_key, seen_writer_enabled, seen_writer_mode
                seen_metadata.append(dict(kwargs["metadata"]))  # type: ignore[arg-type]
                config = kwargs["config"]
                self.assertIsInstance(config, AgentConfig)
                seen_project_key = config.memory_project_key
                seen_writer_enabled = config.memory_evolver_writer_enabled
                seen_writer_mode = config.memory_evolver_writer_mode
                work_repo = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                (work_repo / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                trace_path = write_agent_trace(trace_dir, "run-task-one")
                return SimpleNamespace(
                    trace_path=trace_path,
                    run_id="run-task-one",
                    steps=1,
                    done=True,
                    stop_reason="finish_called",
                )

            result = run_manifest_benchmark(
                tasks_path=manifest,
                output_dir=base / "out",
                config=fake_config(
                    base / "traces",
                    memory_enabled=True,
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_mode="fallback",
                ),
                agent_runner=fake_agent_runner,
            )

        self.assertTrue(result.results[0].resolved)
        self.assertTrue(seen_writer_enabled)
        self.assertEqual(seen_writer_mode, "fallback")
        self.assertEqual(len(seen_metadata), 1)
        metadata = seen_metadata[0]
        self.assertEqual(metadata["source_task"], "task-one")
        self.assertEqual(metadata["task_id"], "task-one")
        self.assertEqual(metadata["task_type"], "manifest")
        self.assertEqual(metadata["stream_id"], "python-stream")
        self.assertEqual(metadata["memory_mode"], "shared_stream")
        self.assertEqual(seen_project_key, "override:key")
        self.assertEqual(metadata["memory_project_key"], "override:key")
        self.assertEqual(metadata["tags"], ["python", "writer"])

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
        self.assertEqual(benchmark_payload["memory_mode"], "per_task")
        self.assertEqual(benchmark_payload["stream_id"], "hidden-case")
        self.assertEqual(benchmark_payload["memory_entries_before"], 0)
        self.assertEqual(benchmark_payload["memory_entries_after"], 0)
        self.assertEqual(benchmark_payload["memory_growth"], 0)
        self.assertEqual(benchmark_payload["memory_entries_total_before"], 0)
        self.assertEqual(benchmark_payload["memory_entries_total_after"], 0)
        self.assertEqual(benchmark_payload["memory_total_growth"], 0)
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
            legacy_memory = base / "legacy-memory"
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
                        json.dumps(
                            {
                                "id": "memory-legacy",
                                "repo": str(repo),
                                "task": "legacy memory",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                                "env_overrides": {"MY_AGENT_MEMORY_DIR": str(legacy_memory)},
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
        self.assertEqual(seen_memory_dirs["memory-legacy"], legacy_memory)
        self.assertTrue(default_memory_dir_exists)
        self.assertEqual(snapshots["memory-default"], str(expected_default))
        self.assertEqual(snapshots["memory-custom"], str(custom_memory))
        self.assertEqual(snapshots["memory-legacy"], str(legacy_memory))

    def test_shared_stream_uses_same_memory_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "memory_mode": "shared_stream",
                        "stream_id": "humaneval-python",
                        "tasks": [
                            {
                                "id": "stream-first",
                                "repo": str(repo),
                                "task": "Fix VALUE.",
                                "stream_id": "   ",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                            },
                            {
                                "id": "stream-second",
                                "repo": str(repo),
                                "task": "Fix VALUE again.",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_memory_dirs: dict[str, Path] = {}
            seen_project_keys: dict[str, str] = {}

            def fake_agent_runner(**kwargs: object) -> object:
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                config = kwargs["config"]
                self.assertIsInstance(config, AgentConfig)
                seen_memory_dirs[trace_dir.name] = config.memory_dir
                seen_project_keys[trace_dir.name] = config.memory_project_key
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
            expected_memory_dir = base / "out" / "memory" / "streams" / "humaneval-python"
            expected_memory_dir_exists = expected_memory_dir.is_dir()
            snapshots = {item.task_id: item.resolved_config["memory_project_key"] for item in result.results}

        self.assertTrue(all(item.resolved for item in result.results))
        self.assertEqual(seen_memory_dirs["stream-first"], expected_memory_dir)
        self.assertEqual(seen_memory_dirs["stream-second"], expected_memory_dir)
        self.assertTrue(seen_project_keys["stream-first"])
        self.assertEqual(seen_project_keys["stream-first"], seen_project_keys["stream-second"])
        self.assertEqual(snapshots["stream-first"], seen_project_keys["stream-first"])
        self.assertEqual(snapshots["stream-second"], seen_project_keys["stream-first"])
        self.assertTrue(expected_memory_dir_exists)

    def test_shared_stream_project_memory_is_visible_across_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "memory_mode": "shared_stream",
                        "stream_id": "memory-stream",
                        "tasks": [
                            {
                                "id": "first",
                                "repo": str(repo),
                                "task": "Save stream memory.",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                            },
                            {
                                "id": "second",
                                "repo": str(repo),
                                "task": "Read stream memory.",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            second_saw_marker = False

            def fake_agent_runner(**kwargs: object) -> object:
                nonlocal second_saw_marker
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                config = kwargs["config"]
                self.assertIsInstance(config, AgentConfig)
                repo_path = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                memory = MemoryManager.from_config(config=config, llm=None, repo_path=repo_path)
                if trace_dir.name == "first":
                    save_typed_experience(memory, "stream marker: prefer VALUE = 1", tier="tip")
                else:
                    entries = memory.experience_store.all(project_key=memory.project_key)
                    second_saw_marker = any("stream marker" in entry.content for entry in entries)
                (repo_path / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
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
            result_records = read_jsonl(result.results_path)

        self.assertTrue(all(item.resolved for item in result.results))
        self.assertTrue(second_saw_marker)
        self.assertEqual(result.results[0].memory_mode, "shared_stream")
        self.assertEqual(result.results[0].stream_id, "memory-stream")
        self.assertEqual(result.results[0].memory_entries_before, 0)
        self.assertEqual(result.results[0].memory_entries_after, 1)
        self.assertEqual(result.results[0].memory_growth, 1)
        self.assertEqual(result.results[1].memory_entries_before, 1)
        self.assertEqual(result.results[1].memory_entries_after, 1)
        self.assertEqual(result.results[1].memory_growth, 0)
        self.assertEqual(result_records[0]["memory_mode"], "shared_stream")
        self.assertEqual(result_records[0]["stream_id"], "memory-stream")
        self.assertTrue(result_records[0]["memory_project_key"])
        self.assertEqual(result_records[0]["memory_entries_total_before"], 0)
        self.assertEqual(result_records[0]["memory_entries_total_after"], 1)
        self.assertEqual(result.summary["memory"]["modes"]["shared_stream"], 2)
        self.assertEqual(result.summary["memory"]["visible_growth"], 1)
        self.assertEqual(result.summary["memory"]["total_growth"], 1)
        self.assertIn("memory-stream", result.summary["streams"])
        stream_summary = result.summary["streams"]["memory-stream"]
        self.assertEqual(stream_summary["total"], 2)
        self.assertEqual(stream_summary["scored"], 2)
        self.assertEqual(stream_summary["resolved"], 2)
        self.assertEqual(stream_summary["solve_rate"], 100.0)
        self.assertEqual(stream_summary["memory_entries_before"], 0)
        self.assertEqual(stream_summary["memory_entries_after"], 1)
        self.assertEqual(stream_summary["memory_growth"], 1)

    def test_my_agent_memory_project_key_override_wins_over_base_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "memory_mode": "shared_stream",
                        "stream_id": "memory-stream",
                        "tasks": [
                            {
                                "id": "override-key",
                                "repo": str(repo),
                                "task": "Use explicit project key.",
                                "visible_test_command": [sys.executable, "visible_check.py"],
                                "env_overrides": {"MY_AGENT_MEMORY_PROJECT_KEY": "task:legacy"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            seen_project_key = ""

            def fake_agent_runner(**kwargs: object) -> object:
                nonlocal seen_project_key
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                config = kwargs["config"]
                self.assertIsInstance(config, AgentConfig)
                seen_project_key = config.memory_project_key
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
                config=fake_config(base / "traces", memory_enabled=True, memory_project_key="base:key"),
                agent_runner=fake_agent_runner,
            )

        self.assertTrue(result.results[0].resolved)
        self.assertEqual(seen_project_key, "task:legacy")
        self.assertEqual(result.results[0].resolved_config["memory_project_key"], "task:legacy")

    def test_per_task_project_memory_is_isolated_between_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.jsonl"
            rows = [
                {
                    "id": "first",
                    "repo": str(repo),
                    "task": "Save per-task memory.",
                    "visible_test_command": [sys.executable, "visible_check.py"],
                },
                {
                    "id": "second",
                    "repo": str(repo),
                    "task": "Do not see per-task memory.",
                    "visible_test_command": [sys.executable, "visible_check.py"],
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            second_saw_marker = False

            def fake_agent_runner(**kwargs: object) -> object:
                nonlocal second_saw_marker
                trace_dir = Path(kwargs["trace_dir"])  # type: ignore[arg-type]
                config = kwargs["config"]
                self.assertIsInstance(config, AgentConfig)
                repo_path = Path(kwargs["repo_path"])  # type: ignore[arg-type]
                memory = MemoryManager.from_config(config=config, llm=None, repo_path=repo_path)
                if trace_dir.name == "first":
                    save_typed_experience(memory, "isolated stream marker VALUE", tier="tip")
                else:
                    entries = memory.experience_store.all(project_key=memory.project_key)
                    second_saw_marker = any("isolated stream marker" in entry.content for entry in entries)
                (repo_path / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
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

        self.assertTrue(all(item.resolved for item in result.results))
        self.assertFalse(second_saw_marker)

    def test_shared_by_group_uses_group_memory_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.jsonl"
            rows = [
                {
                    "id": "python-a",
                    "repo": str(repo),
                    "task": "Fix VALUE.",
                    "memory_mode": "shared_by_group",
                    "stream_id": "   ",
                    "group": "python",
                    "visible_test_command": [sys.executable, "visible_check.py"],
                },
                {
                    "id": "python-b",
                    "repo": str(repo),
                    "task": "Fix VALUE again.",
                    "memory_mode": "shared_by_group",
                    "group": "python",
                    "visible_test_command": [sys.executable, "visible_check.py"],
                },
                {
                    "id": "java-a",
                    "repo": str(repo),
                    "task": "Fix VALUE separately.",
                    "memory_mode": "shared_by_group",
                    "stream_id": "java",
                    "visible_test_command": [sys.executable, "visible_check.py"],
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
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
            python_memory_dir = base / "out" / "memory" / "groups" / "python"
            java_memory_dir = base / "out" / "memory" / "groups" / "java"
            python_memory_dir_exists = python_memory_dir.is_dir()
            java_memory_dir_exists = java_memory_dir.is_dir()

        self.assertTrue(all(item.resolved for item in result.results))
        self.assertEqual(seen_memory_dirs["python-a"], python_memory_dir)
        self.assertEqual(seen_memory_dirs["python-b"], python_memory_dir)
        self.assertEqual(seen_memory_dirs["java-a"], java_memory_dir)
        self.assertNotEqual(python_memory_dir, java_memory_dir)
        self.assertTrue(python_memory_dir_exists)
        self.assertTrue(java_memory_dir_exists)

    def test_shared_by_group_requires_group_or_stream_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            write_repo(repo, value=0)
            manifest = base / "tasks.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "missing-group",
                        "repo": str(repo),
                        "task": "Fix VALUE.",
                        "memory_mode": "shared_by_group",
                        "visible_test_command": [sys.executable, "visible_check.py"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "shared_by_group requires"):
                run_manifest_benchmark(
                    tasks_path=manifest,
                    output_dir=base / "out",
                    config=fake_config(base / "traces", memory_enabled=True),
                    agent_runner=lambda **_: None,
                )

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
