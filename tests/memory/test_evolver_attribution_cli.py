from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.cli import main
from my_agent.memory.evolver import build_experience_entry
from my_agent.memory.long_term import LongTermMemoryStore
from my_agent.memory.types import MemoryScope


PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:python"


def _candidate_event(memory_id: str, *, timestamp: str, run_id: str = "") -> dict:
    return {
        "event": "memory.evolver_candidates",
        "time": timestamp,
        "run_id": run_id,
        "payload": {
            "candidate_count": 1,
            "candidate_summaries": [{"id": memory_id, "tier": "skill", "score": 1.0, "tokens": 4}],
            "candidate_ids": [memory_id],
            "selection_policy": "rule_tier_weighted_v1",
            "memory_project_key": PROJECT_KEY,
            "timestamp": timestamp,
        },
    }


def _selected_event(ids: list[str], *, timestamp: str, run_id: str = "") -> dict:
    return {
        "event": "memory.evolver_selected",
        "time": timestamp,
        "run_id": run_id,
        "payload": {
            "selected_count": len(ids),
            "selected_ids": ids,
            "selection_policy": "rule_tier_weighted_v1",
            "memory_project_key": PROJECT_KEY,
            "timestamp": timestamp,
        },
    }


def _benchmark_event(task_id: str, *, resolved: bool, timestamp: str, run_id: str = "") -> dict:
    return {
        "event": "benchmark_result",
        "time": timestamp,
        "run_id": run_id,
        "payload": {
            "task_id": task_id,
            "resolved": resolved,
            "status": "resolved" if resolved else "failed",
            "failure_type": "" if resolved else "assertion_failed",
            "source": "humaneval",
            "stream_id": "python",
            "memory_mode": "shared_stream",
            "memory_project_key": PROJECT_KEY,
            "timestamp": timestamp,
        },
    }


def _write_trace(path: Path, task_id: str, memory_id: str, *, selected: bool, resolved: bool) -> None:
    run_id = f"run-{task_id}"
    events = [
        _candidate_event(
            memory_id,
            timestamp=f"2026-01-01T00:00:{task_id[-1]}0+00:00",
            run_id=run_id,
        ),
        _selected_event(
            [memory_id] if selected else [],
            timestamp=f"2026-01-01T00:00:{task_id[-1]}1+00:00",
            run_id=run_id,
        ),
        _benchmark_event(
            task_id,
            resolved=resolved,
            timestamp=f"2026-01-01T00:01:{task_id[-1]}0+00:00",
            run_id=run_id,
        ),
    ]
    path.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n", encoding="utf-8")


def _add_memory(store: LongTermMemoryStore, memory_id: str) -> None:
    store.add(
        build_experience_entry(
            id=memory_id,
            content=f"experience {memory_id}",
            tier="skill",
            project_key=PROJECT_KEY,
            scope=MemoryScope.PROJECT,
        )
    )


class EvolverAttributionCliTests(unittest.TestCase):
    def test_cli_end_to_end_usage_attribution_dataset_and_writeback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            memory_dir = base / "memory"
            memory_dir.mkdir()
            store = LongTermMemoryStore.from_dir(memory_dir)
            store.load()
            _add_memory(store, "mem-good")
            _add_memory(store, "mem-bad")

            traces_dir = base / "traces"
            traces_dir.mkdir()
            tasks = [
                ("task-A", "mem-good", True, True),
                ("task-B", "mem-good", False, False),
                ("task-C", "mem-bad", True, False),
                ("task-D", "mem-bad", False, True),
            ]
            result_rows = []
            for task_id, memory_id, selected, resolved in tasks:
                trace_path = traces_dir / f"{task_id}.jsonl"
                _write_trace(trace_path, task_id, memory_id, selected=selected, resolved=resolved)
                result_rows.append({
                    "task_id": task_id,
                    "resolved": resolved,
                    "status": "resolved" if resolved else "failed",
                    "source": "humaneval",
                    "stream_id": "python",
                    "memory_mode": "shared_stream",
                    "memory_project_key": PROJECT_KEY,
                    "trace_path": str(trace_path),
                })
            results = base / "results.jsonl"
            results.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows) + "\n",
                encoding="utf-8",
            )

            usage_log = base / "usage_logs.jsonl"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main([
                    "data",
                    "build-memory-usage-log",
                    "--results", str(results),
                    "--output", str(usage_log),
                ])
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(usage_log.read_text(encoding="utf-8").splitlines()), 4)
            usage_summary = json.loads(Path(str(usage_log) + ".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(usage_summary["usage_logs"], 4)
            self.assertEqual(usage_summary["selection_events_seen"], 4)
            self.assertEqual(usage_summary["selection_events_used"], 4)
            usage_rows = [json.loads(line) for line in usage_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["run_id"] for row in usage_rows}, {f"run-{task_id}" for task_id, *_ in tasks})

            attribution = base / "memory_attribution.jsonl"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main([
                    "data",
                    "score-memory-attribution",
                    "--memory-dir", str(memory_dir),
                    "--memory-project-key", PROJECT_KEY,
                    "--usage-log", str(usage_log),
                    "--output", str(attribution),
                    "--write-back",
                    "--min-abs-value-to-write", "0",
                ])
            self.assertEqual(exit_code, 0)
            records = [json.loads(line) for line in attribution.read_text(encoding="utf-8").splitlines()]
            by_id = {row["memory_id"]: row for row in records}
            self.assertGreater(by_id["mem-good"]["value"], 0.0)
            self.assertLess(by_id["mem-bad"]["value"], 0.0)
            attribution_summary = json.loads(Path(str(attribution) + ".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(attribution_summary["write_back_updated"], 2)

            reloaded = LongTermMemoryStore.from_dir(memory_dir)
            reloaded.load()
            memory_by_id = {entry.id: entry for entry in reloaded.all(project_key=PROJECT_KEY)}
            self.assertGreater(memory_by_id["mem-good"].metadata["evolver_value"], 0.0)
            self.assertLess(memory_by_id["mem-bad"].metadata["evolver_value"], 0.0)

            writer_dataset = base / "writer.jsonl"
            writer_output = base / "writer.scored.jsonl"
            writer_dataset.write_text(
                json.dumps({"task_id": "writer-1", "saved_records": [{"id": "mem-good", "tier": "skill"}]}) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main([
                    "data",
                    "score-memory-datasets",
                    "--attribution", str(attribution),
                    "--writer-dataset", str(writer_dataset),
                    "--writer-output", str(writer_output),
                ])
            self.assertEqual(exit_code, 0)
            writer_row = json.loads(writer_output.read_text(encoding="utf-8").strip())
            self.assertGreater(writer_row["score"], 0.0)
            self.assertEqual(writer_row["created_memory_scores"], {
                "skill": {"mem-good": by_id["mem-good"]["value"]},
            })
            self.assertTrue(Path(str(writer_output) + ".summary.json").exists())

    def test_build_usage_log_strict_fails_on_bad_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trace = base / "bad.jsonl"
            trace.write_text("{bad}\n", encoding="utf-8")
            results = base / "results.jsonl"
            results.write_text(
                json.dumps({
                    "task_id": "bad-task",
                    "resolved": True,
                    "memory_project_key": PROJECT_KEY,
                    "trace_path": str(trace),
                }) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main([
                    "data",
                    "build-memory-usage-log",
                    "--results", str(results),
                    "--output", str(base / "usage_logs.jsonl"),
                    "--strict",
                ])

            self.assertNotEqual(exit_code, 0)
            self.assertIn("invalid trace JSONL", stderr.getvalue())

    def test_build_usage_log_non_strict_counts_bad_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trace = base / "bad.jsonl"
            trace.write_text("{bad}\n", encoding="utf-8")
            results = base / "results.jsonl"
            results.write_text(
                json.dumps({
                    "task_id": "bad-task",
                    "resolved": True,
                    "memory_project_key": PROJECT_KEY,
                    "trace_path": str(trace),
                }) + "\n",
                encoding="utf-8",
            )
            output = base / "usage_logs.jsonl"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([
                    "data",
                    "build-memory-usage-log",
                    "--results", str(results),
                    "--output", str(output),
                ])

            self.assertEqual(exit_code, 0)
            summary = json.loads(Path(str(output) + ".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["bad_trace"], 1)
            self.assertEqual(summary["missing_selection"], 1)

    def test_build_usage_log_resolves_missing_trace_path_from_trace_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trace_dir = base / "traces"
            trace_dir.mkdir()
            trace_path = trace_dir / "task-A.jsonl"
            _write_trace(trace_path, "task-A", "mem-good", selected=True, resolved=True)
            results = base / "results.jsonl"
            results.write_text(
                json.dumps({
                    "task_id": "task-A",
                    "resolved": True,
                    "status": "resolved",
                    "source": "humaneval",
                    "stream_id": "python",
                    "memory_mode": "shared_stream",
                    "memory_project_key": PROJECT_KEY,
                }) + "\n",
                encoding="utf-8",
            )
            usage_log = base / "usage_logs.jsonl"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([
                    "data",
                    "build-memory-usage-log",
                    "--results", str(results),
                    "--trace-dir", str(trace_dir),
                    "--output", str(usage_log),
                ])

            self.assertEqual(exit_code, 0)
            row = json.loads(usage_log.read_text(encoding="utf-8").strip())
            self.assertEqual(row["task_id"], "task-A")
            self.assertEqual(row["trace_path"], str(trace_path))

    def test_score_memory_datasets_missing_attribution_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            writer_dataset = base / "writer.jsonl"
            writer_dataset.write_text(
                json.dumps({"task_id": "writer-1", "saved_ids": ["mem-good"]}) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                exit_code = main([
                    "data",
                    "score-memory-datasets",
                    "--attribution", str(base / "missing_attribution.jsonl"),
                    "--writer-dataset", str(writer_dataset),
                    "--writer-output", str(base / "writer.scored.jsonl"),
                ])

            self.assertNotEqual(exit_code, 0)
            self.assertIn("missing_attribution.jsonl", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
