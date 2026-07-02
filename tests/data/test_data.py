from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.data import (
    AlpacaOutput,
    BuildReport,
    build_humaneval,
    build_mbpp,
    export_alpaca,
    local_tasks_to_sft,
    swebench_to_sft,
    traces_to_sft,
)


# ------- helpers -----------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tool_event(run_id: str, tool: str, ok: bool, blocked: bool = False, step: int = 1) -> dict:
    return {
        "run_id": run_id,
        "event": "tool.completed",
        "payload": {
            "id": f"call_{step}",
            "name": tool,
            "arguments": "{}",
            "ok": ok,
            "content": "",
            "blocked": blocked,
            "error_code": "",
        },
    }


def _benchmark_event(run_id: str, status: str = "passed") -> dict:
    return {
        "run_id": run_id,
        "event": "benchmark_result",
        "payload": {
            "benchmark": "mbpp",
            "task_id": "1",
            "status": status,
            "scored": True,
            "test_command": "python -m pytest -q",
            "test_output": "ok",
        },
    }


# ------- Trace → SFT -------------------------------------------------


class TracesToSftTests(unittest.TestCase):
    def test_converts_successful_tool_calls_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {"run_id": "r1", "event": "repo.indexed", "payload": {"task": "fix subtract bug"}},
                _tool_event("r1", "retrieve_context", ok=True, step=1),
                _tool_event("r1", "read_file", ok=True, step=2),
                _tool_event("r1", "replace_in_file", ok=True, step=3),
                _tool_event("r1", "run_tests", ok=False, step=4),  # failed → skipped
                _tool_event("r1", "finish", ok=True, step=5),        # finish → skipped
                _benchmark_event("r1", "passed"),
            ]
            _write_jsonl(trace, events)

            output = Path(tmp) / "sft.jsonl"
            report = traces_to_sft(trace, output)

            self.assertEqual(report.written, 3)
            samples = _read_jsonl(output)
            self.assertEqual(len(samples), 3)

            for sample in samples:
                self.assertIn("instruction", sample)
                self.assertIn("input", sample)
                self.assertIn("output", sample)
                self.assertIn("metadata", sample)

            tools = [s["output"]["tool"] for s in samples]
            self.assertEqual(tools, ["retrieve_context", "read_file", "replace_in_file"])
            self.assertNotIn("finish", tools)

    def test_failed_benchmark_trace_outputs_no_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {"run_id": "r1", "event": "repo.indexed", "payload": {"task": "fix subtract bug"}},
                _tool_event("r1", "read_file", ok=True, step=1),
                _tool_event("r1", "write_file", ok=True, step=2),
                _benchmark_event("r1", "failed"),
            ]
            _write_jsonl(trace, events)

            output = Path(tmp) / "sft.jsonl"
            report = traces_to_sft(trace, output)

            self.assertEqual(report.written, 0)
            self.assertEqual(_read_jsonl(output), [])

    def test_trace_without_benchmark_result_outputs_no_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {"run_id": "r1", "event": "repo.indexed", "payload": {"task": "fix subtract bug"}},
                _tool_event("r1", "read_file", ok=True, step=1),
            ]
            _write_jsonl(trace, events)

            output = Path(tmp) / "sft.jsonl"
            report = traces_to_sft(trace, output)

            self.assertEqual(report.written, 0)
            self.assertEqual(_read_jsonl(output), [])

    def test_empty_traces_produce_zero_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            _write_jsonl(trace, [{"run_id": "r1", "event": "repo.indexed", "payload": {"task": "t"}}])
            output = Path(tmp) / "sft.jsonl"
            report = traces_to_sft(trace, output)
            self.assertEqual(report.written, 0)

    def test_handles_missing_trace_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sft.jsonl"
            with self.assertRaises(FileNotFoundError):
                traces_to_sft(Path(tmp) / "nonexistent.jsonl", output)

    def test_skips_invalid_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                "not json\n"
                + json.dumps(_tool_event("r1", "grep", ok=True, step=1)) + "\n"
                + json.dumps(_benchmark_event("r1", "passed")) + "\n"
                + "\n",  # empty line
                encoding="utf-8",
            )
            output = Path(tmp) / "sft.jsonl"
            report = traces_to_sft(trace, output)
            self.assertEqual(report.written, 1)
            self.assertEqual(report.skipped, 1)
            self.assertEqual(len(report.errors), 1)

    def test_history_capped_at_six(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {"run_id": "r1", "event": "repo.indexed", "payload": {"task": "t"}},
            ]
            for i in range(10):
                events.append(_tool_event("r1", "grep", ok=True, step=i + 1))
            events.append(_benchmark_event("r1", "passed"))
            _write_jsonl(trace, events)

            output = Path(tmp) / "sft.jsonl"
            traces_to_sft(trace, output)
            samples = _read_jsonl(output)
            # The last few samples should have history capped at ≤6
            last_sample = samples[-1]
            self.assertLessEqual(len(last_sample["input"]["history"]), 6)


# ------- Local tasks → SFT -------------------------------------------


class LocalTasksToSftTests(unittest.TestCase):
    def test_converts_valid_task_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "tasks.jsonl"
            _write_jsonl(
                tasks_file,
                [
                    {
                        "id": "task-1",
                        "source": "local",
                        "repo": "examples/sample_repo",
                        "task": "Fix the subtract function.",
                        "test_command": "python -m pytest -q",
                    },
                ],
            )
            output = Path(tmp) / "strategy_sft.jsonl"
            report = local_tasks_to_sft(tasks_file, output)

            self.assertEqual(report.written, 1)
            samples = _read_jsonl(output)
            sample = samples[0]
            self.assertIn("制定并执行最小工具调用策略", sample["instruction"])
            self.assertEqual(sample["input"]["repo"], "examples/sample_repo")
            self.assertEqual(sample["input"]["task"], "Fix the subtract function.")
            self.assertIsInstance(sample["output"]["strategy"], list)
            tools_in_strategy = [s["tool"] for s in sample["output"]["strategy"]]
            self.assertIn("retrieve_context", tools_in_strategy)
            self.assertIn("read_file", tools_in_strategy)
            self.assertIn("run_tests", tools_in_strategy)

    def test_skips_records_without_repo_or_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "tasks.jsonl"
            _write_jsonl(
                tasks_file,
                [
                    {"id": "bad-1", "source": "local", "repo": "", "task": ""},
                    {"id": "bad-2", "source": "local", "repo": "r", "task": None},
                    {"id": "good-1", "source": "local", "repo": "r", "task": "t"},
                ],
            )
            output = Path(tmp) / "sft.jsonl"
            report = local_tasks_to_sft(tasks_file, output, strict=False)
            self.assertEqual(report.written, 1)
            self.assertEqual(report.skipped, 2)
            self.assertEqual(len(report.errors), 2)

    def test_strict_mode_rejects_records_without_repo_or_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "tasks.jsonl"
            _write_jsonl(tasks_file, [{"id": "bad-1", "source": "local", "repo": "", "task": ""}])
            output = Path(tmp) / "sft.jsonl"
            with self.assertRaises(ValueError):
                local_tasks_to_sft(tasks_file, output)

    def test_single_json_object_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "task.json"
            tasks_file.write_text(
                json.dumps({"id": "t1", "source": "local", "repo": "r", "task": "t"}),
                encoding="utf-8",
            )
            output = Path(tmp) / "sft.jsonl"
            report = local_tasks_to_sft(tasks_file, output)
            self.assertEqual(report.written, 1)


# ------- SWE-bench → SFT ---------------------------------------------


class SwebenchToSftTests(unittest.TestCase):
    def test_plan_mode_produces_plan_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "swebench.jsonl"
            _write_jsonl(
                tasks_file,
                [
                    {
                        "id": "django__django-1",
                        "source": "SWE-bench_Lite",
                        "repo_name": "django/django",
                        "base_commit": "abc123",
                        "task": "Fix auth bug",
                        "patch": (
                            "diff --git a/auth.py b/auth.py\n"
                            "--- a/auth.py\n"
                            "+++ b/auth.py\n"
                            "@@ -1 +1 @@\n"
                            "-old\n"
                            "+new\n"
                        ),
                        "test_patch": "",
                    },
                ],
            )
            output = Path(tmp) / "sft.jsonl"
            report = swebench_to_sft(tasks_file, output, mode="plan")

            self.assertEqual(report.written, 1)
            samples = _read_jsonl(output)
            self.assertIn("plan", samples[0]["output"])
            self.assertIn("auth.py", samples[0]["output"]["plan"])
            self.assertEqual(samples[0]["metadata"]["source"], "SWE-bench_Lite")

    def test_patch_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "swebench.jsonl"
            _write_jsonl(
                tasks_file,
                [
                    {
                        "id": "x-1",
                        "source": "SWE-bench_Lite",
                        "repo_name": "x/x",
                        "base_commit": "abc",
                        "task": "Fix x",
                        "patch": "diff --git a/x.py b/x.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
                    },
                ],
            )
            output = Path(tmp) / "sft.jsonl"
            with self.assertRaises(ValueError):
                swebench_to_sft(tasks_file, output, mode="patch")

    def test_non_swebench_rows_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "mixed.jsonl"
            _write_jsonl(
                tasks_file,
                [
                    {"id": "mbpp_1", "source": "MBPP", "repo": "r", "task": "t"},
                    {"id": "swe_1", "source": "SWE-bench_Lite", "repo_name": "x", "task": "t", "patch": "p"},
                ],
            )
            output = Path(tmp) / "sft.jsonl"
            report = swebench_to_sft(tasks_file, output, mode="plan")
            self.assertEqual(report.written, 1)
            self.assertEqual(report.skipped, 1)


# ------- Alpaca export ------------------------------------------------


class ExportAlpacaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sft_file = self.tmp / "sft.jsonl"
        _write_jsonl(
            self.sft_file,
            [
                {
                    "instruction": "根据任务选择工具。",
                    "input": {"task": "fix bug"},
                    "output": {"tool": "read_file", "arguments": {"path": "x.py"}, "reason": "check"},
                    "metadata": {"id": "1"},
                },
                {
                    "instruction": "根据任务选择工具。",
                    "input": {"task": "fix another bug"},
                    "output": {
                        "tool": "replace_in_file",
                        "arguments": {"path": "y.py", "old": "a", "new": "b"},
                        "reason": "fix",
                    },
                    "metadata": {"id": "2"},
                },
            ],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_produces_alpaca_format(self) -> None:
        result = export_alpaca(
            input_files=[self.sft_file],
            output_dir=self.tmp / "alpaca",
            train_ratio=0.5,
            seed=42,
        )

        self.assertEqual(result.total, 2)
        self.assertEqual(result.train + result.val, 2)
        self.assertIn(str(self.sft_file.name), result.source_counts)

        # verify alpaca JSON structure
        train_records = json.loads(result.train_path.read_text(encoding="utf-8"))
        self.assertIsInstance(train_records, list)
        for rec in train_records:
            self.assertIn("system", rec)
            self.assertIn("instruction", rec)
            self.assertIn("input", rec)
            self.assertIn("output", rec)

    def test_dataset_info_format(self) -> None:
        result = export_alpaca(
            input_files=[self.sft_file],
            output_dir=self.tmp / "alpaca",
            train_ratio=0.5,
        )
        info = json.loads(result.dataset_info_path.read_text(encoding="utf-8"))
        self.assertIn("coding_agent_train", info)
        self.assertEqual(info["coding_agent_train"]["formatting"], "alpaca")
        self.assertEqual(
            info["coding_agent_train"]["columns"]["prompt"],
            "instruction",
        )

    def test_dataset_stats_format(self) -> None:
        result = export_alpaca(
            input_files=[self.sft_file],
            output_dir=self.tmp / "alpaca",
            train_ratio=0.5,
        )
        stats = json.loads(result.stats_path.read_text(encoding="utf-8"))
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["train"] + stats["val"], 2)
        self.assertEqual(stats["skipped"], 0)
        self.assertIn("source_counts", stats)

    def test_missing_input_files_are_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            export_alpaca(
                input_files=[self.tmp / "nonexistent.jsonl", self.sft_file],
                output_dir=self.tmp / "alpaca",
                train_ratio=0.5,
            )

    def test_empty_inputs_handled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            result = export_alpaca(
                input_files=[empty],
                output_dir=Path(d) / "out",
            )
            self.assertEqual(result.total, 0)
            self.assertEqual(result.train, 0)
            self.assertEqual(result.val, 0)

    def test_invalid_sft_records_are_errors(self) -> None:
        bad_file = self.tmp / "bad.jsonl"
        _write_jsonl(bad_file, [{"instruction": "missing output", "input": {}}])

        with self.assertRaises(ValueError):
            export_alpaca(input_files=[bad_file], output_dir=self.tmp / "alpaca")

    def test_invalid_json_lines_are_errors(self) -> None:
        bad_file = self.tmp / "bad_json.jsonl"
        bad_file.write_text("{not json}\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            export_alpaca(input_files=[bad_file], output_dir=self.tmp / "alpaca")

    def test_non_strict_reports_and_skips_bad_sft_rows(self) -> None:
        mixed_file = self.tmp / "mixed.jsonl"
        valid = {
            "instruction": "根据任务选择工具。",
            "input": {"task": "fix bug"},
            "output": {"tool": "read_file", "arguments": {"path": "x.py"}, "reason": "check"},
        }
        invalid_schema = {"instruction": "missing output", "input": {}}
        mixed_file.write_text(
            json.dumps(valid, ensure_ascii=False)
            + "\n{not json}\n"
            + json.dumps(invalid_schema, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        result = export_alpaca(input_files=[mixed_file], output_dir=self.tmp / "alpaca", strict=False)

        self.assertEqual(result.total, 1)
        self.assertEqual(result.skipped, 2)
        self.assertEqual(len(result.errors), 2)
        stats = json.loads(result.stats_path.read_text(encoding="utf-8"))
        self.assertEqual(stats["skipped"], 2)
        self.assertEqual(len(stats["errors"]), 2)

    def test_one_record_goes_to_train_split(self) -> None:
        one_file = self.tmp / "one.jsonl"
        _write_jsonl(
            one_file,
            [
                {
                    "instruction": "根据任务选择工具。",
                    "input": {"task": "fix bug"},
                    "output": {"tool": "read_file", "arguments": {"path": "x.py"}, "reason": "check"},
                }
            ],
        )

        result = export_alpaca(input_files=[one_file], output_dir=self.tmp / "alpaca")

        self.assertEqual(result.total, 1)
        self.assertEqual(result.train, 1)
        self.assertEqual(result.val, 0)

    def test_invalid_train_ratio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            export_alpaca(input_files=[self.sft_file], output_dir=self.tmp / "alpaca", train_ratio=1.0)


# ------- BuildReport --------------------------------------------------


class BuildReportTests(unittest.TestCase):
    def test_render_includes_all_fields(self) -> None:
        result = BuildReport(
            source="MBPP",
            total=6,
            written=5,
            skipped=1,
            tasks_path=Path("/tmp/tasks.jsonl"),
            sft_path=Path("/tmp/sft.jsonl"),
            repo_dir=Path("/tmp/repos"),
        )
        rendered = result.render()
        self.assertIn("MBPP", rendered)
        self.assertIn("5 sample", rendered)
        self.assertIn("tasks:", rendered)
        self.assertIn("sft:", rendered)
        self.assertIn("repos:", rendered)


# ------- SFT sample format validation ---------------------------------


class SftFormatValidationTests(unittest.TestCase):
    """Each SFT sample must have instruction, input, and output fields."""

    def test_trace_sft_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            _write_jsonl(
                trace,
                [
                    {"run_id": "r1", "event": "repo.indexed", "payload": {"task": "t"}},
                    _tool_event("r1", "read_file", ok=True, step=1),
                    _benchmark_event("r1", "passed"),
                ],
            )
            sft_output = Path(tmp) / "sft.jsonl"
            traces_to_sft(trace, sft_output)
            samples = _read_jsonl(sft_output)

            for sample in samples:
                with self.subTest(sample=sample):
                    self.assertIsInstance(sample["instruction"], str)
                    self.assertIsInstance(sample["input"], dict)
                    self.assertIsInstance(sample["output"], dict)

    def test_tasks_to_sft_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks.jsonl"
            _write_jsonl(tasks, [{"id": "1", "source": "local", "repo": "r", "task": "t"}])
            sft_output = Path(tmp) / "sft.jsonl"
            local_tasks_to_sft(tasks, sft_output)
            samples = _read_jsonl(sft_output)

            for sample in samples:
                with self.subTest(sample=sample):
                    self.assertIsInstance(sample["instruction"], str)
                    self.assertIsInstance(sample["input"], dict)
                    self.assertIsInstance(sample["output"], dict)


# ------- Offline seed samples -----------------------------------------


class OfflineSeedTests(unittest.TestCase):
    """When the network is unavailable, builders should fall back to bundled seed data."""

    def test_mbpp_internally_produces_valid_output(self) -> None:
        """Smoke test: build_mbpp should complete with offline seeds when offline."""
        with tempfile.TemporaryDirectory() as tmp:
            result = build_mbpp(output_dir=tmp, limit=3, split="test")

            self.assertGreaterEqual(result.written, 1)
            self.assertTrue(result.tasks_path.exists(), result.tasks_path)
            self.assertTrue(result.sft_path.exists(), result.sft_path)

            sft_samples = _read_jsonl(result.sft_path)
            for s in sft_samples:
                self.assertIn("instruction", s)
                self.assertIn("input", s)
                self.assertIn("output", s)
                self.assertEqual(s["output"]["tool"], "write_file")

    def test_humaneval_internally_produces_valid_output(self) -> None:
        """Smoke test: build_humaneval should complete with offline seeds when offline."""
        with tempfile.TemporaryDirectory() as tmp:
            result = build_humaneval(output_dir=tmp, limit=3, split="test")

            self.assertGreaterEqual(result.written, 1)
            self.assertTrue(result.tasks_path.exists(), result.tasks_path)
            self.assertTrue(result.sft_path.exists(), result.sft_path)

            sft_samples = _read_jsonl(result.sft_path)
            for s in sft_samples:
                self.assertIn("instruction", s)
                self.assertIn("output", s)
                self.assertEqual(s["output"]["tool"], "write_file")


if __name__ == "__main__":
    unittest.main()
