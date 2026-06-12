from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.evaluation.protocol_metrics import (
    build_detailed_results,
    detect_task_type,
    evaluate_responses,
    score_response,
)
from my_agent.evaluation.protocol_runner import load_response_file, run_evaluation


def _alpaca_sample(instruction: str, user_input: dict, output: dict) -> dict:
    return {
        "system": "system prompt",
        "instruction": instruction,
        "input": json.dumps(user_input, ensure_ascii=False, indent=2),
        "output": json.dumps(output, ensure_ascii=False, indent=2),
    }


def _read_file_response() -> dict:
    return {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect"}


class SftProtocolMetricTests(unittest.TestCase):
    def test_detects_task_types_from_reference_output(self) -> None:
        tool_sample = _alpaca_sample(
            "Choose the next tool call.",
            {"task": "inspect calculator.py"},
            {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect"},
        )
        strategy_sample = _alpaca_sample(
            "制定并执行最小工具调用策略。",
            {"task": "fix"},
            {"strategy": [{"tool": "retrieve_context"}]},
        )
        plan_sample = _alpaca_sample(
            "制定代码仓库修复计划。",
            {"task": "fix src/foo.py"},
            {"plan": "Inspect src/foo.py", "validation": "Run tests"},
        )

        self.assertEqual(detect_task_type(tool_sample), "tool_call")
        self.assertEqual(detect_task_type(strategy_sample), "strategy")
        self.assertEqual(detect_task_type(plan_sample), "repair_plan")

    def test_scores_json_fields_tool_and_file_mentions(self) -> None:
        sample = _alpaca_sample(
            "Choose the next tool call.",
            {"task": "inspect calculator.py"},
            {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect implementation"},
        )
        response = json.dumps(
            {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect implementation"},
            ensure_ascii=False,
        )

        score = score_response(response, sample)

        self.assertTrue(score["json_valid"])
        self.assertEqual(score["field_hit_rate"], 1.0)
        self.assertEqual(score["tool_accuracy"], 1.0)
        self.assertEqual(score["file_mention_rate"], 1.0)
        self.assertGreater(score["rouge_l"], 0.5)

    def test_invalid_json_counts_as_tool_accuracy_failure(self) -> None:
        sample = _alpaca_sample(
            "Choose the next tool call.",
            {"task": "inspect calculator.py"},
            {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect implementation"},
        )

        score = score_response("read calculator.py first", sample)

        self.assertFalse(score["json_valid"])
        self.assertEqual(score["field_hit_rate"], 0.0)
        self.assertEqual(score["tool_accuracy"], 0.0)
        self.assertEqual(score["file_mention_rate"], 1.0)

    def test_evaluate_responses_aggregates_required_metrics(self) -> None:
        samples = [
            _alpaca_sample(
                "Choose the next tool call.",
                {"task": "inspect calculator.py"},
                {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect"},
            ),
            _alpaca_sample(
                "制定代码仓库修复计划。",
                {"task": "fix src/foo.py"},
                {"plan": "Inspect src/foo.py and tests/test_foo.py", "validation": "Run pytest"},
            ),
        ]
        responses = [
            json.dumps({"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect"}),
            json.dumps({"plan": "Inspect src/foo.py", "validation": "Run pytest"}),
        ]

        metrics = evaluate_responses(responses, samples)

        self.assertEqual(metrics["n_samples"], 2)
        self.assertEqual(metrics["json_valid_rate"], 1.0)
        self.assertEqual(metrics["field_hit_rate"], 1.0)
        self.assertEqual(metrics["tool_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["file_mention_rate"], 0.75)
        self.assertIn("repair_plan", metrics["task_counts"])


class SftProtocolOutputTests(unittest.TestCase):
    def test_builds_detailed_results_with_required_fields(self) -> None:
        samples = [
            _alpaca_sample(
                "Choose the next tool call.",
                {"task": "inspect calculator.py"},
                {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect"},
            )
        ]
        base = ["not json"]
        sft = [json.dumps({"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect"})]

        details = build_detailed_results(samples, base, sft)

        self.assertEqual(len(details), 1)
        detail = details[0]
        self.assertEqual(detail["task_type"], "tool_call")
        self.assertIn("reference_output", detail)
        self.assertEqual(detail["base_output"], "not json")
        self.assertIn("sft_metrics", detail)

    def test_metric_only_runner_writes_summary_details_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            val_data = base / "val_alpaca.json"
            output_dir = base / "eval"
            samples = [
                _alpaca_sample(
                    "Choose the next tool call.",
                    {"task": "inspect calculator.py"},
                    {"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "inspect"},
                )
            ]
            val_data.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
            base_responses = base / "base_responses.json"
            sft_responses = base / "sft_responses.json"
            base_responses.write_text(json.dumps(["not json"], ensure_ascii=False), encoding="utf-8")
            sft_responses.write_text(
                json.dumps(
                    [_read_file_response()],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = run_evaluation(
                val_data=val_data,
                base_responses_path=base_responses,
                sft_responses_path=sft_responses,
                output_dir=output_dir,
            )

            self.assertIn("base", summary)
            summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
            self.assertIn("base", summary)
            self.assertIn("sft", summary)
            self.assertIn("delta", summary)
            self.assertNotIn("passed", summary)
            details = json.loads((output_dir / "detailed_results.json").read_text(encoding="utf-8"))
            self.assertEqual(details[0]["task_type"], "tool_call")
            report = (output_dir / "experiment_report.md").read_text(encoding="utf-8")
            self.assertIn("does not prove complex real-world code repair ability", report)

    def test_response_file_accepts_structured_json_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response_file = Path(tmp) / "responses.json"
            response_file.write_text(
                json.dumps(
                    {"responses": [_read_file_response()]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            responses = load_response_file(response_file, expected=1)

            self.assertEqual(json.loads(responses[0])["tool"], "read_file")

    def test_response_file_accepts_single_top_level_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response_file = Path(tmp) / "response.json"
            response_file.write_text(
                json.dumps(_read_file_response()),
                encoding="utf-8",
            )

            responses = load_response_file(response_file, expected=1)

            self.assertEqual(json.loads(responses[0])["arguments"]["path"], "calculator.py")

    def test_response_file_rejects_primitive_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response_file = Path(tmp) / "responses.json"
            response_file.write_text(json.dumps([123]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "response #1 must be a string"):
                load_response_file(response_file, expected=1)

    def test_training_script_exposes_phase7_knobs(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "train_llamafactory_lora.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn("DATASET_DIR", text)
        self.assertIn("BASE_MODEL", text)
        self.assertIn("LOCAL_MODEL_DIR", text)
        self.assertIn("OUTPUT_DIR", text)
        self.assertIn("BATCH_SIZE", text)
        self.assertIn("LEARNING_RATE", text)
        self.assertIn("NUM_TRAIN_EPOCHS", text)
        self.assertIn("CUTOFF_LEN", text)
        self.assertIn("llamafactory-cli", text)


if __name__ == "__main__":
    unittest.main()
