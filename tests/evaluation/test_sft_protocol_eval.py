from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.evaluation.protocol_metrics import (
    build_detailed_results,
    detect_task_type,
    evaluate_responses,
    score_response,
)
from my_agent.evaluation import model_inference
from my_agent.evaluation.model_inference import _print_progress
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
        self.assertEqual(score["runtime_tool_call_parse"], 1.0)
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
        self.assertEqual(score["runtime_tool_call_parse"], 0.0)
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
        self.assertEqual(metrics["runtime_tool_call_parse_rate"], 1.0)
        self.assertAlmostEqual(metrics["file_mention_rate"], 0.75)
        self.assertIn("repair_plan", metrics["task_counts"])


class SftProtocolOutputTests(unittest.TestCase):
    def test_protocol_prompt_disables_qwen35_thinking(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] = {}

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = dict(kwargs)
                return "prompt"

        tokenizer = Tokenizer()
        prompt = model_inference._build_prompt(
            tokenizer,
            {"system": "system", "instruction": "instruction", "input": "input"},
        )

        self.assertEqual(prompt, "prompt")
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_protocol_loader_prefers_conditional_generation_architecture(self) -> None:
        calls: list[str] = []

        class ImageTextLoader:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object) -> object:
                calls.append("conditional")
                return {"model_path": model_path, "kwargs": kwargs}

        class CausalLoader:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object) -> object:
                calls.append("causal")
                return {"model_path": model_path, "kwargs": kwargs}

        loaded = model_inference._load_generation_model(
            SimpleNamespace(
                AutoModelForImageTextToText=ImageTextLoader,
                AutoModelForCausalLM=CausalLoader,
            ),
            "Qwen/Qwen3.5-4B",
            local_files_only=True,
        )

        self.assertEqual(calls, ["conditional"])
        self.assertEqual(loaded["model_path"], "Qwen/Qwen3.5-4B")

    def test_model_inference_progress_reports_phase_count_and_speed(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            _print_progress(
                "base",
                completed=2,
                total=4,
                total_tokens=16,
                total_time=2.0,
            )

        self.assertIn("[protocol-eval] base: 2/4 ( 50.0%)", output.getvalue())
        self.assertIn("8.00 tok/s", output.getvalue())

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
            self.assertEqual(summary["sft"]["runtime_tool_call_parse_rate"], 1.0)
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
        script = Path(__file__).resolve().parents[2] / "scripts" / "train_llamafactory_lora.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn("DATASET_DIR", text)
        self.assertIn("BASE_MODEL", text)
        self.assertIn("LOCAL_MODEL_DIR", text)
        self.assertIn("OUTPUT_DIR", text)
        self.assertIn("BATCH_SIZE", text)
        self.assertIn("GRADIENT_CHECKPOINTING", text)
        self.assertIn("LEARNING_RATE", text)
        self.assertIn("NUM_TRAIN_EPOCHS", text)
        self.assertIn("CUTOFF_LEN", text)
        self.assertIn("llamafactory-cli", text)

    def test_training_script_uses_opd_warm_start_contract(self) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts" / "train_llamafactory_lora.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "dataset_info.json").write_text("{}", encoding="utf-8")
            sample = [{"instruction": "tool", "input": "{}", "output": "{}"}]
            (dataset / "train_alpaca.json").write_text(json.dumps(sample), encoding="utf-8")
            (dataset / "val_alpaca.json").write_text(json.dumps(sample), encoding="utf-8")
            capture = root / "args.json"
            fake_cli = root / "llamafactory-cli"
            fake_cli.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "if sys.argv[1:] == ['version']:\n"
                "    print('LLaMA-Factory 0.9.6.dev0')\n"
                "else:\n"
                "    open(os.environ['CAPTURE_FILE'], 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            environment = dict(os.environ)
            environment.update({
                "DATASET_DIR": str(dataset),
                "OUTPUT_DIR": str(root / "output"),
                "LLAMAFACTORY_CMD": str(fake_cli),
                "CAPTURE_FILE": str(capture),
            })

            subprocess.run(["bash", str(script)], check=True, env=environment)

            arguments = json.loads(capture.read_text(encoding="utf-8"))
            pairs = dict(zip(arguments[1::2], arguments[2::2]))
            self.assertEqual(arguments[0], "train")
            self.assertEqual(pairs["--model_name_or_path"], "Qwen/Qwen3.5-4B")
            self.assertEqual(pairs["--model_revision"], "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
            self.assertEqual(pairs["--template"], "qwen3_5_nothink")
            self.assertEqual(pairs["--lora_rank"], "16")
            self.assertEqual(pairs["--lora_alpha"], "32")
            self.assertEqual(pairs["--lora_dropout"], "0.0")
            self.assertEqual(pairs["--lora_target"], "q_proj,k_proj,v_proj,o_proj")
            self.assertEqual(pairs["--train_on_prompt"], "false")
            self.assertEqual(pairs["--mask_history"], "true")
            self.assertEqual(pairs["--cutoff_len"], "8192")
            self.assertEqual(pairs["--gradient_checkpointing"], "true")
            manifest = json.loads(
                (root / "output" / "sft_training_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["schema_version"], "agentcli-legacy-sft-training-v1"
            )
            self.assertEqual(manifest["template"], "qwen3_5_nothink")
            self.assertEqual(
                manifest["base_model"], "Qwen/Qwen3.5-4B"
            )
            self.assertEqual(
                manifest["adapter_config_hash"],
                "sha256:fc2d911dc40bbf3965a70afab1547eea4102a2c1c54bd0a44cbf5b40cbc5f91c",
            )


if __name__ == "__main__":
    unittest.main()
