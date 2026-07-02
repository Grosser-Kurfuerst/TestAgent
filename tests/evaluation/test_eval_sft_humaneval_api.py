from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests._path import add_src_to_path


add_src_to_path()

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_sft_humaneval_api.py"
SPEC = importlib.util.spec_from_file_location("eval_sft_humaneval_api_script", SCRIPT_PATH)
assert SPEC is not None
eval_sft_humaneval_api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = eval_sft_humaneval_api
SPEC.loader.exec_module(eval_sft_humaneval_api)


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        return 0


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class SlowExitProcess(FakeProcess):
    def wait(self, timeout=None):
        self.wait_calls += 1
        if not self.killed:
            raise subprocess.TimeoutExpired("llamafactory-cli", timeout)
        return 0


class LlamaFactoryConfigTests(unittest.TestCase):
    def test_writes_base_and_sft_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "base.yaml"
            sft_path = Path(tmp) / "sft.yaml"

            eval_sft_humaneval_api.write_llamafactory_config(
                base_path,
                eval_sft_humaneval_api.LlamaFactoryServeConfig(
                    model_name_or_path="Qwen/Base",
                    adapter_name_or_path=None,
                    template="qwen",
                    infer_backend="vllm",
                    finetuning_type="lora",
                    trust_remote_code=True,
                    overrides={"vllm_enforce_eager": True, "max_model_len": 4096},
                ),
            )
            eval_sft_humaneval_api.write_llamafactory_config(
                sft_path,
                eval_sft_humaneval_api.LlamaFactoryServeConfig(
                    model_name_or_path="Qwen/Base",
                    adapter_name_or_path="/tmp/adapter",
                    template="qwen",
                    infer_backend="huggingface",
                    finetuning_type="lora",
                    trust_remote_code=False,
                    overrides={},
                ),
            )

            base_text = base_path.read_text(encoding="utf-8")
            sft_text = sft_path.read_text(encoding="utf-8")

        self.assertIn('model_name_or_path: "Qwen/Base"', base_text)
        self.assertIn('template: "qwen"', base_text)
        self.assertIn('infer_backend: "vllm"', base_text)
        self.assertIn("trust_remote_code: true", base_text)
        self.assertIn("vllm_enforce_eager: true", base_text)
        self.assertIn("max_model_len: 4096", base_text)
        self.assertNotIn("adapter_name_or_path", base_text)
        self.assertIn('adapter_name_or_path: "/tmp/adapter"', sft_text)
        self.assertIn('finetuning_type: "lora"', sft_text)
        self.assertIn("trust_remote_code: false", sft_text)

    def test_parse_args_defaults_and_overrides(self) -> None:
        args = eval_sft_humaneval_api.parse_args(
            [
                "--base-model",
                "base",
                "--adapter-dir",
                "adapter",
                "--serve-override",
                "vllm_enforce_eager=true",
                "--serve-override",
                "max_model_len=4096",
            ]
        )

        self.assertEqual(args.output_dir, "outputs/sft_humaneval_api_eval")
        self.assertEqual(args.api_port, 8000)
        self.assertEqual(args.template, "qwen")
        self.assertTrue(args.trust_remote_code)
        self.assertEqual(dict(args.serve_override), {"vllm_enforce_eager": True, "max_model_len": 4096})


class LlamaFactoryServerTests(unittest.TestCase):
    def test_server_starts_waits_for_models_and_terminates(self) -> None:
        process = FakeProcess()
        popen_calls = []

        def fake_popen(*args, **kwargs):
            popen_calls.append((args, kwargs))
            return process

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llamafactory_api.yaml"
            config_path.write_text("model_name_or_path: base\n", encoding="utf-8")
            log_path = Path(tmp) / "server.log"
            with mock.patch.object(eval_sft_humaneval_api.subprocess, "Popen", side_effect=fake_popen):
                with mock.patch.object(eval_sft_humaneval_api.socket, "create_connection", side_effect=OSError):
                    with mock.patch.object(
                        eval_sft_humaneval_api.urllib.request,
                        "urlopen",
                        return_value=FakeResponse({"data": [{"id": "served-model"}]}),
                    ) as urlopen_mock:
                        with eval_sft_humaneval_api.LlamaFactoryApiServer(
                            command="llamafactory-cli",
                            config_path=config_path,
                            log_path=log_path,
                            api_host="0.0.0.0",
                            api_port=8000,
                            api_key="dummy",
                            timeout_sec=1.0,
                            shutdown_sec=1.0,
                            cuda_visible_devices="0",
                        ) as server:
                            self.assertEqual(server.base_url, "http://127.0.0.1:8000/v1")
                            self.assertEqual(server.model_ids, ["served-model"])

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertEqual(popen_calls[0][0][0], ["llamafactory-cli", "api", str(config_path)])
        env = popen_calls[0][1]["env"]
        self.assertEqual(env["API_HOST"], "0.0.0.0")
        self.assertEqual(env["API_PORT"], "8000")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertTrue(urlopen_mock.called)

    def test_server_refuses_to_start_when_port_already_has_service(self) -> None:
        existing_connection = FakeConnection()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llamafactory_api.yaml"
            config_path.write_text("model_name_or_path: base\n", encoding="utf-8")
            server = eval_sft_humaneval_api.LlamaFactoryApiServer(
                command="llamafactory-cli",
                config_path=config_path,
                log_path=Path(tmp) / "server.log",
                api_host="0.0.0.0",
                api_port=8000,
                api_key="dummy",
                timeout_sec=1.0,
                shutdown_sec=1.0,
            )
            with mock.patch.object(
                eval_sft_humaneval_api.socket,
                "create_connection",
                return_value=existing_connection,
            ) as connect_mock:
                with mock.patch.object(eval_sft_humaneval_api.subprocess, "Popen") as popen_mock:
                    with self.assertRaisesRegex(RuntimeError, "already accepting connections"):
                        server.start()

            log_exists = (Path(tmp) / "server.log").exists()

        connect_mock.assert_called_once_with(("127.0.0.1", 8000), timeout=1.0)
        popen_mock.assert_not_called()
        self.assertTrue(existing_connection.closed)
        self.assertFalse(log_exists)

    def test_server_kills_after_shutdown_timeout(self) -> None:
        process = SlowExitProcess()
        with tempfile.TemporaryDirectory() as tmp:
            server = eval_sft_humaneval_api.LlamaFactoryApiServer(
                command="llamafactory-cli",
                config_path=Path(tmp) / "config.yaml",
                log_path=Path(tmp) / "server.log",
                api_host="127.0.0.1",
                api_port=8000,
                api_key="dummy",
                timeout_sec=1.0,
                shutdown_sec=0.1,
            )
            server.process = process
            server._log_file = mock.Mock()

            server.stop()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.wait_calls, 2)


class SftHumanEvalApiOrchestrationTests(unittest.TestCase):
    def test_main_runs_base_and_sft_and_writes_comparison(self) -> None:
        summaries = {
            "base": {
                "total": 2,
                "scored": 2,
                "passed": 1,
                "failed": 1,
                "error": 0,
                "transient_excluded": 0,
                "transient_counted": 0,
                "solve_rate": 50.0,
                "end_to_end_rate": 50.0,
            },
            "sft": {
                "total": 2,
                "scored": 2,
                "passed": 2,
                "failed": 0,
                "error": 0,
                "transient_excluded": 0,
                "transient_counted": 0,
                "solve_rate": 100.0,
                "end_to_end_rate": 100.0,
            },
        }

        class FakeServer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model_ids = [f"{kwargs['config_path'].parent.name}-served"]
                self.base_url = "http://127.0.0.1:8000/v1"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

        def fake_run_benchmark_with_config(**kwargs):
            phase = Path(kwargs["output_dir"]).name
            return SimpleNamespace(summary=summaries[phase])

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--base-model",
                "Qwen/Base",
                "--adapter-dir",
                "/tmp/adapter",
                "--output-dir",
                tmp,
                "--limit",
                "2",
            ]
            with mock.patch.object(eval_sft_humaneval_api, "LlamaFactoryApiServer", FakeServer):
                with mock.patch.object(
                    eval_sft_humaneval_api,
                    "run_benchmark_with_config",
                    side_effect=fake_run_benchmark_with_config,
                ) as benchmark_mock:
                    exit_code = eval_sft_humaneval_api.main(argv)

            comparison = json.loads((Path(tmp) / "comparison_summary.json").read_text(encoding="utf-8"))
            base_config = (Path(tmp) / "base" / "llamafactory_api.yaml").read_text(encoding="utf-8")
            sft_config = (Path(tmp) / "sft" / "llamafactory_api.yaml").read_text(encoding="utf-8")
            report_exists = (Path(tmp) / "experiment_report.md").exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(benchmark_mock.call_count, 2)
        first_config = benchmark_mock.call_args_list[0].kwargs["config"]
        second_config = benchmark_mock.call_args_list[1].kwargs["config"]
        self.assertEqual(first_config.model, "base-served")
        self.assertEqual(second_config.model, "sft-served")
        self.assertEqual(benchmark_mock.call_args_list[0].kwargs["summary_scope"], "results_file")
        self.assertEqual(benchmark_mock.call_args_list[1].kwargs["summary_scope"], "results_file")
        self.assertEqual(comparison["delta"]["solve_rate"], 50.0)
        self.assertNotIn("adapter_name_or_path", base_config)
        self.assertIn('adapter_name_or_path: "/tmp/adapter"', sft_config)
        self.assertTrue(report_exists)


if __name__ == "__main__":
    unittest.main()
