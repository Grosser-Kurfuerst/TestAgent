#!/usr/bin/env python3
from __future__ import annotations

"""Compare base vs SFT LoRA models on HumanEval through LLaMA-Factory API."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_humaneval import _benchmark_spec, load_humaneval  # noqa: E402
from my_agent.config import AgentConfig  # noqa: E402
from my_agent.evaluation.agent_benchmark import run_benchmark_with_config  # noqa: E402


@dataclass(frozen=True)
class LlamaFactoryServeConfig:
    model_name_or_path: str
    adapter_name_or_path: str | None
    template: str
    infer_backend: str
    finetuning_type: str
    trust_remote_code: bool
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_name_or_path": self.model_name_or_path,
            "template": self.template,
            "infer_backend": self.infer_backend,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.adapter_name_or_path:
            payload["adapter_name_or_path"] = self.adapter_name_or_path
            payload["finetuning_type"] = self.finetuning_type
        payload.update(self.overrides)
        return payload


class LlamaFactoryApiServer:
    def __init__(
        self,
        *,
        command: str,
        config_path: Path,
        log_path: Path,
        api_host: str,
        api_port: int,
        api_key: str,
        timeout_sec: float,
        shutdown_sec: float,
        cuda_visible_devices: str | None = None,
    ) -> None:
        self.command = command
        self.config_path = config_path
        self.log_path = log_path
        self.api_host = api_host
        self.api_port = api_port
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.shutdown_sec = shutdown_sec
        self.cuda_visible_devices = cuda_visible_devices
        self.process: subprocess.Popen[str] | None = None
        self._log_file: Any | None = None
        self.model_ids: list[str] = []

    @property
    def base_url(self) -> str:
        return f"http://{_client_host(self.api_host)}:{self.api_port}/v1"

    def __enter__(self) -> "LlamaFactoryApiServer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def start(self) -> None:
        ensure_api_port_free(api_host=self.api_host, api_port=self.api_port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["API_HOST"] = self.api_host
        env["API_PORT"] = str(self.api_port)
        if self.api_key:
            env["API_KEY"] = self.api_key
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        try:
            self.process = subprocess.Popen(
                [self.command, "api", str(self.config_path)],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as exc:
            self._close_log()
            raise RuntimeError(f"Missing {self.command}. Install LLaMA-Factory before running this script.") from exc

        try:
            self.model_ids = wait_for_openai_models(
                base_url=self.base_url,
                api_key=self.api_key,
                process=self.process,
                timeout_sec=self.timeout_sec,
                log_path=self.log_path,
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.shutdown_sec)
        self._close_log()

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base vs SFT LoRA models on HumanEval through sequential LLaMA-Factory API servers."
    )
    parser.add_argument("--base-model", required=True, help="Base model path or HuggingFace model id.")
    parser.add_argument("--adapter-dir", required=True, help="LoRA adapter directory or LLaMA-Factory output directory.")
    parser.add_argument("--output-dir", default="outputs/sft_humaneval_api_eval", help="Evaluation artifact directory.")
    parser.add_argument("--limit", type=int, default=10, help="Number of HumanEval tasks to evaluate.")
    parser.add_argument("--start", type=int, default=0, help="Start index for resumable evaluation.")
    parser.add_argument("--split", default="test", help="HumanEval dataset split.")
    parser.add_argument("--max-steps", type=int, default=10, help="Agent max steps per task.")
    parser.add_argument("--llm-retries", type=int, default=2, help="Transient LLM retry count.")
    parser.add_argument("--retry-delay-sec", type=float, default=2.0, help="Delay between transient retries.")
    parser.add_argument(
        "--count-transient-errors",
        action="store_true",
        help="Count transient API errors in the solve-rate denominator.",
    )
    parser.add_argument("--llamafactory-cmd", default="llamafactory-cli", help="LLaMA-Factory CLI command.")
    parser.add_argument("--api-host", default="127.0.0.1", help="Host passed to API_HOST.")
    parser.add_argument("--api-port", type=int, default=8000, help="Port passed to API_PORT.")
    parser.add_argument("--api-key", default="dummy", help="Bearer token used by the Agent client.")
    parser.add_argument("--server-timeout-sec", type=float, default=600.0, help="Seconds to wait for API readiness.")
    parser.add_argument("--server-shutdown-sec", type=float, default=30.0, help="Seconds to wait after terminate.")
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES for served models.")
    parser.add_argument("--template", default="qwen", help="LLaMA-Factory chat template.")
    parser.add_argument("--infer-backend", default="huggingface", help="LLaMA-Factory inference backend.")
    parser.add_argument("--finetuning-type", default="lora", help="Fine-tuning type for the SFT adapter server.")
    parser.add_argument("--base-api-model", help="Model name sent in base chat-completions requests.")
    parser.add_argument("--sft-api-model", help="Model name sent in SFT chat-completions requests.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Agent LLM temperature.")
    parser.add_argument("--command-timeout", type=int, default=60, help="Agent shell command timeout in seconds.")
    parser.add_argument(
        "--serve-override",
        action="append",
        default=[],
        type=parse_serve_override,
        metavar="KEY=VALUE",
        help="Extra LLaMA-Factory YAML key/value. Can be passed multiple times.",
    )
    trust = parser.add_mutually_exclusive_group()
    trust.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    trust.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args(argv)


def parse_serve_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--serve-override must use KEY=VALUE syntax.")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("--serve-override key must not be empty.")
    return key, parse_scalar(raw.strip())


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def write_llamafactory_config(path: Path, config: LlamaFactoryServeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {_yaml_scalar(value)}" for key, value in config.to_dict().items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def wait_for_openai_models(
    *,
    base_url: str,
    api_key: str,
    process: subprocess.Popen[str],
    timeout_sec: float,
    log_path: Path,
) -> list[str]:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"LLaMA-Factory API exited before becoming ready. See {log_path}.")
        try:
            return fetch_model_ids(base_url=base_url, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 - readiness loop records and retries all startup failures
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {base_url}/models. Last error: {last_error}. See {log_path}.")


def fetch_model_ids(*, base_url: str, api_key: str) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body[:500]}") from exc
    parsed = json.loads(body)
    data = parsed.get("data", [])
    if not isinstance(data, list):
        return []
    return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]


def ensure_api_port_free(*, api_host: str, api_port: int, timeout_sec: float = 1.0) -> None:
    probe_host = _client_host(api_host)
    try:
        connection = socket.create_connection((probe_host, api_port), timeout=timeout_sec)
    except OSError:
        return
    connection.close()
    raise RuntimeError(
        f"Refusing to start LLaMA-Factory API because {probe_host}:{api_port} is already accepting "
        "connections. Stop the existing service or choose another --api-port to avoid evaluating "
        "against a stale model server."
    )


def run_phase(args: argparse.Namespace, *, phase: str, adapter_dir: str | None) -> dict[str, float | int]:
    output_dir = Path(args.output_dir)
    phase_dir = output_dir / phase
    overrides = dict(args.serve_override)
    serve_config = LlamaFactoryServeConfig(
        model_name_or_path=args.base_model,
        adapter_name_or_path=adapter_dir,
        template=args.template,
        infer_backend=args.infer_backend,
        finetuning_type=args.finetuning_type,
        trust_remote_code=args.trust_remote_code,
        overrides=overrides,
    )
    config_path = phase_dir / "llamafactory_api.yaml"
    write_llamafactory_config(config_path, serve_config)

    explicit_model = args.base_api_model if phase == "base" else args.sft_api_model
    with LlamaFactoryApiServer(
        command=args.llamafactory_cmd,
        config_path=config_path,
        log_path=phase_dir / "server.log",
        api_host=args.api_host,
        api_port=args.api_port,
        api_key=args.api_key,
        timeout_sec=args.server_timeout_sec,
        shutdown_sec=args.server_shutdown_sec,
        cuda_visible_devices=args.cuda_visible_devices,
    ) as server:
        api_model = explicit_model or first_model_id(server.model_ids) or args.base_model
        agent_config = AgentConfig(
            provider="openai",
            api_key=args.api_key,
            base_url=server.base_url,
            model=api_model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            command_timeout=args.command_timeout,
            trace_dir=phase_dir / "traces",
            use_fake_llm=False,
        )
        run = run_benchmark_with_config(
            config=agent_config,
            output_dir=phase_dir,
            spec=_benchmark_spec(),
            load_rows=load_humaneval,
            split=args.split,
            start=args.start,
            limit=args.limit,
            max_steps=args.max_steps,
            llm_retries=args.llm_retries,
            retry_delay_sec=args.retry_delay_sec,
            count_transient_errors=args.count_transient_errors,
            write_summary=True,
            summary_scope="results_file",
        )
    return run.summary


def first_model_id(model_ids: list[str]) -> str | None:
    return model_ids[0] if model_ids else None


def build_comparison(base: dict[str, float | int], sft: dict[str, float | int]) -> dict[str, Any]:
    keys = sorted(set(base) | set(sft))
    delta: dict[str, float | int | None] = {}
    for key in keys:
        base_value = base.get(key)
        sft_value = sft.get(key)
        if isinstance(base_value, (int, float)) and isinstance(sft_value, (int, float)):
            delta[key] = sft_value - base_value
        else:
            delta[key] = None
    return {"base": base, "sft": sft, "delta": delta}


def render_report(args: argparse.Namespace, comparison: dict[str, Any]) -> str:
    base = comparison["base"]
    sft = comparison["sft"]
    delta = comparison["delta"]
    command = " ".join(_shell_quote(part) for part in [sys.executable, *sys.argv])
    lines = [
        "# HumanEval Base-vs-SFT API Evaluation",
        "",
        "This report compares full Agent HumanEval runs served through sequential LLaMA-Factory OpenAI-compatible APIs.",
        "",
        "## Configuration",
        "",
        f"- Base model: `{args.base_model}`",
        f"- Adapter dir: `{args.adapter_dir}`",
        f"- Split/start/limit: `{args.split}` / `{args.start}` / `{args.limit}`",
        f"- Max steps: `{args.max_steps}`",
        f"- API: `http://{_client_host(args.api_host)}:{args.api_port}/v1`",
        f"- Template/backend: `{args.template}` / `{args.infer_backend}`",
        "",
        "## Results",
        "",
        "| Metric | Base | SFT | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| solve_rate | {_format_percent(base.get('solve_rate'))} | {_format_percent(sft.get('solve_rate'))} | {_format_delta_percent(delta.get('solve_rate'))} |",
        f"| end_to_end_rate | {_format_percent(base.get('end_to_end_rate'))} | {_format_percent(sft.get('end_to_end_rate'))} | {_format_delta_percent(delta.get('end_to_end_rate'))} |",
        f"| passed | {base.get('passed', 0)} | {sft.get('passed', 0)} | {_format_delta_number(delta.get('passed'))} |",
        f"| failed | {base.get('failed', 0)} | {sft.get('failed', 0)} | {_format_delta_number(delta.get('failed'))} |",
        f"| error | {base.get('error', 0)} | {sft.get('error', 0)} | {_format_delta_number(delta.get('error'))} |",
        f"| transient_excluded | {base.get('transient_excluded', 0)} | {sft.get('transient_excluded', 0)} | {_format_delta_number(delta.get('transient_excluded'))} |",
        f"| transient_counted | {base.get('transient_counted', 0)} | {sft.get('transient_counted', 0)} | {_format_delta_number(delta.get('transient_counted'))} |",
        "",
        "## Artifacts",
        "",
        "- `base/results.jsonl`, `base/summary.json`, `base/server.log`",
        "- `sft/results.jsonl`, `sft/summary.json`, `sft/server.log`",
        "- `comparison_summary.json`",
        "",
        "## Reproduction",
        "",
        "```bash",
        command,
        "```",
    ]
    return "\n".join(lines) + "\n"


def print_comparison(comparison: dict[str, Any], output_dir: Path) -> None:
    print()
    print("HumanEval Base-vs-SFT Comparison")
    print("--------------------------------")
    for key in ("solve_rate", "end_to_end_rate"):
        print(
            f"{key:<20} "
            f"base={_format_percent(comparison['base'].get(key)):>8} "
            f"sft={_format_percent(comparison['sft'].get(key)):>8} "
            f"delta={_format_delta_percent(comparison['delta'].get(key)):>8}"
        )
    print(f"Results written to {output_dir}")


def _format_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:.1f}%"


def _format_delta_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:+.1f}%"


def _format_delta_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:+g}"


def _client_host(api_host: str) -> str:
    if api_host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return api_host


def _shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Base HumanEval run ===")
    base_summary = run_phase(args, phase="base", adapter_dir=None)
    print("\n=== SFT HumanEval run ===")
    sft_summary = run_phase(args, phase="sft", adapter_dir=args.adapter_dir)

    comparison = build_comparison(base_summary, sft_summary)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "experiment_report.md").write_text(render_report(args, comparison), encoding="utf-8")
    print_comparison(comparison, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
