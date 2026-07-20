"""Protocol evaluation orchestration and artifact writing."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from my_agent.evaluation.model_inference import generate_base_responses, generate_sft_responses
from my_agent.evaluation.protocol_metrics import (
    METRIC_KEYS,
    build_detailed_results,
    compare_metrics,
    evaluate_responses,
)


def load_samples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    sample_path = Path(path)
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Validation data must be a JSON array.")
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Sample #{index} must be a JSON object.")
        for field in ("instruction", "input", "output"):
            if field not in item:
                raise ValueError(f"Sample #{index} missing required field: {field}")
        samples.append(item)
    if limit is not None:
        return samples[:limit]
    return samples


def load_response_file(path: str | Path, expected: int) -> list[str]:
    response_path = Path(path)
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "responses" in payload:
        payload = payload.get("responses")
    elif expected == 1 and isinstance(payload, (str, dict)):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Response file must be a JSON array or an object with a responses array.")
    responses = [
        _normalize_response_item(item, index=index, path=response_path)
        for index, item in enumerate(payload, start=1)
    ]
    if len(responses) != expected:
        raise ValueError(f"Expected {expected} responses, found {len(responses)}.")
    return responses


def _normalize_response_item(item: Any, *, index: int, path: Path) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (dict, list)):
        return json.dumps(item, ensure_ascii=False)
    raise ValueError(f"{path}: response #{index} must be a string, JSON object, or JSON array.")


def run_evaluation(
    *,
    val_data: str | Path,
    output_dir: str | Path,
    base_model: str | None = None,
    adapter_dir: str | Path | None = None,
    base_responses_path: str | Path | None = None,
    sft_responses_path: str | Path | None = None,
    limit: int | None = None,
    max_new_tokens: int = 512,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    samples = load_samples(val_data, limit=limit)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if base_responses_path:
        base_responses = load_response_file(base_responses_path, len(samples))
        base_tokens_per_sec = None
    else:
        if not base_model:
            raise ValueError("--base-model is required unless --base-responses is provided.")
        base_responses, base_tokens_per_sec = generate_base_responses(
            base_model=base_model,
            samples=samples,
            max_new_tokens=max_new_tokens,
            device=device,
            dtype=dtype,
        )

    if sft_responses_path:
        sft_responses = load_response_file(sft_responses_path, len(samples))
        sft_tokens_per_sec = None
    else:
        if not base_model or not adapter_dir:
            raise ValueError("--base-model and --adapter-dir are required unless --sft-responses is provided.")
        sft_responses, sft_tokens_per_sec = generate_sft_responses(
            base_model=base_model,
            adapter_dir=adapter_dir,
            samples=samples,
            max_new_tokens=max_new_tokens,
            device=device,
            dtype=dtype,
        )

    base_metrics = evaluate_responses(base_responses, samples)
    sft_metrics = evaluate_responses(sft_responses, samples)
    base_metrics["avg_tokens_per_sec"] = base_tokens_per_sec
    sft_metrics["avg_tokens_per_sec"] = sft_tokens_per_sec
    summary = compare_metrics(base_metrics, sft_metrics)

    (output / "base_responses.json").write_text(
        json.dumps({"responses": base_responses, "avg_tokens_per_sec": base_tokens_per_sec}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "sft_responses.json").write_text(
        json.dumps({"responses": sft_responses, "avg_tokens_per_sec": sft_tokens_per_sec}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    details = build_detailed_results(samples, base_responses, sft_responses)
    (output / "detailed_results.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "experiment_report.md").write_text(render_report(summary, details), encoding="utf-8")
    return summary


def render_report(summary: dict[str, Any], details: list[dict[str, Any]]) -> str:
    lines = [
        "# SFT Protocol Evaluation Report",
        "",
        "## Scope",
        "",
        "This report evaluates protocol-following behavior on held-out Alpaca validation samples.",
        "It measures structured output alignment and does not prove complex real-world code repair ability.",
        "",
        "## Metrics",
        "",
        "| Metric | Base | SFT | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    base = summary["base"]
    sft = summary["sft"]
    delta = summary["delta"]
    for key in METRIC_KEYS:
        lines.append(f"| {key} | {_format_metric(base.get(key))} | {_format_metric(sft.get(key))} | {_format_delta(delta.get(key))} |")
    lines.extend(
        [
            "",
            "## Sample Mix",
            "",
            f"- Total validation samples: {base.get('n_samples', len(details))}",
            f"- Task counts: {json.dumps(base.get('task_counts', {}), ensure_ascii=False, sort_keys=True)}",
            "",
            "## Interpretation",
            "",
            "- Higher JSON validity and field hit rate indicate better adherence to the agent output protocol.",
            "- Tool accuracy is meaningful only for tool-call samples with reference tool names.",
            "- Runtime tool-call parse rate verifies that the current AgentCli parser can execute the response.",
            "- File mention rate checks whether referenced implementation or test files appear in the prediction.",
            "- ROUGE-L is a weak similarity signal and should be inspected with detailed results.",
            "- End-to-end repair quality still requires separate agent runs, diffs, and tests.",
            "",
            "## Artifacts",
            "",
            "- `metrics_summary.json`: aggregate base/SFT metrics and deltas.",
            "- `detailed_results.json`: per-sample reference output, base output, SFT output, task type, and per-sample scores.",
            "- `base_responses.json` and `sft_responses.json`: raw model responses used for scoring.",
        ]
    )
    return "\n".join(lines) + "\n"


def print_comparison(summary: dict[str, Any]) -> None:
    print("Metric                 Base        SFT      Delta")
    print("--------------------------------------------------")
    for key in METRIC_KEYS:
        print(
            f"{key:<20} "
            f"{_format_metric(summary['base'].get(key)):>8} "
            f"{_format_metric(summary['sft'].get(key)):>8} "
            f"{_format_delta(summary['delta'].get(key)):>8}"
        )


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    if math.isnan(numeric):
        return "N/A"
    return f"{numeric * 100:.1f}%"


def _format_delta(value: Any) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    if math.isnan(numeric):
        return "N/A"
    return f"{numeric * 100:+.1f}%"
