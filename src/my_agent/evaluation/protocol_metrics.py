from __future__ import annotations

"""Pure protocol metrics for base-vs-SFT evaluation.

This module intentionally has no model-loading or filesystem side effects, so
later end-to-end evaluation can reuse these metrics without importing heavy
training dependencies.
"""

import json
import re
from collections import Counter
from typing import Any


METRIC_KEYS = (
    "json_valid_rate",
    "field_hit_rate",
    "tool_accuracy",
    "file_mention_rate",
    "rouge_l",
)

REQUIRED_FIELDS = {
    "tool_call": ("tool", "arguments", "reason"),
    "strategy": ("strategy",),
    "repair_plan": ("plan", "validation"),
    "other": (),
}

FILE_PATTERN = re.compile(
    r"(?<![\w./-])[\w./-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|c|h|hpp|md|toml|yaml|yml|json)(?![\w/-])"
)
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]")


def is_valid_json(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(parsed, dict)


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    if not isinstance(text, str):
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def detect_task_type(sample: dict[str, Any]) -> str:
    reference = parse_json_object(str(sample.get("output", "")))
    instruction = str(sample.get("instruction", "")).lower()
    if reference:
        if "tool" in reference and "arguments" in reference:
            return "tool_call"
        if "strategy" in reference:
            return "strategy"
        if "plan" in reference and "validation" in reference:
            return "repair_plan"
    if "tool" in instruction or "工具调用" in instruction:
        return "tool_call"
    if "strategy" in instruction or "策略" in instruction:
        return "strategy"
    if "plan" in instruction or "修复计划" in instruction:
        return "repair_plan"
    return "other"


def compute_field_hit_rate(prediction: dict[str, Any] | None, task_type: str) -> float:
    required = REQUIRED_FIELDS.get(task_type, ())
    if not required:
        return 1.0
    if prediction is None:
        return 0.0
    hits = sum(1 for field in required if field in prediction)
    return hits / len(required)


def compute_tool_accuracy(prediction: dict[str, Any] | None, reference: dict[str, Any] | None) -> float | None:
    if not reference:
        return None
    expected = reference.get("tool")
    if not isinstance(expected, str) or not expected.strip():
        return None
    if prediction is None:
        return 0.0
    actual = prediction.get("tool")
    if not isinstance(actual, str):
        return 0.0
    return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0


def compute_file_mention_rate(prediction_text: str, sample: dict[str, Any], reference: dict[str, Any] | None) -> float | None:
    files = sorted(_collect_reference_files(sample, reference))
    if not files:
        return None
    hits = sum(1 for file_name in files if file_name in prediction_text)
    return hits / len(files)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens_for_rouge(prediction)
    ref_tokens = _tokens_for_rouge(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_response(response: str, sample: dict[str, Any]) -> dict[str, Any]:
    task_type = detect_task_type(sample)
    reference_text = str(sample.get("output", ""))
    prediction = parse_json_object(response)
    reference = parse_json_object(reference_text)
    tool_accuracy = compute_tool_accuracy(prediction, reference) if task_type == "tool_call" else None
    file_mention_rate = compute_file_mention_rate(response, sample, reference)
    return {
        "task_type": task_type,
        "json_valid": is_valid_json(response),
        "field_hit_rate": compute_field_hit_rate(prediction, task_type),
        "tool_accuracy": tool_accuracy,
        "file_mention_rate": file_mention_rate,
        "rouge_l": rouge_l_f1(response, reference_text),
    }


def evaluate_responses(responses: list[str], samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(responses) != len(samples):
        raise ValueError(f"Response count {len(responses)} does not match sample count {len(samples)}.")

    json_valid: list[float] = []
    field_hits: list[float] = []
    tool_accs: list[float] = []
    file_mentions: list[float] = []
    rouge_scores: list[float] = []
    task_counts: Counter[str] = Counter()

    for response, sample in zip(responses, samples):
        score = score_response(response, sample)
        task_type = str(score["task_type"])
        task_counts[task_type] += 1
        json_valid.append(1.0 if score["json_valid"] else 0.0)
        field_hits.append(float(score["field_hit_rate"]))
        rouge_scores.append(float(score["rouge_l"]))
        if score["tool_accuracy"] is not None:
            tool_accs.append(float(score["tool_accuracy"]))
        if score["file_mention_rate"] is not None:
            file_mentions.append(float(score["file_mention_rate"]))

    return {
        "n_samples": len(samples),
        "task_counts": dict(sorted(task_counts.items())),
        "json_valid_rate": _mean(json_valid),
        "field_hit_rate": _mean(field_hits),
        "tool_accuracy": _mean(tool_accs),
        "file_mention_rate": _mean(file_mentions),
        "rouge_l": _mean(rouge_scores),
        "_n_tool_accuracy": len(tool_accs),
        "_n_file_mention": len(file_mentions),
    }


def compare_metrics(base: dict[str, Any], sft: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        base_value = base.get(key)
        sft_value = sft.get(key)
        if base_value is None or sft_value is None:
            delta[key] = None
        else:
            delta[key] = float(sft_value) - float(base_value)
    return {"base": base, "sft": sft, "delta": delta}


def build_detailed_results(
    samples: list[dict[str, Any]],
    base_responses: list[str],
    sft_responses: list[str],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for idx, (sample, base_response, sft_response) in enumerate(zip(samples, base_responses, sft_responses)):
        details.append(
            {
                "index": idx,
                "task_type": detect_task_type(sample),
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "reference_output": sample.get("output", ""),
                "base_output": base_response,
                "sft_output": sft_response,
                "base_metrics": score_response(base_response, sample),
                "sft_metrics": score_response(sft_response, sample),
            }
        )
    return details


def _collect_reference_files(sample: dict[str, Any], reference: dict[str, Any] | None) -> set[str]:
    files: set[str] = set()
    _collect_files_from_value(sample.get("input"), files)
    _collect_files_from_value(reference, files)
    return files


def _collect_files_from_value(value: Any, files: set[str]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _collect_files_from_value(child, files)
    elif isinstance(value, list):
        for child in value:
            _collect_files_from_value(child, files)
    elif isinstance(value, str):
        for match in FILE_PATTERN.findall(value):
            files.add(match)


def _tokens_for_rouge(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if tokens:
        return tokens
    return list(text.lower())


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
