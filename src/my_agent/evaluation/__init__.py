from __future__ import annotations

"""Evaluation utilities for SFT protocol and later agent benchmarks."""

from my_agent.evaluation.protocol_metrics import (
    METRIC_KEYS,
    build_detailed_results,
    compare_metrics,
    compute_field_hit_rate,
    compute_file_mention_rate,
    compute_tool_accuracy,
    detect_task_type,
    evaluate_responses,
    is_valid_json,
    parse_json_object,
    rouge_l_f1,
    score_response,
)
from my_agent.evaluation.protocol_runner import load_response_file, load_samples, run_evaluation

__all__ = [
    "METRIC_KEYS",
    "build_detailed_results",
    "compare_metrics",
    "compute_field_hit_rate",
    "compute_file_mention_rate",
    "compute_tool_accuracy",
    "detect_task_type",
    "evaluate_responses",
    "is_valid_json",
    "load_response_file",
    "load_samples",
    "parse_json_object",
    "rouge_l_f1",
    "run_evaluation",
    "score_response",
]
