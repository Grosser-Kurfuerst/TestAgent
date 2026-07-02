from __future__ import annotations

"""Evaluation utilities for SFT protocol and later agent benchmarks."""

from my_agent.evaluation.agent_benchmark import (
    TEST_COMMAND,
    BenchmarkSpec,
    EvalResult,
    is_transient_llm_error,
    prepare_results_file,
    run_benchmark,
    run_import_test_fallback,
    run_one_task,
    run_pytest_or_fallback,
    status_label,
    summarize_results,
)
from my_agent.evaluation.manifest_benchmark import (
    ManifestBenchmarkResult,
    ManifestEvalResult,
    load_manifest_tasks,
    run_manifest_benchmark,
    run_test_command,
    summarize_manifest_results,
)
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
from my_agent.observability.trace_metrics import TraceMetrics, collect_trace_metrics, format_trace_metrics

__all__ = [
    "TEST_COMMAND",
    "BenchmarkSpec",
    "EvalResult",
    "METRIC_KEYS",
    "ManifestBenchmarkResult",
    "ManifestEvalResult",
    "build_detailed_results",
    "compare_metrics",
    "compute_field_hit_rate",
    "compute_file_mention_rate",
    "compute_tool_accuracy",
    "detect_task_type",
    "evaluate_responses",
    "is_valid_json",
    "is_transient_llm_error",
    "load_response_file",
    "load_manifest_tasks",
    "load_samples",
    "parse_json_object",
    "prepare_results_file",
    "rouge_l_f1",
    "run_benchmark",
    "run_import_test_fallback",
    "run_evaluation",
    "run_manifest_benchmark",
    "run_one_task",
    "run_pytest_or_fallback",
    "run_test_command",
    "score_response",
    "status_label",
    "summarize_results",
    "summarize_manifest_results",
    "TraceMetrics",
    "collect_trace_metrics",
    "format_trace_metrics",
]
