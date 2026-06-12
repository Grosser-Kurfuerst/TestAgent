#!/usr/bin/env python3
"""HumanEval evaluation script: run the Agent on generated task repos and score tests.

Usage:
    # Quick test
    uv run python scripts/eval_humaneval.py --limit 3

    # Larger run with an explicit output directory
    uv run python scripts/eval_humaneval.py --limit 164 --output-dir /tmp/humaneval_eval

Workflow:
    1. Load HumanEval rows.
    2. For each task, create a temporary repo with solution.py and tests/test_solution.py.
    3. Run the coding Agent against the repo.
    4. Run pytest or a fallback import-based test runner.
    5. Write results.jsonl and print score summaries.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from my_agent.config import AgentConfig
from my_agent.data.sources import load_humaneval_rows
from my_agent.data.task_repos import safe_id, write_repo
from my_agent.evaluation.agent_benchmark import (
    TEST_COMMAND,
    BenchmarkSpec,
    EvalResult,
    is_transient_llm_error as _is_transient_llm_error,
    prepare_results_file as _prepare_results_file,
    run_benchmark,
    run_import_test_fallback,
    run_one_task as _run_one_task,
    run_pytest_or_fallback,
    status_label as _status_label,
    summarize_results,
)
from my_agent.runtime import run_agent


def load_humaneval(split: str = "test") -> list[dict[str, Any]]:
    """Load HumanEval rows via the shared data loader."""
    return load_humaneval_rows(split=split)


def build_humaneval_repo(row: dict[str, Any], base_dir: Path) -> Path:
    """Create a small HumanEval repo for one task and return its path."""
    task_id = str(row.get("task_id", "unknown"))
    prompt = str(row.get("prompt", ""))
    test = str(row.get("test", ""))
    entry_point = str(row.get("entry_point", ""))
    if not prompt or not test or not entry_point:
        raise ValueError(f"HumanEval row {task_id!r} missing prompt, test, or entry_point")

    repo_path = base_dir / f"humaneval_{safe_id(task_id)}"
    if repo_path.exists():
        shutil.rmtree(repo_path)

    skeleton = prompt.rstrip() + "\n    pass\n"
    test_content = (
        f"from solution import {entry_point}\n\n"
        f"{test}\n\n"
        f"def test_humaneval() -> None:\n"
        f"    check({entry_point})\n"
    )
    write_repo(repo_path, {"solution.py": skeleton, "tests/test_solution.py": test_content})
    return repo_path


def evaluate_solution(repo_path: Path) -> tuple[bool, str]:
    """Run HumanEval tests in *repo_path* and return (passed, output)."""
    return run_pytest_or_fallback(repo_path, evaluate_solution_fallback)


def evaluate_solution_fallback(repo_path: Path) -> tuple[bool, str]:
    """Fallback when pytest is unavailable: import the generated test module."""
    return run_import_test_fallback(repo_path, "test_humaneval")


def run_one_task(
    row: dict[str, Any],
    base_dir: Path,
    config: AgentConfig,
    max_steps: int,
    llm_retries: int = 2,
    retry_delay_sec: float = 2.0,
    count_transient_errors: bool = False,
) -> EvalResult:
    return _run_one_task(
        row,
        _benchmark_spec(),
        base_dir,
        config,
        max_steps=max_steps,
        llm_retries=llm_retries,
        retry_delay_sec=retry_delay_sec,
        count_transient_errors=count_transient_errors,
        agent_runner=run_agent,
    )


def _benchmark_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="humaneval",
        display_name="HumanEval",
        test_command=TEST_COMMAND,
        build_repo=build_humaneval_repo,
        task_id=_task_id,
        task_prompt=_task_prompt,
        evaluate_solution=evaluate_solution,
    )


def _task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id", "unknown"))


def _task_prompt(row: dict[str, Any]) -> str:
    entry_point = str(row.get("entry_point", "function")).strip() or "function"
    return f"Implement the {entry_point} function in solution.py so that the HumanEval test passes."


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanEval Agent 评估")
    parser.add_argument("--limit", type=int, default=10, help="评测任务数量（默认 10）")
    parser.add_argument("--output-dir", default="/tmp/humaneval_eval", help="临时仓库和结果目录")
    parser.add_argument("--max-steps", type=int, default=10, help="每个任务的 Agent 最大步数")
    parser.add_argument("--split", default="test", help="HumanEval 数据集 split")
    parser.add_argument("--start", type=int, default=0, help="从第几个任务开始（断点续跑）")
    parser.add_argument("--llm-retries", type=int, default=2, help="LLM transient 错误重试次数（默认 2）")
    parser.add_argument("--retry-delay-sec", type=float, default=2.0, help="LLM transient 错误重试间隔秒数")
    parser.add_argument(
        "--count-transient-errors",
        action="store_true",
        help="把 API 抖动类 transient_error 计入解题评分分母。",
    )
    args = parser.parse_args()

    run_benchmark(args, _benchmark_spec(), load_humaneval)


if __name__ == "__main__":
    main()
