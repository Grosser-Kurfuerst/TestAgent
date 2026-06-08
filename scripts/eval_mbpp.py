#!/usr/bin/env python3
"""MBPP 评估脚本：用真实 MBPP 数据集运行 Agent 并统计测试通过率。

用法：
    # 跑 10 个任务（快速验证）
    uv run python scripts/eval_mbpp.py --limit 10

    # 跑全部 500 个任务
    uv run python scripts/eval_mbpp.py --limit 500 --output-dir /tmp/mbpp_eval

工作流程：
    1. 从 HuggingFace 加载 MBPP test split
    2. 对每个任务：创建临时仓库 → 运行 Agent → 跑 pytest → 记录 pass/fail
    3. 输出汇总分数和逐任务明细
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- 确保项目根目录在 sys.path 中 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from my_agent.config import AgentConfig
from my_agent.data.task_repos import (
    python_skeleton_from_solution,
    safe_id,
    write_python_task_repo,
)
from my_agent.runtime import run_agent


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    test_output: str = ""
    agent_steps: int = 0
    agent_done: bool = False
    agent_stop_reason: str = ""
    error: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "test_output": self.test_output[:2000],
            "agent_steps": self.agent_steps,
            "agent_done": self.agent_done,
            "agent_stop_reason": self.agent_stop_reason,
            "error": self.error,
            "elapsed_sec": round(self.elapsed_sec, 1),
        }


def load_mbpp(split: str = "test") -> list[dict[str, Any]]:
    """加载真实 MBPP 数据集（需要 datasets 包）。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "需要 datasets 包。安装: uv sync --extra data"
        ) from exc
    return list(load_dataset("google-research-datasets/mbpp", split=split))


def build_mbpp_repo(row: dict[str, Any], base_dir: Path) -> Path:
    """为一个 MBPP 任务创建临时仓库，返回仓库路径。"""
    task_id = str(row.get("task_id", "unknown"))
    code = str(row.get("code", "")).strip()
    tests = row.get("test_list") or []

    repo_path = base_dir / f"mbpp_{safe_id(task_id)}"
    skeleton = python_skeleton_from_solution(code)
    write_python_task_repo(repo_path, skeleton, tests)
    return repo_path


def evaluate_solution(repo_path: Path) -> tuple[bool, str]:
    """在仓库中运行 pytest，返回 (passed, output)。

    优先使用 pytest；pytest 不可用时自动回退到 Python import 方式。
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        # pytest 模块缺失时自动回退
        if "No module named pytest" in output:
            return evaluate_solution_fallback(repo_path)
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "pytest timed out after 120s"
    except FileNotFoundError:
        return evaluate_solution_fallback(repo_path)


def evaluate_solution_fallback(repo_path: Path) -> tuple[bool, str]:
    """pytest 不可用时的回退方案：直接 import 并运行测试函数。"""
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(
            "test_solution", str(repo_path / "tests" / "test_solution.py")
        )
        if spec is None or spec.loader is None:
            return False, "Cannot load test module"
        old_path = list(sys.path)
        sys.path.insert(0, str(repo_path))
        test_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(test_module)
            test_module.test_mbpp_generated()
        finally:
            sys.path[:] = old_path
        return True, "All tests passed (fallback runner)."
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"


def run_one_task(
    row: dict[str, Any],
    base_dir: Path,
    config: AgentConfig,
    max_steps: int,
) -> EvalResult:
    """对单个 MBPP 任务：建仓库 → 跑 Agent → 评测。"""
    task_id = str(row.get("task_id", "unknown"))
    text = str(row.get("text", "")).strip()
    t0 = time.monotonic()

    try:
        # 1. 建仓库
        repo_path = build_mbpp_repo(row, base_dir)

        # 2. 运行 Agent
        state = run_agent(
            repo_path=repo_path,
            task=f"Implement the solution.py skeleton so that all tests pass. Task description: {text}",
            test_command="python -m pytest -q",
            config=config,
            max_steps=max_steps,
        )

        # 3. 评测：运行 pytest
        passed, test_output = evaluate_solution(repo_path)

        return EvalResult(
            task_id=task_id,
            passed=passed,
            test_output=test_output,
            agent_steps=state.steps,
            agent_done=state.done,
            agent_stop_reason=state.stop_reason,
            elapsed_sec=time.monotonic() - t0,
        )
    except Exception as e:
        return EvalResult(
            task_id=task_id,
            passed=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_sec=time.monotonic() - t0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MBPP Agent 评估")
    parser.add_argument("--limit", type=int, default=10, help="评测任务数量（默认 10）")
    parser.add_argument("--output-dir", default="/tmp/mbpp_eval", help="临时仓库和结果目录")
    parser.add_argument("--max-steps", type=int, default=10, help="每个任务的 Agent 最大步数")
    parser.add_argument("--split", default="test", help="MBPP 数据集 split")
    parser.add_argument("--start", type=int, default=0, help="从第几个任务开始（断点续跑）")
    args = parser.parse_args()

    # 准备
    config = AgentConfig.from_env()
    output_dir = Path(args.output_dir)
    repos_dir = output_dir / "repos"
    results_path = output_dir / "results.jsonl"
    repos_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print(f"Loading MBPP ({args.split} split)...")
    dataset = load_mbpp(split=args.split)
    print(f"Loaded {len(dataset)} tasks, will evaluate {args.limit} (starting at #{args.start})\n")

    # 逐任务评测
    results: list[EvalResult] = []
    end = min(args.start + args.limit, len(dataset))

    for idx in range(args.start, end):
        row = dataset[idx]
        task_id = str(row.get("task_id", idx))
        print(f"[{idx + 1}/{len(dataset)}] task_id={task_id} ... ", end="", flush=True)

        result = run_one_task(row, repos_dir, config, max_steps=args.max_steps)

        status = "PASS" if result.passed else "FAIL"
        if result.error:
            status = f"ERROR ({result.error[:60]})"
        print(
            f"{status} | steps={result.agent_steps} | "
            f"done={result.agent_done} | {result.elapsed_sec:.0f}s"
        )

        results.append(result)

        # 增量写入结果文件
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    # 汇总
    passed = sum(1 for r in results if r.passed)
    errored = sum(1 for r in results if r.error)
    total = len(results)
    rate = passed / total * 100 if total > 0 else 0

    print()
    print("=" * 60)
    print(f"MBPP Evaluation Results")
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {total - passed - errored}")
    print(f"  Error:   {errored}")
    print(f"  Rate:    {rate:.1f}%")
    print(f"  Results: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
