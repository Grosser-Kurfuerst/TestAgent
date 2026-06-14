from __future__ import annotations

"""Shared agent benchmark runner utilities."""

import json
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from my_agent.config import AgentConfig
from my_agent.runtime import run_agent
from my_agent.tracing import append_benchmark_result


TEST_COMMAND = "python -m pytest -q"

BuildRepoFn = Callable[[dict[str, Any], Path], Path]
TaskFn = Callable[[dict[str, Any]], str]
EvaluateFn = Callable[[Path], tuple[bool, str]]
LoadRowsFn = Callable[[str], list[dict[str, Any]]]
AgentRunnerFn = Callable[..., Any]


@dataclass
class EvalResult:
    task_id: str
    status: str
    scored: bool = True
    test_output: str = ""
    agent_steps: int = 0
    agent_done: bool = False
    agent_stop_reason: str = ""
    error: str = ""
    elapsed_sec: float = 0.0
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "scored": self.scored,
            "test_output": self.test_output[:2000],
            "agent_steps": self.agent_steps,
            "agent_done": self.agent_done,
            "agent_stop_reason": self.agent_stop_reason,
            "error": self.error,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    test_command: str
    build_repo: BuildRepoFn
    task_id: TaskFn
    task_prompt: TaskFn
    evaluate_solution: EvaluateFn


@dataclass(frozen=True)
class BenchmarkRunResult:
    results: list[EvalResult]
    summary: dict[str, float | int]
    output_dir: Path
    results_path: Path


def run_one_task(
    row: dict[str, Any],
    spec: BenchmarkSpec,
    base_dir: Path,
    config: AgentConfig,
    max_steps: int,
    llm_retries: int = 2,
    retry_delay_sec: float = 2.0,
    count_transient_errors: bool = False,
    *,
    agent_runner: AgentRunnerFn = run_agent,
) -> EvalResult:
    task_id = spec.task_id(row)
    t0 = time.monotonic()
    max_attempts = max(1, llm_retries + 1)
    last_transient_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            repo_path = spec.build_repo(row, base_dir)
            state = agent_runner(
                repo_path=repo_path,
                task=spec.task_prompt(row),
                test_command=spec.test_command,
                config=config,
                max_steps=max_steps,
            )
            test_passed, test_output = spec.evaluate_solution(repo_path)
            status = "passed" if test_passed else "failed"
            record_benchmark_result(
                state,
                benchmark=spec.name,
                task_id=task_id,
                status=status,
                scored=True,
                test_command=spec.test_command,
                test_output=test_output,
            )
            return EvalResult(
                task_id=task_id,
                status=status,
                scored=True,
                test_output=test_output,
                agent_steps=state.steps,
                agent_done=state.done,
                agent_stop_reason=state.stop_reason,
                elapsed_sec=time.monotonic() - t0,
                attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001 - evaluation boundary records all failures
            message = f"{type(exc).__name__}: {exc}"
            if is_transient_llm_error(exc):
                last_transient_error = message
                if attempt < max_attempts:
                    if retry_delay_sec > 0:
                        time.sleep(retry_delay_sec)
                    continue
                return EvalResult(
                    task_id=task_id,
                    status="transient_error",
                    scored=count_transient_errors,
                    error=last_transient_error,
                    elapsed_sec=time.monotonic() - t0,
                    attempts=attempt,
                )
            return EvalResult(
                task_id=task_id,
                status="error",
                scored=True,
                error=message,
                elapsed_sec=time.monotonic() - t0,
                attempts=attempt,
            )

    return EvalResult(
        task_id=task_id,
        status="transient_error",
        scored=count_transient_errors,
        error=last_transient_error or "Transient LLM error",
        elapsed_sec=time.monotonic() - t0,
        attempts=max_attempts,
    )


def run_pytest_or_fallback(repo_path: Path, fallback: EvaluateFn) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        if "No module named pytest" in output:
            return fallback(repo_path)
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "pytest timed out after 120s"
    except FileNotFoundError:
        return fallback(repo_path)


def run_import_test_fallback(repo_path: Path, test_function_name: str) -> tuple[bool, str]:
    previous_solution = sys.modules.pop("solution", None)
    previous_test_solution = sys.modules.pop("test_solution", None)
    old_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            "test_solution", str(repo_path / "tests" / "test_solution.py")
        )
        if spec is None or spec.loader is None:
            return False, "Cannot load test module"
        sys.path.insert(0, str(repo_path))
        test_module = importlib.util.module_from_spec(spec)
        sys.modules["test_solution"] = test_module
        spec.loader.exec_module(test_module)
        getattr(test_module, test_function_name)()
        return True, "All tests passed (fallback runner)."
    except AssertionError as exc:
        return False, f"Assertion failed: {exc}"
    except Exception as exc:  # noqa: BLE001 - evaluation boundary records failure text
        return False, f"Error: {type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = old_path
        sys.modules.pop("solution", None)
        sys.modules.pop("test_solution", None)
        if previous_solution is not None:
            sys.modules["solution"] = previous_solution
        if previous_test_solution is not None:
            sys.modules["test_solution"] = previous_test_solution


def is_transient_llm_error(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}"
    transient_markers = (
        "LLM response message content was empty.",
        "HTTP 429",
        "HTTP 500",
        "HTTP 502",
        "HTTP 503",
        "HTTP 504",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "temporary failure",
        "connection reset",
        "connection aborted",
        "remote end closed connection",
    )
    lowered = message.lower()
    return any(marker.lower() in lowered for marker in transient_markers)


def record_benchmark_result(
    state: Any,
    *,
    benchmark: str,
    task_id: str,
    status: str,
    scored: bool,
    test_command: str,
    test_output: str,
) -> None:
    trace_path = getattr(state, "trace_path", None)
    run_id = getattr(state, "run_id", "")
    if trace_path is None or not run_id:
        return
    append_benchmark_result(
        trace_path,
        run_id=str(run_id),
        benchmark=benchmark,
        task_id=task_id,
        status=status,
        scored=scored,
        test_command=test_command,
        test_output=test_output,
    )


def summarize_results(results: list[EvalResult]) -> dict[str, float | int]:
    total = len(results)
    scored_total = sum(1 for r in results if r.scored)
    passed_count = sum(1 for r in results if r.status == "passed")
    failed_count = sum(1 for r in results if r.status == "failed")
    error_count = sum(1 for r in results if r.status == "error")
    transient_excluded_count = sum(1 for r in results if r.status == "transient_error" and not r.scored)
    transient_counted_count = sum(1 for r in results if r.status == "transient_error" and r.scored)
    return {
        "total": total,
        "scored": scored_total,
        "passed": passed_count,
        "failed": failed_count,
        "error": error_count,
        "transient_excluded": transient_excluded_count,
        "transient_counted": transient_counted_count,
        "solve_rate": passed_count / scored_total * 100 if scored_total > 0 else 0.0,
        "end_to_end_rate": passed_count / total * 100 if total > 0 else 0.0,
    }


def load_results_file(path: str | Path) -> list[EvalResult]:
    results_path = Path(path)
    results: list[EvalResult] = []
    if not results_path.exists():
        return results
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{results_path}: line {line_number} is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{results_path}: line {line_number} must be a JSON object.")
        results.append(_eval_result_from_record(payload, path=results_path, line_number=line_number))
    return results


def _eval_result_from_record(payload: dict[str, Any], *, path: Path, line_number: int) -> EvalResult:
    status = payload.get("status")
    if status is None and isinstance(payload.get("passed"), bool):
        status = "passed" if payload["passed"] else "failed"
    if not isinstance(status, str) or not status:
        raise ValueError(f"{path}: line {line_number} missing result status.")
    return EvalResult(
        task_id=str(payload.get("task_id", "")),
        status=status,
        scored=bool(payload.get("scored", True)),
        test_output=str(payload.get("test_output", "")),
        agent_steps=_int_or_default(payload.get("agent_steps"), 0),
        agent_done=bool(payload.get("agent_done", False)),
        agent_stop_reason=str(payload.get("agent_stop_reason", "")),
        error=str(payload.get("error", "")),
        elapsed_sec=_float_or_default(payload.get("elapsed_sec"), 0.0),
        attempts=_int_or_default(payload.get("attempts"), 1),
    )


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def status_label(result: EvalResult) -> str:
    if result.status == "passed":
        return "PASS"
    if result.status == "failed":
        return "FAIL"
    if result.status == "transient_error":
        return f"TRANSIENT ({result.error[:60]})" if result.error else "TRANSIENT"
    if result.status == "error":
        return f"ERROR ({result.error[:60]})" if result.error else "ERROR"
    return result.status.upper()


def prepare_results_file(results_path: Path, start: int) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if start == 0:
        results_path.write_text("", encoding="utf-8")


def run_benchmark(args: Any, spec: BenchmarkSpec, load_rows: LoadRowsFn) -> None:
    config = AgentConfig.from_env()
    run_benchmark_with_config(
        config=config,
        output_dir=args.output_dir,
        spec=spec,
        load_rows=load_rows,
        split=args.split,
        start=args.start,
        limit=args.limit,
        max_steps=args.max_steps,
        llm_retries=args.llm_retries,
        retry_delay_sec=args.retry_delay_sec,
        count_transient_errors=args.count_transient_errors,
    )


def run_benchmark_with_config(
    *,
    config: AgentConfig,
    output_dir: str | Path,
    spec: BenchmarkSpec,
    load_rows: LoadRowsFn,
    split: str,
    start: int,
    limit: int,
    max_steps: int,
    llm_retries: int = 2,
    retry_delay_sec: float = 2.0,
    count_transient_errors: bool = False,
    write_summary: bool = False,
    summary_scope: str = "current",
    agent_runner: AgentRunnerFn = run_agent,
) -> BenchmarkRunResult:
    if summary_scope not in {"current", "results_file"}:
        raise ValueError("summary_scope must be one of: current, results_file.")
    output_dir = Path(output_dir)
    repos_dir = output_dir / "repos"
    results_path = output_dir / "results.jsonl"
    repos_dir.mkdir(parents=True, exist_ok=True)
    prepare_results_file(results_path, start)

    print(f"Loading {spec.display_name} ({split} split)...")
    dataset = load_rows(split)
    print(f"Loaded {len(dataset)} tasks, will evaluate {limit} (starting at #{start})\n")

    results: list[EvalResult] = []
    end = min(start + limit, len(dataset))

    for idx in range(start, end):
        row = dataset[idx]
        task_id = spec.task_id(row) or str(idx)
        print(f"[{idx + 1}/{len(dataset)}] task_id={task_id} ... ", end="", flush=True)

        result = run_one_task(
            row,
            spec,
            repos_dir,
            config,
            max_steps=max_steps,
            llm_retries=max(0, llm_retries),
            retry_delay_sec=max(0.0, retry_delay_sec),
            count_transient_errors=count_transient_errors,
            agent_runner=agent_runner,
        )

        status = status_label(result)
        print(
            f"{status} | attempts={result.attempts} | steps={result.agent_steps} | "
            f"done={result.agent_done} | {result.elapsed_sec:.0f}s"
        )

        results.append(result)
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    summary_results = load_results_file(results_path) if summary_scope == "results_file" else results
    summary = summarize_results(summary_results)
    print()
    print("=" * 60)
    title_suffix = " (cumulative)" if summary_scope == "results_file" else ""
    print(f"{spec.display_name} Evaluation Results{title_suffix}")
    print(f"  Total:              {summary['total']}")
    print(f"  Scored:             {summary['scored']}")
    print(f"  Passed:             {summary['passed']}")
    print(f"  Failed:             {summary['failed']}")
    print(f"  Error:              {summary['error']}")
    print(f"  Transient excluded: {summary['transient_excluded']}")
    print(f"  Transient counted:  {summary['transient_counted']}")
    print(f"  Solve rate:         {summary['solve_rate']:.1f}%")
    print(f"  End-to-end rate:    {summary['end_to_end_rate']:.1f}%")
    print(f"  Results: {results_path}")
    print("=" * 60)
    if write_summary:
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return BenchmarkRunResult(results=results, summary=summary, output_dir=output_dir, results_path=results_path)
