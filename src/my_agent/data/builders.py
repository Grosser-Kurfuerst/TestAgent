from __future__ import annotations

"""Phase 6 SFT data builders and conversion orchestration."""

import json
import re
from pathlib import Path
from typing import Any, Iterable

from my_agent.data.jsonl import read_jsonl
from my_agent.data.reports import BuildReport
from my_agent.data.sft_samples import (
    make_repair_plan_sample,
    make_strategy_sample,
    make_tool_call_sample,
    make_write_file_sample,
)
from my_agent.data.sources import load_dataset_rows, load_humaneval_rows, load_mbpp_rows
from my_agent.data.task_repos import (
    python_skeleton_from_solution,
    safe_id,
    write_python_task_repo,
    write_repo,
)


def build_mbpp(
    output_dir: str | Path,
    limit: int = 50,
    split: str = "test",
) -> BuildReport:
    """Download (or load cached) MBPP and convert to task repos + SFT samples."""

    dataset = load_mbpp_rows(split=split)
    output = Path(output_dir)
    repo_dir = output / "repos" / "mbpp"
    tasks_path = output / "tasks" / "mbpp_tasks.jsonl"
    sft_path = output / "sft" / "mbpp_sft.jsonl"
    _ensure_parents(tasks_path, sft_path, repo_dir)

    total = 0
    written = 0
    skipped = 0
    errors: list[str] = []
    with tasks_path.open("w", encoding="utf-8") as tasks_file, sft_path.open("w", encoding="utf-8") as sft_file:
        for row in _take(dataset, limit):
            total += 1
            task_id = str(row.get("task_id", written))
            text = str(row.get("text", "")).strip()
            code = str(row.get("code", "")).strip()
            tests = row.get("test_list") or []
            if not code or not tests:
                skipped += 1
                errors.append(f"MBPP:{task_id}: missing code or tests")
                continue

            repo_path = repo_dir / f"mbpp_{safe_id(task_id)}"
            skeleton = python_skeleton_from_solution(code)
            write_python_task_repo(repo_path, skeleton, tests)

            record_id = f"mbpp_{safe_id(task_id)}"
            task_record = {
                "id": record_id,
                "source": "MBPP",
                "repo": f"{output.name}/{repo_path.relative_to(output).as_posix()}",
                "task": f"Implement the solution.py skeleton so that all tests pass. Task description: {text}",
                "test_command": "pytest -q",
                "success_hint": "All generated pytest tests pass.",
            }
            sft_record = make_write_file_sample(
                task=task_record["task"],
                repo_context="solution.py contains a function stub. tests/test_solution.py contains MBPP tests.",
                path="solution.py",
                content=code + "\n",
                reason="Write the complete solution to pass the provided tests.",
                metadata={"source": "MBPP", "id": record_id},
            )
            tasks_file.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            sft_file.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            written += 1

    return BuildReport(
        source="MBPP",
        total=total,
        written=written,
        skipped=skipped,
        errors=errors,
        tasks_path=tasks_path,
        sft_path=sft_path,
        repo_dir=repo_dir,
    )


def build_humaneval(
    output_dir: str | Path,
    limit: int = 50,
    split: str = "test",
) -> BuildReport:
    """Download (or load cached) HumanEval and convert to task repos + SFT samples."""

    dataset = load_humaneval_rows(split=split)
    output = Path(output_dir)
    repo_dir = output / "repos" / "humaneval"
    tasks_path = output / "tasks" / "humaneval_tasks.jsonl"
    sft_path = output / "sft" / "humaneval_sft.jsonl"
    _ensure_parents(tasks_path, sft_path, repo_dir)

    total = 0
    written = 0
    skipped = 0
    errors: list[str] = []
    with tasks_path.open("w", encoding="utf-8") as tasks_file, sft_path.open("w", encoding="utf-8") as sft_file:
        for row in _take(dataset, limit):
            total += 1
            task_id = safe_id(str(row.get("task_id", written)))
            prompt = str(row.get("prompt", ""))
            canonical_solution = str(row.get("canonical_solution", ""))
            test = str(row.get("test", ""))
            entry_point = str(row.get("entry_point", ""))
            if not prompt or not canonical_solution or not test or not entry_point:
                skipped += 1
                errors.append(f"HumanEval:{task_id}: missing prompt, solution, test, or entry point")
                continue

            repo_path = repo_dir / task_id
            skeleton = prompt.rstrip() + "\n    pass\n"
            solution = prompt.rstrip() + "\n" + canonical_solution.rstrip() + "\n"
            test_content = (
                f"from solution import {entry_point}\n\n"
                f"{test}\n\n"
                f"def test_humaneval() -> None:\n"
                f"    check({entry_point})\n"
            )
            write_repo(repo_path, {"solution.py": skeleton, "tests/test_solution.py": test_content})

            task_record = {
                "id": task_id,
                "source": "HumanEval",
                "repo": f"{output.name}/{repo_path.relative_to(output).as_posix()}",
                "task": f"Implement the {entry_point} function in solution.py so that the HumanEval test passes.",
                "test_command": "pytest -q",
                "success_hint": "HumanEval check passes.",
            }
            sft_record = make_write_file_sample(
                task=task_record["task"],
                repo_context=(
                    "solution.py contains a code prompt and a partial implementation. "
                    "tests/test_solution.py contains the HumanEval test harness."
                ),
                path="solution.py",
                content=solution,
                reason="Complete the function stub to satisfy the HumanEval test.",
                metadata={"source": "HumanEval", "id": task_id},
            )
            tasks_file.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            sft_file.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            written += 1

    return BuildReport(
        source="HumanEval",
        total=total,
        written=written,
        skipped=skipped,
        errors=errors,
        tasks_path=tasks_path,
        sft_path=sft_path,
        repo_dir=repo_dir,
    )


def build_swebench_lite(
    output_dir: str | Path,
    limit: int = 50,
    split: str = "test",
) -> BuildReport:
    """Load SWE-bench Lite instances and write task manifests."""

    dataset = load_dataset_rows("princeton-nlp/SWE-bench_Lite", split=split)
    output = Path(output_dir)
    tasks_path = output / "tasks" / "swebench_lite_tasks.jsonl"
    _ensure_parents(tasks_path)

    written = 0
    with tasks_path.open("w", encoding="utf-8") as tasks_file:
        for row in _take(dataset, limit):
            instance_id = str(row.get("instance_id", written))
            task_record = {
                "id": instance_id,
                "source": "SWE-bench_Lite",
                "repo_name": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "task": row.get("problem_statement"),
                "test_command": "SWE-bench evaluation harness required.",
                "patch": row.get("patch"),
                "test_patch": row.get("test_patch"),
                "note": (
                    "This manifest does not create local repos. "
                    "Use SWE-bench harness to materialize each instance."
                ),
            }
            tasks_file.write(json.dumps(task_record, ensure_ascii=False) + "\n")
            written += 1

    return BuildReport(source="SWE-bench_Lite", total=written, written=written, tasks_path=tasks_path)


def swebench_to_sft(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "plan",
    *,
    strict: bool = True,
) -> BuildReport:
    """Convert SWE-bench Lite manifest records into repair-plan SFT samples."""

    if mode != "plan":
        raise ValueError("SWE-bench SFT conversion only supports plan mode in Phase 6.")

    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    read_result = read_jsonl(source, strict=strict, allow_single=True)
    records = read_result.records
    errors = list(read_result.errors)

    written = 0
    skipped = read_result.skipped
    with output.open("w", encoding="utf-8") as out:
        for _line_num, row in records:
            if row.get("source") != "SWE-bench_Lite":
                skipped += 1
                continue
            patch = row.get("patch")
            patch_text = patch if isinstance(patch, str) else ""
            sample = make_repair_plan_sample(
                repo_name=row.get("repo_name"),
                base_commit=row.get("base_commit"),
                problem_statement=row.get("task"),
                plan=_patch_to_plan_hint(patch_text),
                validation="Use SWE-bench harness and the provided test_patch to validate the fix.",
                metadata={"id": row.get("id"), "source": "SWE-bench_Lite"},
            )
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    return BuildReport(
        source="SWE-bench_Lite SFT",
        total=len(records) + read_result.skipped,
        written=written,
        skipped=skipped,
        errors=errors,
        sft_path=output,
    )


def local_tasks_to_sft(
    input_path: str | Path,
    output_path: str | Path,
    *,
    strict: bool = True,
) -> BuildReport:
    """Convert local runnable task manifests into high-level strategy samples."""

    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    read_result = read_jsonl(source, strict=strict, allow_single=True)
    records = read_result.records
    errors = list(read_result.errors)

    written = 0
    skipped = read_result.skipped
    with output.open("w", encoding="utf-8") as out:
        for line_num, row in records:
            repo = row.get("repo")
            task = row.get("task")
            if not _non_empty_str(repo) or not _non_empty_str(task):
                message = f"{source}:{line_num}: requires non-empty repo and task"
                if strict:
                    raise ValueError(message)
                errors.append(message)
                skipped += 1
                continue
            repo_text = str(repo).strip()
            task_text = str(task).strip()

            test_command = row.get("test_command")
            if not _non_empty_str(test_command):
                test_command = "pytest -q"
            else:
                test_command = str(test_command).strip()
            metadata_source = row.get("source")
            if not _non_empty_str(metadata_source):
                metadata_source = "local"
            else:
                metadata_source = str(metadata_source).strip()
            sample = make_strategy_sample(
                repo=repo_text,
                task=task_text,
                test_command=test_command,
                metadata={"id": row.get("id"), "source": metadata_source},
            )
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    return BuildReport(
        source="local task SFT",
        total=len(records) + read_result.skipped,
        written=written,
        skipped=skipped,
        errors=errors,
        sft_path=output,
    )


def traces_to_sft(
    trace_path: str | Path,
    output_path: str | Path,
    *,
    strict: bool = False,
) -> BuildReport:
    """Walk agent trace JSONL files and convert successful tool calls into SFT records."""

    traces = _trace_files(Path(trace_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    errors: list[str] = []
    with output.open("w", encoding="utf-8") as out:
        for trace_file in traces:
            read_result = read_jsonl(trace_file, strict=strict, allow_single=True)
            records = read_result.records
            errors.extend(read_result.errors)
            total += len(records) + read_result.skipped
            skipped += read_result.skipped
            if not _trace_is_successful(records):
                continue
            task = ""
            plan = ""
            history: list[dict[str, Any]] = []
            pending_calls: dict[str, dict[str, Any]] = {}
            for _line_num, record in records:
                event = record.get("event")
                payload = record.get("payload", {})
                if not isinstance(payload, dict):
                    continue

                if event == "repo.indexed":
                    payload_task = payload.get("task")
                    if isinstance(payload_task, str):
                        task = payload_task
                elif event == "tool.started":
                    call_id = payload.get("id")
                    tool = payload.get("name")
                    if not isinstance(call_id, str) or not isinstance(tool, str):
                        continue
                    pending_calls[call_id] = {
                        "tool": tool,
                        "arguments": _decode_tool_arguments(payload.get("arguments")),
                        "reason": "Native LLM tool_call.",
                    }
                elif event == "tool.completed":
                    result_id = payload.get("id")
                    if not isinstance(result_id, str):
                        continue
                    call = pending_calls.get(result_id)
                    if not isinstance(call, dict):
                        tool_name = payload.get("name")
                        if not isinstance(tool_name, str):
                            continue
                        call = {
                            "tool": tool_name,
                            "arguments": _decode_tool_arguments(payload.get("arguments")),
                            "reason": "Native LLM tool_call.",
                        }
                    tool = call.get("tool")
                    if not isinstance(tool, str):
                        continue
                    result = {
                        "ok": bool(payload.get("ok")),
                        "output": payload.get("content", ""),
                        "blocked": bool(payload.get("blocked")),
                        "reason": payload.get("error_code", ""),
                    }
                    if tool == "finish":
                        history.append({"call": call, "result": result})
                        continue
                    if bool(payload.get("ok")):
                        arguments = call.get("arguments", {})
                        if not isinstance(arguments, dict):
                            arguments = {}
                        reason = call.get("reason", "")
                        if not isinstance(reason, str):
                            reason = ""
                        sample = make_tool_call_sample(
                            task=task or "(task text not recorded in trace)",
                            plan=plan,
                            history=history[-6:],
                            tool=tool,
                            arguments=arguments,
                            reason=reason,
                            metadata={
                                "source": "Agent trace",
                                "trace_file": str(trace_file),
                                "tool_call_id": result_id,
                            },
                        )
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        written += 1
                    history.append({"call": call, "result": result})

    return BuildReport(
        source="Agent trace",
        total=total,
        written=written,
        skipped=skipped,
        errors=errors,
        sft_path=output,
        extra={"trace_files": len(traces)},
    )


def _trace_is_successful(records: list[tuple[int, dict[str, Any]]]) -> bool:
    for _line_num, record in records:
        if record.get("event") != "benchmark_result":
            continue
        payload = record.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "passed":
            return True
    return False


def _decode_tool_arguments(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _take(items: Iterable[dict[str, Any]], limit: int) -> Iterable[dict[str, Any]]:
    for idx, item in enumerate(items):
        if idx >= limit:
            return
        yield item


def _ensure_parents(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True) if path.suffix else path.mkdir(parents=True, exist_ok=True)


def _patch_to_plan_hint(patch: str) -> str:
    files = re.findall(r"^diff --git a/(.*?) b/", patch, flags=re.MULTILINE)
    if not files:
        return "Inspect the issue, locate relevant files, make a minimal fix, then run the provided tests."
    unique: list[str] = []
    for file_path in files:
        if file_path not in unique:
            unique.append(file_path)
    return (
        "Inspect and modify these likely relevant files: "
        + ", ".join(unique[:8])
        + ". Make the smallest behavior-preserving fix and validate with the provided tests."
    )


def _trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob("*.jsonl") if p.is_file())
    raise FileNotFoundError(f"Trace path not found: {path}")


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "BuildReport",
    "build_humaneval",
    "build_mbpp",
    "build_swebench_lite",
    "local_tasks_to_sft",
    "python_skeleton_from_solution",
    "safe_id",
    "swebench_to_sft",
    "traces_to_sft",
    "write_python_task_repo",
]
