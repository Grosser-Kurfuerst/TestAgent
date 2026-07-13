from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from my_agent.cli.common import CliContext
from my_agent.data import (
    build_humaneval,
    build_mbpp,
    build_swebench_lite,
    export_alpaca,
    local_tasks_to_sft,
    swebench_to_sft,
    traces_to_sft,
)
from my_agent.memory.evolver import (
    AttributionConfig,
    UsageLogger,
    annotate_selector_dataset_scores,
    annotate_writer_dataset_scores,
    attribution_summary,
    load_attribution_jsonl,
    render_attribution_summary,
    score_all_memories,
    selection_from_trace,
    usage_entry_from_result_row,
    write_attribution_jsonl,
    write_back_attribution,
    write_dataset_summary_json,
)
from my_agent.memory.long_term import LongTermMemoryStore

DATA_COMMANDS = {
    "build-mbpp",
    "build-humaneval",
    "build-swebench",
    "swebench-to-sft",
    "tasks-to-sft",
    "traces-to-sft",
    "export-alpaca",
    "build-memory-usage-log",
    "score-memory-attribution",
    "score-memory-datasets",
    "data",
}


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    def add_limit_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--limit", type=int, default=50, help="Maximum number of dataset rows to process.")

    def add_output_dir_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", required=True, help="Output directory for generated data.")

    mbpp_parser = subparsers.add_parser("build-mbpp", help="Build MBPP task repos and SFT samples.")
    add_limit_arg(mbpp_parser)
    add_output_dir_arg(mbpp_parser)
    mbpp_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")
    mbpp_parser.set_defaults(_handler=handle)

    he_parser = subparsers.add_parser("build-humaneval", help="Build HumanEval task repos and SFT samples.")
    add_limit_arg(he_parser)
    add_output_dir_arg(he_parser)
    he_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")
    he_parser.set_defaults(_handler=handle)

    swe_parser = subparsers.add_parser("build-swebench", help="Build SWE-bench Lite task manifests.")
    add_limit_arg(swe_parser)
    add_output_dir_arg(swe_parser)
    swe_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")
    swe_parser.set_defaults(_handler=handle)

    swe2sft_parser = subparsers.add_parser("swebench-to-sft", help="Convert SWE-bench manifests to SFT samples.")
    swe2sft_parser.add_argument("--input", required=True, help="SWE-bench Lite task JSONL file.")
    swe2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    swe2sft_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip malformed input records and include them in the report.",
    )
    swe2sft_parser.set_defaults(_handler=handle)

    tasks2sft_parser = subparsers.add_parser(
        "tasks-to-sft",
        help="Convert local task manifests to SFT strategy samples.",
    )
    tasks2sft_parser.add_argument(
        "--input",
        required=True,
        help="Task manifest JSONL file (can also be a single JSON object).",
    )
    tasks2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    tasks2sft_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip malformed input records and include them in the report.",
    )
    tasks2sft_parser.set_defaults(_handler=handle)

    trace2sft_parser = subparsers.add_parser("traces-to-sft", help="Convert agent trace JSONL files to SFT samples.")
    trace2sft_parser.add_argument("--input", required=True, help="Trace JSONL file or directory of trace files.")
    trace2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    trace2sft_parser.add_argument("--strict", action="store_true", help="Fail on malformed trace JSONL records.")
    trace2sft_parser.set_defaults(_handler=handle)

    alpaca_parser = subparsers.add_parser(
        "export-alpaca",
        help="Convert SFT JSONL files to LLaMA-Factory alpaca format.",
    )
    alpaca_parser.add_argument("--inputs", nargs="*", default=[], help="SFT JSONL files to include.")
    alpaca_parser.add_argument("--output-dir", required=True, help="Directory for alpaca outputs.")
    alpaca_parser.add_argument("--train-ratio", type=float, default=0.95, help="Train split ratio (default 0.95).")
    alpaca_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    alpaca_parser.add_argument("--system-prompt", help="Override the default system prompt.")
    alpaca_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip malformed SFT records and include them in dataset_stats.json.",
    )
    alpaca_parser.set_defaults(_handler=handle)

    _add_memory_attribution_parsers(subparsers)
    data_parser = subparsers.add_parser("data", help="OPD-Evolver data utilities.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    _add_memory_attribution_parsers(data_subparsers)


def _add_memory_attribution_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    usage_parser = subparsers.add_parser(
        "build-memory-usage-log",
        help="Build OPD-Evolver memory usage logs from manifest results and traces.",
    )
    usage_parser.add_argument("--results", required=True, help="Manifest results.jsonl path.")
    usage_parser.add_argument("--trace-dir", help="Optional directory for resolving trace paths.")
    usage_parser.add_argument("--output", required=True, help="Output usage log JSONL path.")
    usage_parser.add_argument("--append", action="store_true", help="Append instead of replacing output.")
    usage_parser.add_argument("--strict", action="store_true", help="Fail on bad trace rows or missing joins.")
    usage_parser.set_defaults(_handler=handle)

    attribution_parser = subparsers.add_parser(
        "score-memory-attribution",
        help="Score experience memories from usage logs.",
    )
    attribution_parser.add_argument("--memory-dir", required=True, help="Directory containing long_term_memory.jsonl.")
    attribution_parser.add_argument(
        "--memory-project-key",
        required=True,
        type=_nonempty_text,
        help="Exact project/stream key to score.",
    )
    attribution_parser.add_argument("--usage-log", help="Usage log JSONL path (default: <memory-dir>/usage_logs.jsonl).")
    attribution_parser.add_argument("--output", help="Attribution JSONL path (default: <memory-dir>/memory_attribution.jsonl).")
    attribution_parser.add_argument("--min-candidate-count", type=int, default=2)
    attribution_parser.add_argument("--min-selected-count", type=int, default=1)
    attribution_parser.add_argument("--min-not-selected-count", type=int, default=1)
    attribution_parser.add_argument("--value-clip", type=float, default=0.5)
    attribution_parser.add_argument("--min-abs-value-to-write", type=float, default=0.01)
    attribution_parser.add_argument("--write-back", action="store_true", help="Update long-term memory metadata.")
    attribution_parser.add_argument("--dry-run", action="store_true", help="Do not write back metadata.")
    attribution_parser.add_argument("--all-projects", action="store_true", help="Allow write-back across project keys.")
    attribution_parser.add_argument("--top-n", type=int, default=5, help="Number of top/bottom records in summary.")
    attribution_parser.set_defaults(_handler=handle)

    dataset_parser = subparsers.add_parser(
        "score-memory-datasets",
        help="Annotate writer/selector datasets with memory attribution scores.",
    )
    dataset_parser.add_argument("--memory-dir", help="Compatibility option; attribution file supplies scores.")
    dataset_parser.add_argument("--attribution", required=True, help="memory_attribution.jsonl path.")
    dataset_parser.add_argument("--writer-dataset", help="Writer dataset JSONL input.")
    dataset_parser.add_argument("--writer-output", help="Writer scored JSONL output.")
    dataset_parser.add_argument("--selector-dataset", help="Selector dataset JSONL input.")
    dataset_parser.add_argument("--selector-output", help="Selector scored JSONL output.")
    dataset_parser.add_argument("--score-mode", choices=("binary", "weighted"), default="weighted")
    dataset_parser.add_argument("--threshold", type=float, default=0.0)
    dataset_parser.add_argument("--w-success", type=float, default=0.8)
    dataset_parser.add_argument("--w-mean", type=float, default=0.2)
    dataset_parser.add_argument("--only-missing", action="store_true", help="Keep rows that already have numeric score.")
    dataset_parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    _ = ctx
    command = getattr(args, "data_command", None) or args.command
    try:
        if command == "build-mbpp":
            result = build_mbpp(output_dir=args.output_dir, limit=args.limit, split=args.split)
        elif command == "build-humaneval":
            result = build_humaneval(output_dir=args.output_dir, limit=args.limit, split=args.split)
        elif command == "build-swebench":
            result = build_swebench_lite(output_dir=args.output_dir, limit=args.limit, split=args.split)
        elif command == "swebench-to-sft":
            result = swebench_to_sft(input_path=args.input, output_path=args.output, strict=not args.non_strict)
        elif command == "tasks-to-sft":
            result = local_tasks_to_sft(input_path=args.input, output_path=args.output, strict=not args.non_strict)
        elif command == "traces-to-sft":
            result = traces_to_sft(trace_path=args.input, output_path=args.output, strict=args.strict)
        elif command == "export-alpaca":
            inputs = args.inputs if args.inputs else []
            if not inputs:
                print("Error: --inputs is required (one or more SFT JSONL files).", file=sys.stderr)
                return 1
            kwargs: dict[str, object] = {
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "strict": not args.non_strict,
            }
            if args.system_prompt:
                kwargs["system_prompt"] = args.system_prompt
            result = export_alpaca(input_files=inputs, output_dir=args.output_dir, **kwargs)
        elif command == "build-memory-usage-log":
            result = _build_memory_usage_log(args)
        elif command == "score-memory-attribution":
            result = _score_memory_attribution(args)
        elif command == "score-memory-datasets":
            result = _score_memory_datasets(args)
        else:
            raise ValueError(f"Unknown data command: {command}")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if command == "build-swebench":
            print("Hint: install data dependencies: uv sync --extra data", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.render())
    return 0


class _RenderedResult:
    def __init__(self, text: str) -> None:
        self._text = text

    def render(self) -> str:
        return self._text


def _build_memory_usage_log(args: argparse.Namespace) -> _RenderedResult:
    results_path = Path(args.results)
    rows = _read_jsonl(results_path)
    entries = []
    summary = {
        "output": str(args.output),
        "usage_logs": 0,
        "missing_trace": 0,
        "missing_selection": 0,
        "missing_outcome": 0,
        "missing_memory_project_key": 0,
        "bad_trace": 0,
        "selection_events_seen": 0,
        "selection_events_used": 0,
    }
    for row in rows:
        trace_path = _resolve_trace_path(row, results_path.parent, Path(args.trace_dir) if args.trace_dir else None)
        if trace_path is None:
            summary["missing_trace"] += 1
            _raise_if_strict(args.strict, f"missing trace for task_id={row.get('task_id', '')}")
            continue
        try:
            trace_events, bad_lines = _read_trace_events_for_cli(trace_path, strict=args.strict)
        except ValueError:
            summary["bad_trace"] += 1
            if args.strict:
                raise
            continue
        if bad_lines:
            summary["bad_trace"] += 1

        row_with_trace = dict(row)
        row_with_trace["trace_path"] = str(trace_path)
        entry = usage_entry_from_result_row(row_with_trace, trace_events=trace_events)
        selection = selection_from_trace(
            trace_events,
            run_id=entry.run_id if entry is not None else str(row.get("run_id") or ""),
        )
        summary["selection_events_seen"] += selection.selection_events_seen
        summary["selection_events_used"] += selection.selection_events_used
        if selection.is_empty:
            summary["missing_selection"] += 1
            _raise_if_strict(args.strict, f"missing selection for task_id={row.get('task_id', '')}")
            continue

        if entry is None:
            summary["missing_outcome"] += 1
            _raise_if_strict(args.strict, f"missing outcome for task_id={row.get('task_id', '')}")
            continue
        if entry.stream_id and not entry.memory_project_key:
            summary["missing_memory_project_key"] += 1
            _raise_if_strict(args.strict, f"missing memory_project_key for task_id={entry.task_id}")
            continue
        entries.append(entry)

    logger = UsageLogger(args.output)
    count = logger.append_many(entries) if args.append else logger.overwrite(entries)
    summary["usage_logs"] = count
    _write_summary_json(summary, args.output)
    return _RenderedResult(_render_usage_summary(summary))


def _score_memory_attribution(args: argparse.Namespace) -> _RenderedResult:
    memory_dir = Path(args.memory_dir)
    usage_log = Path(args.usage_log) if args.usage_log else memory_dir / "usage_logs.jsonl"
    output = Path(args.output) if args.output else memory_dir / "memory_attribution.jsonl"
    store = LongTermMemoryStore.from_dir(memory_dir)
    store.load()
    if not usage_log.exists():
        raise FileNotFoundError(str(usage_log))
    logs = UsageLogger(usage_log).load_all()
    config = AttributionConfig(
        min_candidate_count=args.min_candidate_count,
        min_selected_count=args.min_selected_count,
        min_not_selected_count=args.min_not_selected_count,
        value_clip=args.value_clip,
    )
    records = score_all_memories(
        entries=store.all(project_key=None),
        usage_logs=logs,
        project_key=str(args.memory_project_key or ""),
        config=config,
    )
    write_attribution_jsonl(records, output)
    write_back_summary = None
    if args.write_back and not args.dry_run:
        write_back_summary = write_back_attribution(
            store=store,
            records=records,
            project_key=str(args.memory_project_key or "") if args.memory_project_key else None,
            all_projects=bool(args.all_projects),
            min_abs_value_to_write=float(args.min_abs_value_to_write),
            min_candidate_count=int(args.min_candidate_count),
        )
    summary = attribution_summary(
        records,
        output=output,
        top_n=args.top_n,
        write_back=write_back_summary,
        config=config,
    )
    _write_summary_json(summary, output)
    return _RenderedResult(render_attribution_summary(summary))


def _score_memory_datasets(args: argparse.Namespace) -> _RenderedResult:
    attribution_path = Path(args.attribution)
    if not attribution_path.exists():
        raise FileNotFoundError(str(attribution_path))
    attribution = load_attribution_jsonl(attribution_path)
    if not args.writer_dataset and not args.selector_dataset:
        raise ValueError("--writer-dataset or --selector-dataset is required")
    rendered: list[str] = []
    if args.writer_dataset:
        if not args.writer_output:
            raise ValueError("--writer-output is required with --writer-dataset")
        writer_summary = annotate_writer_dataset_scores(
            dataset_path=args.writer_dataset,
            attribution=attribution,
            output_path=args.writer_output,
            only_missing=args.only_missing,
        )
        write_dataset_summary_json(writer_summary)
        rendered.append("Writer dataset\n" + writer_summary.render())
    if args.selector_dataset:
        if not args.selector_output:
            raise ValueError("--selector-output is required with --selector-dataset")
        selector_summary = annotate_selector_dataset_scores(
            dataset_path=args.selector_dataset,
            attribution=attribution,
            output_path=args.selector_output,
            score_mode=args.score_mode,
            threshold=args.threshold,
            w_success=args.w_success,
            w_mean=args.w_mean,
            only_missing=args.only_missing,
        )
        write_dataset_summary_json(selector_summary)
        rendered.append("Selector dataset\n" + selector_summary.render())
    return _RenderedResult("\n\n".join(rendered))


def _nonempty_text(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise argparse.ArgumentTypeError("value must not be empty")
    return normalized


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} line {lineno}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"invalid JSONL in {path} line {lineno}: expected object")
            rows.append(data)
    return rows


def _resolve_trace_path(row: dict[str, object], results_dir: Path, trace_dir: Path | None) -> Path | None:
    raw = str(row.get("trace_path") or "")
    candidates: list[Path] = []
    if raw:
        trace_path = Path(raw)
        candidates.append(trace_path)
        if not trace_path.is_absolute():
            candidates.append(results_dir / trace_path)
            if trace_dir is not None:
                candidates.append(trace_dir / trace_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if trace_dir is not None and raw:
        basename = Path(raw).name
        for candidate in sorted(trace_dir.rglob(basename)):
            if candidate.exists():
                return candidate
    if trace_dir is not None:
        identifiers = _trace_identifiers(row)
        for identifier in identifiers:
            for candidate in sorted(trace_dir.rglob(f"{identifier}.jsonl")):
                if candidate.exists():
                    return candidate
        for candidate in sorted(trace_dir.rglob("*.jsonl")):
            if _trace_matches_row(candidate, identifiers):
                return candidate
    return None


def _read_trace_events_for_cli(
    path: Path,
    *,
    strict: bool,
) -> tuple[list[dict[str, object]], int]:
    events: list[dict[str, object]] = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"invalid trace JSONL in {path} line {lineno}: {exc}") from exc
                bad_lines += 1
                continue
            if isinstance(data, dict):
                events.append(data)
    return events, bad_lines


def _trace_identifiers(row: dict[str, object]) -> list[str]:
    seen: set[str] = set()
    identifiers: list[str] = []
    for key in ("task_id", "run_id", "source_task"):
        value = str(row.get(key) or "").strip()
        if value and value not in seen:
            seen.add(value)
            identifiers.append(value)
    return identifiers


def _trace_matches_row(path: Path, identifiers: list[str]) -> bool:
    if not identifiers:
        return False
    identifiers_set = set(identifiers)
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    payload = {}
                values = {
                    str(event.get("task_id") or ""),
                    str(event.get("run_id") or ""),
                    str(payload.get("task_id") or ""),
                    str(payload.get("run_id") or ""),
                }
                if identifiers_set.intersection(v for v in values if v):
                    return True
    except OSError:
        return False
    return False


def _write_summary_json(summary: dict[str, object], output: str | Path) -> Path:
    path = Path(str(output) + ".summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _render_usage_summary(summary: dict[str, object]) -> str:
    return (
        f"Usage logs: {int(summary.get('usage_logs') or 0)}\n"
        f"Missing trace: {int(summary.get('missing_trace') or 0)}\n"
        f"Missing selection: {int(summary.get('missing_selection') or 0)}\n"
        f"Missing outcome: {int(summary.get('missing_outcome') or 0)}\n"
        f"Missing memory project key: {int(summary.get('missing_memory_project_key') or 0)}\n"
        f"Bad trace: {int(summary.get('bad_trace') or 0)}\n"
        f"Selection events: {int(summary.get('selection_events_used') or 0)}/"
        f"{int(summary.get('selection_events_seen') or 0)} used\n"
        f"Output: {summary.get('output') or ''}"
    )


def _raise_if_strict(strict: bool, message: str) -> None:
    if strict:
        raise ValueError(message)
