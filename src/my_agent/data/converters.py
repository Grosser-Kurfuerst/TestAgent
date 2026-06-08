from __future__ import annotations

"""LLaMA-Factory / Alpaca format conversion for SFT samples.

Takes JSONL SFT records (instruction/input/output/metadata) and produces
the alpaca-format JSON arrays expected by LLaMA-Factory, plus dataset metadata.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from my_agent.data.jsonl import read_jsonl
from my_agent.data.sft_samples import validate_sft_sample

DEFAULT_SYSTEM_PROMPT = (
    "你是一个 Claude Code 风格的代码仓库 Agent。"
    "你需要根据用户任务、仓库上下文和历史工具轨迹，输出严格可解析的 JSON。"
    "优先选择最小、安全、可测试的修改方案。"
)


@dataclass
class AlpacaOutput:
    """Result of an alpaca export run."""

    train_path: Path
    val_path: Path
    dataset_info_path: Path
    stats_path: Path
    total: int = 0
    train: int = 0
    val: int = 0
    skipped: int = 0
    errors: tuple[str, ...] = ()
    source_counts: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"Total records: {self.total}",
            f"Train: {self.train}",
            f"Val:   {self.val}",
            f"Skipped: {self.skipped}",
            "",
            "Source counts:",
        ]
        if self.source_counts:
            for name, count in sorted(self.source_counts.items()):
                lines.append(f"  {name}: {count}")
        else:
            lines.append("  (none)")
        lines.append("")
        lines.extend(
            [
                f"Train:           {self.train_path}",
                f"Val:             {self.val_path}",
                f"Dataset info:    {self.dataset_info_path}",
                f"Dataset stats:   {self.stats_path}",
            ]
        )
        if self.errors:
            lines.append("")
            lines.append(f"Errors: {len(self.errors)}")
            for message in self.errors[:5]:
                lines.append(f"  - {message}")
            if len(self.errors) > 5:
                lines.append(f"  - ... {len(self.errors) - 5} more")
        return "\n".join(lines)


def export_alpaca(
    input_files: list[str | Path],
    output_dir: str | Path,
    train_ratio: float = 0.95,
    seed: int = 42,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    train_name: str = "coding_agent_train",
    val_name: str = "coding_agent_val",
    strict: bool = True,
) -> AlpacaOutput:
    """Read one or more SFT JSONL files and produce LLaMA-Factory alpaca splits.

    Parameters
    ----------
    input_files:
        Paths to SFT JSONL files. Every file must exist and contain valid SFT records.
    output_dir:
        Directory where train_alpaca.json, val_alpaca.json, dataset_info.json,
        and dataset_stats.json will be written.
    train_ratio:
        Fraction of pooled records used for the training split.
    seed:
        Random seed for the train/val shuffle.
    system_prompt:
        System prompt embedded in every alpaca record.
    train_name / val_name:
        Dataset keys used inside dataset_info.json.
    """
    _validate_train_ratio(train_ratio)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # pool rows from all input files
    rows: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    skipped = 0
    errors: list[str] = []
    for input_file in input_files:
        path = Path(input_file)
        if not path.exists():
            message = f"SFT input file not found: {path}"
            if strict:
                raise FileNotFoundError(message)
            errors.append(message)
            skipped += 1
            continue
        read_result = read_jsonl(path, strict=strict, allow_single=True)
        errors.extend(read_result.errors)
        skipped += read_result.skipped
        for line_num, record in read_result.records:
            try:
                validate_sft_sample(record, path=path, line_num=line_num)
            except ValueError as exc:
                if strict:
                    raise
                errors.append(str(exc))
                skipped += 1
                continue
            source = _resolve_source(record, path.name)
            rows.append(_to_alpaca(record, source, system_prompt))
            source_counts[source] = source_counts.get(source, 0) + 1

    # shuffle & split
    rng = random.Random(seed)
    rng.shuffle(rows)
    split_at = _split_index(len(rows), train_ratio)
    train_rows = rows[:split_at]
    val_rows = rows[split_at:]

    # write train / val alpaca JSON
    train_path = output / "train_alpaca.json"
    val_path = output / "val_alpaca.json"
    train_path.write_text(_dumps(train_rows), encoding="utf-8")
    val_path.write_text(_dumps(val_rows), encoding="utf-8")

    # dataset_info.json
    info: dict[str, Any] = {
        train_name: {
            "file_name": train_path.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        val_name: {
            "file_name": val_path.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
    }
    dataset_info_path = output / "dataset_info.json"
    dataset_info_path.write_text(_dumps(info), encoding="utf-8")

    # dataset_stats.json
    stats: dict[str, Any] = {
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "skipped": skipped,
        "source_counts": source_counts,
        "errors": errors,
    }
    stats_path = output / "dataset_stats.json"
    stats_path.write_text(_dumps(stats), encoding="utf-8")

    return AlpacaOutput(
        train_path=train_path,
        val_path=val_path,
        dataset_info_path=dataset_info_path,
        stats_path=stats_path,
        total=len(rows),
        train=len(train_rows),
        val=len(val_rows),
        skipped=skipped,
        errors=tuple(errors),
        source_counts=source_counts,
    )


def _to_alpaca(record: dict[str, Any], source_name: str, system_prompt: str) -> dict[str, str]:
    """Convert one normalized SFT record into an alpaca-style dict."""
    instruction = str(record["instruction"])
    user_input = record["input"]
    output = record["output"]
    return {
        "system": system_prompt,
        "instruction": f"{instruction}\n\n来源: {source_name}",
        "input": json.dumps(user_input, ensure_ascii=False, indent=2),
        "output": json.dumps(output, ensure_ascii=False, indent=2),
    }


def _validate_train_ratio(train_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be greater than 0 and less than 1.")


def _split_index(total: int, train_ratio: float) -> int:
    if total <= 0:
        return 0
    if total == 1:
        return 1
    split_at = int(total * train_ratio)
    return max(1, min(total - 1, split_at))


def _resolve_source(record: dict[str, Any], fallback: str) -> str:
    """Extract a human-readable source label from a SFT record.

    Prefers ``record["metadata"]["source"]``, then ``record["source"]``,
    and finally *fallback* (typically the input filename).
    """
    meta = record.get("metadata")
    if isinstance(meta, dict):
        source = meta.get("source")
        if isinstance(source, str) and source.strip():
            return source.strip()
    top = record.get("source")
    if isinstance(top, str) and top.strip():
        return top.strip()
    return fallback


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
