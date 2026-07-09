"""Annotate OPD-Evolver writer/selector datasets with attribution scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping
import json

from my_agent.memory.evolver.attribution import MemoryAttributionRecord
from my_agent.memory.evolver.usage_log import flatten_tier_ids
from my_agent.text_safety import sanitize_json_value


SCORING_SOURCE = "memory_attribution_v1"


@dataclass(frozen=True)
class DatasetScoringSummary:
    output: str
    rows: int = 0
    rows_scored: int = 0
    missing_attribution: int = 0
    existing_score_kept: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "rows": int(self.rows),
            "rows_scored": int(self.rows_scored),
            "missing_attribution": int(self.missing_attribution),
            "existing_score_kept": int(self.existing_score_kept),
        }

    def render(self) -> str:
        return (
            f"Dataset rows scored: {self.rows_scored}/{self.rows}\n"
            f"Missing attribution: {self.missing_attribution}\n"
            f"Existing scores kept: {self.existing_score_kept}\n"
            f"Output: {self.output}"
        )


def annotate_writer_dataset_scores(
    *,
    dataset_path: str | Path,
    attribution: Mapping[str, MemoryAttributionRecord],
    output_path: str | Path,
    only_missing: bool = False,
) -> DatasetScoringSummary:
    """Score writer rows from ``saved_records[{id,tier}]`` or legacy ``saved_ids``."""
    rows = _read_jsonl(dataset_path)
    output_rows: list[dict[str, Any]] = []
    rows_scored = 0
    missing = 0
    kept = 0

    for row in rows:
        next_row = dict(row)
        if only_missing and _has_numeric_score(next_row):
            kept += 1
            output_rows.append(next_row)
            continue

        saved = _writer_saved_records(next_row)
        score_items, row_missing = _score_items(saved, attribution)
        values = [item["value"] for item in score_items]
        mean_score = mean(values) if values else None
        next_row["created_memory_scores"] = score_items
        next_row["mean_created_memory_score"] = mean_score
        next_row["score"] = mean_score
        next_row["scoring_source"] = SCORING_SOURCE
        rows_scored += 1
        missing += row_missing
        output_rows.append(next_row)

    _write_jsonl(output_path, output_rows)
    return DatasetScoringSummary(
        output=str(output_path),
        rows=len(rows),
        rows_scored=rows_scored,
        missing_attribution=missing,
        existing_score_kept=kept,
    )


def annotate_selector_dataset_scores(
    *,
    dataset_path: str | Path,
    attribution: Mapping[str, MemoryAttributionRecord],
    output_path: str | Path,
    score_mode: str = "weighted",
    threshold: float = 0.0,
    w_success: float = 0.8,
    w_mean: float = 0.2,
    only_missing: bool = False,
) -> DatasetScoringSummary:
    """Score selector rows from upstream nested or AgentCli flat schemas."""
    if score_mode not in {"binary", "weighted"}:
        raise ValueError("score_mode must be 'binary' or 'weighted'")

    rows = _read_jsonl(dataset_path)
    output_rows: list[dict[str, Any]] = []
    rows_scored = 0
    missing = 0
    kept = 0

    for row in rows:
        next_row = dict(row)
        if only_missing and _has_numeric_score(next_row):
            kept += 1
            output_rows.append(next_row)
            continue

        candidate_ids = _selector_candidate_ids(next_row)
        selected_ids = _selector_selected_ids(next_row)
        candidate_items, candidate_missing = _score_items(
            [{"id": mid, "tier": ""} for mid in candidate_ids],
            attribution,
        )
        selected_items, selected_missing = _score_items(
            [{"id": mid, "tier": ""} for mid in selected_ids],
            attribution,
        )
        selected_values = [item["value"] for item in selected_items]
        mean_selected = mean(selected_values) if selected_values else None
        success = _row_success(next_row)
        if score_mode == "binary":
            score = 1.0 if success and float(mean_selected or 0.0) >= float(threshold) else 0.0
        else:
            score = float(w_success) * float(success) + float(w_mean) * float(mean_selected or 0.0)

        next_row["candidate_memory_scores"] = candidate_items
        next_row["selected_memory_scores"] = selected_items
        next_row["mean_selected_memory_score"] = mean_selected
        next_row["score"] = score
        next_row["scoring_source"] = SCORING_SOURCE
        rows_scored += 1
        missing += candidate_missing + selected_missing
        output_rows.append(next_row)

    _write_jsonl(output_path, output_rows)
    return DatasetScoringSummary(
        output=str(output_path),
        rows=len(rows),
        rows_scored=rows_scored,
        missing_attribution=missing,
        existing_score_kept=kept,
    )


def write_dataset_summary_json(summary: DatasetScoringSummary, output_path: str | Path | None = None) -> Path:
    path = Path(str(output_path or summary.output) + ".summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {p} line {lineno}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"invalid JSONL in {p} line {lineno}: expected object")
            rows.append(data)
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(sanitize_json_value(dict(row)), ensure_ascii=False) + "\n")


def _writer_saved_records(row: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    raw_records = row.get("saved_records")
    if isinstance(raw_records, list):
        for item in raw_records:
            if isinstance(item, Mapping) and item.get("id"):
                records.append({"id": str(item.get("id")), "tier": str(item.get("tier") or "")})
    if records:
        return records
    return [{"id": str(mid), "tier": ""} for mid in (row.get("saved_ids") or []) if mid]


def _selector_candidate_ids(row: Mapping[str, Any]) -> list[str]:
    nested = row.get("retrieve")
    if isinstance(nested, Mapping):
        ids = _ids_from_candidate_payload(nested.get("candidates"))
        if ids:
            return ids
    return _ids_from_tier_or_flat(row.get("candidate_memory_ids_by_tier") or row.get("candidate_memory_ids"))


def _selector_selected_ids(row: Mapping[str, Any]) -> list[str]:
    select = row.get("select")
    if isinstance(select, Mapping):
        ids = _ids_from_tier_or_flat(select.get("selected_memory_ids") or select.get("selected_ids"))
        if ids:
            return ids
    return _ids_from_tier_or_flat(row.get("selected_memory_ids_by_tier") or row.get("selected_memory_ids"))


def _ids_from_candidate_payload(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return _ids_from_tier_or_flat(value)
    if isinstance(value, list):
        ids: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("id"):
                ids.append(str(item.get("id")))
            elif item:
                ids.append(str(item))
        return _dedupe(ids)
    return []


def _ids_from_tier_or_flat(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return flatten_tier_ids({str(k): [str(i) for i in (v or [])] for k, v in value.items()})
    if isinstance(value, list):
        return _dedupe(str(i) for i in value if i)
    if isinstance(value, tuple):
        return _dedupe(str(i) for i in value if i)
    return []


def _score_items(
    items: Iterable[Mapping[str, Any]],
    attribution: Mapping[str, MemoryAttributionRecord],
) -> tuple[list[dict[str, Any]], int]:
    scored: list[dict[str, Any]] = []
    missing = 0
    for item in items:
        memory_id = str(item.get("id") or "")
        if not memory_id:
            continue
        record = attribution.get(memory_id)
        if record is None:
            missing += 1
            scored.append({
                "id": memory_id,
                "tier": str(item.get("tier") or ""),
                "value": 0.0,
                "missing_attribution": True,
            })
            continue
        scored.append({
            "id": memory_id,
            "tier": str(item.get("tier") or record.tier),
            "value": round(float(record.value), 6),
            "confidence": round(float(record.confidence), 6),
            "missing_attribution": False,
        })
    return scored, missing


def _row_success(row: Mapping[str, Any]) -> bool:
    for key in ("success", "resolved"):
        if key in row:
            return _truthy(row.get(key))
    return str(row.get("outcome") or "").lower() == "success"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "success", "resolved", "passed"}
    return bool(value)


def _has_numeric_score(row: Mapping[str, Any]) -> bool:
    value = row.get("score")
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = [
    "DatasetScoringSummary",
    "SCORING_SOURCE",
    "annotate_selector_dataset_scores",
    "annotate_writer_dataset_scores",
    "write_dataset_summary_json",
]
