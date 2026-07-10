from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    MemoryAttributionRecord,
    annotate_selector_dataset_scores,
    annotate_writer_dataset_scores,
)


def _attr() -> dict[str, MemoryAttributionRecord]:
    return {
        "mem-good": MemoryAttributionRecord(
            memory_id="mem-good",
            tier="skill",
            value=0.2,
            confidence=0.8,
        ),
        "mem-bad": MemoryAttributionRecord(
            memory_id="mem-bad",
            tier="skill",
            value=-0.1,
            confidence=0.7,
        ),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class WriterDatasetScoringTests(unittest.TestCase):
    def test_scores_saved_records_and_counts_missing_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "writer.jsonl"
            output = base / "writer.scored.jsonl"
            _write_jsonl(dataset, [
                {
                    "task_id": "task-1",
                    "saved_records": [
                        {"id": "mem-good", "tier": "skill"},
                        {"id": "mem-missing", "tier": "tip"},
                    ],
                    "other": "kept",
                }
            ])

            summary = annotate_writer_dataset_scores(
                dataset_path=dataset,
                attribution=_attr(),
                output_path=output,
            )

            row = _read_jsonl(output)[0]
            self.assertEqual(row["other"], "kept")
            self.assertEqual(row["created_memory_scores"], {
                "skill": {"mem-good": 0.2},
                "tip": {"mem-missing": 0.0},
            })
            self.assertEqual(row["mean_created_memory_score"], 0.1)
            self.assertEqual(row["score"], 0.1)
            self.assertEqual(row["scoring_source"], "memory_attribution_v1")
            self.assertEqual(summary.rows_scored, 1)
            self.assertEqual(summary.missing_attribution, 1)

    def test_supports_legacy_saved_ids_and_only_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "writer.jsonl"
            scored = base / "writer.scored.jsonl"
            kept = base / "writer.kept.jsonl"
            _write_jsonl(dataset, [
                {"task_id": "legacy", "saved_ids": ["mem-bad"]},
                {"task_id": "already", "saved_ids": ["mem-good"], "score": 0.75},
            ])

            summary = annotate_writer_dataset_scores(
                dataset_path=dataset,
                attribution=_attr(),
                output_path=scored,
                only_missing=False,
            )
            rows = _read_jsonl(scored)
            self.assertEqual(rows[0]["score"], -0.1)
            self.assertEqual(rows[1]["score"], 0.2)
            self.assertEqual(summary.rows_scored, 2)

            kept_summary = annotate_writer_dataset_scores(
                dataset_path=dataset,
                attribution=_attr(),
                output_path=kept,
                only_missing=True,
            )
            kept_rows = _read_jsonl(kept)
            self.assertEqual(kept_rows[1]["score"], 0.75)
            self.assertNotIn("created_memory_scores", kept_rows[1])
            self.assertEqual(kept_summary.existing_score_kept, 1)


class SelectorDatasetScoringTests(unittest.TestCase):
    def test_scores_nested_upstream_like_selector_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "selector.jsonl"
            output = base / "selector.scored.jsonl"
            _write_jsonl(dataset, [
                {
                    "retrieve": {"candidates": [{"id": "mem-good"}, {"id": "mem-missing"}]},
                    "select": {"selected_memory_ids": {"skill": ["mem-good"]}},
                    "resolved": True,
                }
            ])

            summary = annotate_selector_dataset_scores(
                dataset_path=dataset,
                attribution=_attr(),
                output_path=output,
                score_mode="weighted",
                w_success=0.8,
                w_mean=0.2,
            )

            row = _read_jsonl(output)[0]
            self.assertEqual(row["candidate_memory_scores"], {
                "skill": {"mem-good": 0.2},
                "unknown": {"mem-missing": 0.0},
            })
            self.assertEqual(row["selected_memory_scores"], {
                "skill": {"mem-good": 0.2},
            })
            self.assertEqual(row["mean_selected_memory_score"], 0.2)
            self.assertEqual(row["score"], 0.8400000000000001)
            self.assertEqual(row["scoring_source"], "memory_attribution_v1")
            self.assertEqual(summary.missing_attribution, 1)

    def test_scores_agentcli_flat_schema_and_binary_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "selector.jsonl"
            output = base / "selector.scored.jsonl"
            _write_jsonl(dataset, [
                {
                    "candidate_memory_ids_by_tier": {"skill": ["mem-bad"]},
                    "selected_memory_ids_by_tier": {"skill": ["mem-bad"]},
                    "success": True,
                }
            ])

            annotate_selector_dataset_scores(
                dataset_path=dataset,
                attribution=_attr(),
                output_path=output,
                score_mode="binary",
                threshold=0.0,
            )

            row = _read_jsonl(output)[0]
            self.assertEqual(row["candidate_memory_scores"], {
                "skill": {"mem-bad": -0.1},
            })
            self.assertEqual(row["selected_memory_scores"], {
                "skill": {"mem-bad": -0.1},
            })
            self.assertEqual(row["mean_selected_memory_score"], -0.1)
            self.assertEqual(row["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
