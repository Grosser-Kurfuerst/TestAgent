from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import json

from my_agent.memory.evolver.attribution_export import (
    load_attribution_events,
    load_candidate_exposures,
    write_attribution_events,
    write_candidate_exposures,
)
from my_agent.memory.evolver.attribution_schema import CandidateExposure
from my_agent.memory.evolver.paper_attribution import (
    compute_round_attribution,
    positive_selected_memory_ids,
    teacher_memory_records,
    writing_top_fraction,
)
from my_agent.policy.identity import PolicyIdentity, canonical_sha256


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


def _exposure(
    memory_id: str,
    tier: str,
    task: int,
    selected: bool,
    reward: float,
    *,
    collection_round: int = 0,
) -> CandidateExposure:
    return CandidateExposure(
        task_id=f"task-{task}",
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        memory_id=memory_id,
        tier=tier,
        selected=selected,
        reward=reward,
        collection_round=collection_round,
        task_ordinal=task,
        candidate_snapshot_hash=canonical_sha256({"task": task}),
        policy_identity=_identity(),
        repository_revision=f"rev-{task}",
        evaluator_name="pytest",
        evaluator_version="8",
        evaluator_hash=canonical_sha256({"evaluator": "pytest"}),
    )


class PaperAttributionJoinTests(unittest.TestCase):
    def test_strict_evidence_and_attribution_jsonl_round_trip(self) -> None:
        exposures = (
            _exposure("mem-a", "skill", 1, True, 1.0),
            _exposure("mem-a", "skill", 2, False, 0.0),
        )
        records = compute_round_attribution(
            exposures, collection_round=0, valid_task_ordinals=(1, 2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exposure_path = write_candidate_exposures(exposures, root / "exposures.jsonl")
            attribution_path = write_attribution_events(records, root / "attribution.jsonl")

            loaded_exposures = load_candidate_exposures(exposure_path)
            loaded_records = load_attribution_events(attribution_path)

        self.assertEqual(loaded_exposures, exposures)
        self.assertEqual(loaded_records, records)
        self.assertEqual(loaded_records[0].evidence_refs[0].candidate_snapshot_hash, exposures[0].candidate_snapshot_hash)
        self.assertEqual(loaded_records[0].evidence_refs[0].repository_revision, exposures[0].repository_revision)
        self.assertEqual(loaded_records[0].evidence_refs[0].evaluator_hash, exposures[0].evaluator_hash)

    def test_strict_loader_rejects_formula_inconsistent_ready_record(self) -> None:
        exposures = (
            _exposure("mem-a", "skill", 1, True, 1.0),
            _exposure("mem-a", "skill", 2, False, 0.0),
        )
        record = compute_round_attribution(
            exposures, collection_round=0, valid_task_ordinals=(1, 2),
        )[0]
        payload = record.to_dict()
        payload["groups"]["group-a"]["n_plus"] = 0
        payload["groups"]["group-a"]["mean_plus"] = None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_attribution_events(path)

    def test_strict_loader_rejects_tampered_embedded_exposure_binding(self) -> None:
        exposures = (
            _exposure("mem-a", "skill", 1, True, 1.0),
            _exposure("mem-a", "skill", 2, False, 0.0),
        )
        payload = compute_round_attribution(
            exposures, collection_round=0, valid_task_ordinals=(1, 2),
        )[0].to_dict()
        payload["evidence_refs"][0]["candidate_snapshot_hash"] = canonical_sha256("tampered")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered-evidence.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exposure id"):
                load_attribution_events(path)

    def test_round_accepts_authoritative_boundary_after_last_candidate_exposure(self) -> None:
        exposures = (
            _exposure("mem-a", "skill", 1, True, 1.0),
            _exposure("mem-a", "skill", 2, False, 0.0),
        )
        record = compute_round_attribution(
            exposures, collection_round=0, valid_task_ordinals=(1, 2, 3),
        )[0]
        self.assertEqual(record.as_of_ordinal, 3)
        with self.assertRaisesRegex(ValueError, "authoritative valid task"):
            compute_round_attribution(
                exposures, collection_round=0, valid_task_ordinals=(1,),
            )

    def test_positive_teacher_filter_and_writer_top_thirty_percent(self) -> None:
        exposures = []
        for index, memory_id in enumerate(("mem-a", "mem-b", "mem-c", "mem-d"), 1):
            exposures.extend((
                _exposure(memory_id, "skill", index * 2 - 1, True, 1.0),
                _exposure(memory_id, "skill", index * 2, False, 0.0),
            ))
        records = compute_round_attribution(
            tuple(exposures), collection_round=0, valid_task_ordinals=tuple(range(1, 9)),
        )
        by_id = {record.memory_id: record for record in records}

        positive = positive_selected_memory_ids(("mem-a", "missing", "mem-b"), by_id)
        teacher = teacher_memory_records(by_id, minimum_memory_score=0.01, max_items=20)
        writing = writing_top_fraction(
            ("mem-d", "mem-c", "mem-b", "mem-a"), by_id, collection_round=0,
        )
        writing_with_missing = writing_top_fraction(
            ("mem-a", "missing"), by_id, collection_round=0,
        )

        self.assertEqual(positive, ("mem-a", "mem-b"))
        self.assertEqual({record.memory_id for record in teacher}, set(by_id))
        self.assertEqual(sum(1 for item in writing if item.selected), 2)
        self.assertEqual([item.memory_id for item in writing], sorted(by_id))
        self.assertEqual([item.rank for item in writing], [1, 2, 3, 4])
        self.assertEqual(writing_with_missing[-1].reason, "no_ready_attribution")
        self.assertIsNone(writing_with_missing[-1].rank)
        self.assertEqual(writing[0].cutoff_rank, 2)
        self.assertEqual(writing[0].collection_round, 0)
        self.assertIsNotNone(writing[0].boundary_score)

    def test_writer_filter_rejects_cross_round_records(self) -> None:
        round_zero = compute_round_attribution((
            _exposure("mem-a", "skill", 1, True, 1.0),
            _exposure("mem-a", "skill", 2, False, 0.0),
        ), collection_round=0, valid_task_ordinals=(1, 2))[0]
        round_one = compute_round_attribution((
            _exposure("mem-b", "skill", 1, True, 1.0, collection_round=1),
        ), collection_round=1, valid_task_ordinals=(1,))[0]
        self.assertEqual(round_one.status, "insufficient_counterfactual_evidence")

        with self.assertRaisesRegex(ValueError, "collection_round"):
            writing_top_fraction(
                ("mem-a", "mem-b"),
                {"mem-a": round_zero, "mem-b": round_one},
                collection_round=0,
            )

    def test_filters_reject_mapping_key_record_identity_mismatch(self) -> None:
        record = compute_round_attribution((
            _exposure("mem-b", "skill", 1, True, 1.0),
            _exposure("mem-b", "skill", 2, False, 0.0),
        ), collection_round=0, valid_task_ordinals=(1, 2))[0]
        mismatched = {"mem-a": record}

        with self.assertRaisesRegex(ValueError, "mapping key"):
            positive_selected_memory_ids(("mem-a",), mismatched)
        with self.assertRaisesRegex(ValueError, "mapping key"):
            writing_top_fraction(("mem-a",), mismatched, collection_round=0)


if __name__ == "__main__":
    unittest.main()
