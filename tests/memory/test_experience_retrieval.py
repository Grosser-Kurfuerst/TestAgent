from __future__ import annotations

# ruff: noqa: E402 - tests add the src layout before importing project modules

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.experience_retrieval import ExperienceRetriever
from my_agent.memory.experience_store import ExperienceStore
from tests.memory.experience_fixtures import typed_experience


NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


class ExperienceRetrievalTests(unittest.TestCase):
    def test_payload_only_fields_are_searchable_in_each_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            memories = (
                typed_experience(
                    "trajectory",
                    "generic trajectory content",
                    ExperienceTier.TRAJECTORY,
                    created_at=NOW,
                    payload=TrajectoryPayload(
                        task_description="generic task",
                        steps=(ExperienceTrajectoryStep(step_num=1, action="inspect", reward=1.0),),
                        outcome="success",
                        tags=("cachepoison",),
                    ),
                ),
                typed_experience(
                    "tip",
                    "generic tip content",
                    ExperienceTier.TIP,
                    created_at=NOW,
                    payload=TipPayload(
                        category="debugging",
                        severity="warning",
                        trigger="fixtureleak",
                    ),
                ),
                typed_experience(
                    "skill",
                    "generic skill content",
                    ExperienceTier.SKILL,
                    created_at=NOW,
                    payload=SkillPayload(
                        category="testing",
                        technique="narrowrerun",
                        preconditions=(),
                        steps=("isolatedassertion",),
                    ),
                ),
                typed_experience(
                    "tool",
                    "generic tool content",
                    ExperienceTier.TOOL,
                    created_at=NOW,
                    payload=ToolPayload(
                        name="focused_test",
                        language="bash",
                        code="pytestcode tests/unit -q",
                    ),
                ),
            )
            for memory in memories:
                store.add(memory)
            retriever = ExperienceRetriever(now=NOW)

            expectations = {
                "cachepoison": "trajectory",
                "fixtureleak": "tip",
                "isolatedassertion": "skill",
                "pytestcode": "tool",
            }
            for query, expected_id in expectations.items():
                with self.subTest(query=query):
                    hits = retriever.retrieve_candidates(
                        query,
                        store=store,
                        project_key="/repo",
                        top_k_per_tier=2,
                    )
                    self.assertEqual([hit.entry.id for hit in hits], [expected_id])

    def test_top_k_is_applied_per_tier_before_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            for index in range(8):
                store.add(typed_experience(
                    f"skill-{index}",
                    f"pytest skill candidate {index}",
                    ExperienceTier.SKILL,
                    created_at=NOW,
                ))
            store.add(typed_experience("tip", "pytest tip candidate", ExperienceTier.TIP, created_at=NOW))
            store.add(typed_experience("tool", "pytest tool candidate", ExperienceTier.TOOL, created_at=NOW))

            hits = ExperienceRetriever(now=NOW).retrieve_candidates(
                "pytest",
                store=store,
                project_key="/repo",
                top_k_per_tier=1,
            )

            tiers = [hit.entry.tier for hit in hits]
            self.assertEqual(tiers.count(ExperienceTier.SKILL), 1)
            self.assertEqual(tiers.count(ExperienceTier.TIP), 1)
            self.assertEqual(tiers.count(ExperienceTier.TOOL), 1)

    def test_trajectory_indexes_only_positive_reward_step_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience(
                "trajectory",
                "generic trajectory content",
                ExperienceTier.TRAJECTORY,
                created_at=NOW,
                payload=TrajectoryPayload(
                    task_description="generic task",
                    steps=(
                        ExperienceTrajectoryStep(
                            step_num=1, action="successfulneedle", result="passed", reward=1.0
                        ),
                        ExperienceTrajectoryStep(
                            step_num=2, action="failedneedle", result="failed", reward=0.0
                        ),
                        ExperienceTrajectoryStep(
                            step_num=3, action="unknownneedle", result="unknown", reward=None
                        ),
                    ),
                    outcome="success",
                ),
            ))
            retriever = ExperienceRetriever(now=NOW)

            success = retriever.retrieve_candidates(
                "successfulneedle", store=store, project_key="/repo", top_k_per_tier=2
            )
            failed = retriever.retrieve_candidates(
                "failedneedle", store=store, project_key="/repo", top_k_per_tier=2
            )
            unknown = retriever.retrieve_candidates(
                "unknownneedle", store=store, project_key="/repo", top_k_per_tier=2
            )

            self.assertEqual([hit.entry.id for hit in success], ["trajectory"])
            self.assertEqual(failed, ())
            self.assertEqual(unknown, ())

    def test_tool_args_schema_field_name_and_description_are_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience(
                "tool",
                "generic tool content",
                ExperienceTier.TOOL,
                created_at=NOW,
                payload=ToolPayload(
                    name="generic_tool",
                    language="python",
                    code="pass",
                    args_schema={
                        "type": "object",
                        "properties": {
                            "unique_path_field": {
                                "type": "object",
                                "description": "uniquedescriptionneedle",
                                "properties": {
                                    "nested_public_field": {
                                        "type": "string",
                                        "description": "publicnesteddescription",
                                    }
                                },
                                "default": {
                                    "private_runtime_key": "defaultsecretvalue",
                                },
                                "examples": [
                                    {"private_example_key": "examplesecretvalue"},
                                ],
                            }
                        },
                    },
                ),
            ))
            retriever = ExperienceRetriever(now=NOW)

            field_hits = retriever.retrieve_candidates(
                "unique_path_field",
                store=store,
                project_key="/repo",
                top_k_per_tier=2,
            )
            description_hits = retriever.retrieve_candidates(
                "uniquedescriptionneedle",
                store=store,
                project_key="/repo",
                top_k_per_tier=2,
            )
            nested_field_hits = retriever.retrieve_candidates(
                "nested_public_field",
                store=store,
                project_key="/repo",
                top_k_per_tier=2,
            )
            nested_description_hits = retriever.retrieve_candidates(
                "publicnesteddescription",
                store=store,
                project_key="/repo",
                top_k_per_tier=2,
            )

            self.assertEqual([hit.entry.id for hit in field_hits], ["tool"])
            self.assertEqual([hit.entry.id for hit in description_hits], ["tool"])
            self.assertEqual([hit.entry.id for hit in nested_field_hits], ["tool"])
            self.assertEqual([hit.entry.id for hit in nested_description_hits], ["tool"])
            for raw_data_term in (
                "private_runtime_key",
                "defaultsecretvalue",
                "private_example_key",
                "examplesecretvalue",
            ):
                with self.subTest(raw_data_term=raw_data_term):
                    self.assertEqual(
                        retriever.retrieve_candidates(
                            raw_data_term,
                            store=store,
                            project_key="/repo",
                            top_k_per_tier=2,
                        ),
                        (),
                    )

    def test_equal_score_order_is_independent_of_jsonl_line_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ids_by_order: list[list[str]] = []
            for name, order in (("forward", ("b", "a")), ("reverse", ("a", "b"))):
                store = ExperienceStore.from_dir(Path(tmp) / name)
                for memory_id in order:
                    store.add(typed_experience(
                        memory_id,
                        f"pytest tie {memory_id}",
                        ExperienceTier.TIP,
                        created_at=NOW,
                    ))
                hits = ExperienceRetriever(now=NOW).retrieve_candidates(
                    "pytest",
                    store=store,
                    project_key="/repo",
                    top_k_per_tier=5,
                )
                ids_by_order.append([hit.entry.id for hit in hits])
            self.assertEqual(ids_by_order, [["a", "b"], ["a", "b"]])

    def test_indexed_and_tier_bucket_fallback_have_recall_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            for memory in (
                typed_experience("tip", "测试失败后先清理缓存", ExperienceTier.TIP, created_at=NOW),
                typed_experience("skill", "pytest focused retry", ExperienceTier.SKILL, created_at=NOW),
                typed_experience("other", "unrelated record", ExperienceTier.TIP, created_at=NOW),
            ):
                store.add(memory)
            indexed = ExperienceRetriever(now=NOW)
            indexed_ids = [
                hit.entry.id
                for hit in indexed.retrieve_candidates(
                    "测试", store=store, project_key="/repo", top_k_per_tier=5
                )
            ]
            self.assertEqual(indexed.last_metrics.retrieval_fallback, "")
            self.assertGreater(indexed.last_metrics.posting_candidate_count, 0)

            fallback = ExperienceRetriever(now=NOW)
            with patch.object(store, "index_snapshot", side_effect=RuntimeError("index unavailable")):
                fallback_ids = [
                    hit.entry.id
                    for hit in fallback.retrieve_candidates(
                        "测试", store=store, project_key="/repo", top_k_per_tier=5
                    )
                ]

            self.assertEqual(fallback_ids, indexed_ids)
            self.assertEqual(fallback.last_metrics.retrieval_fallback, "tier_bucket_scan")
            self.assertEqual(fallback.last_metrics.posting_candidate_count, 0)

    def test_invalidated_and_other_project_memories_are_not_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience("active", "pytest active", created_at=NOW))
            store.add(typed_experience("invalid", "pytest invalid", created_at=NOW, invalidated=True))
            store.add(typed_experience(
                "other-project", "pytest other", created_at=NOW, project_key="/other"
            ))

            hits = ExperienceRetriever(now=NOW).retrieve_candidates(
                "pytest", store=store, project_key="/repo", top_k_per_tier=5
            )

            self.assertEqual([hit.entry.id for hit in hits], ["active"])

    def test_lexical_index_is_reused_until_repository_revision_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience("first", "pytest first", created_at=NOW))
            retriever = ExperienceRetriever(now=NOW)

            retriever.retrieve_candidates(
                "pytest",
                store=store,
                project_key="/repo",
                top_k_per_tier=5,
            )
            first_index = retriever.last_index
            retriever.retrieve_candidates(
                "pytest",
                store=store,
                project_key="/repo",
                top_k_per_tier=5,
            )
            self.assertIs(retriever.last_index, first_index)

            store.add(typed_experience("second", "pytest second", created_at=NOW))
            retriever.retrieve_candidates(
                "pytest",
                store=store,
                project_key="/repo",
                top_k_per_tier=5,
            )

            self.assertIsNot(retriever.last_index, first_index)
            self.assertEqual(
                retriever.last_index.repository_revision,
                store.index_snapshot().revision,
            )


if __name__ == "__main__":
    unittest.main()
