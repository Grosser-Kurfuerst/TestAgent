from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver.selection.contracts import (
    ExperienceCandidate,
    SelectedExperience,
)
from my_agent.memory.evolver.selection.legacy import (
    ExperienceSelector,
    selection_candidate_summary,
    selection_score,
    selection_tier_counts,
)
from my_agent.memory.evolver.selection.rendering import (
    render_selected_experiences,
)
from my_agent.memory.experience.models import ExperienceTier
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType, RetrievalHit
from tests.memory.experience.fixtures import typed_experience


NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


def _experience(
    memory_id: str,
    content: str,
    tier: ExperienceTier,
    *,
    source_task: str | None = None,
    writer_confidence: float = 1.0,
    attribution_value: float = 0.0,
    attribution_confidence: float = 0.0,
    candidate_count: int = 0,
) :
    return typed_experience(
        memory_id,
        content,
        tier,
        created_at=NOW,
        source_task=source_task if source_task is not None else f"task-{memory_id}",
        writer_confidence=writer_confidence,
        attribution_value=attribution_value,
        attribution_confidence=attribution_confidence,
        candidate_count=candidate_count,
        selected_count=candidate_count,
    )


def _hit(entry, score: float = 1.0, terms: tuple[str, ...] = ("pytest",)) -> RetrievalHit:
    return RetrievalHit(
        entry=entry,
        score=score,
        matched_terms=terms,
        source_weight=1.2,
        time_decay=1.0,
    )


def _selector(
    *,
    tier_caps: dict[str, int] | None = None,
    selected_max_items: int = 20,
    min_score: float = 0.0,
) -> ExperienceSelector:
    return ExperienceSelector(
        tier_weights={"trajectory": 0.8, "tip": 1.0, "skill": 1.25, "tool": 1.35},
        tier_caps=tier_caps or {"trajectory": 1, "tip": 8, "skill": 8, "tool": 8},
        selected_max_items=selected_max_items,
        min_score=min_score,
    )


class EvolverSelectorTests(unittest.TestCase):
    def test_four_tier_experience_hits_convert_to_candidates(self) -> None:
        hits = [
            _hit(_experience("traj", "trajectory pytest failure", ExperienceTier.TRAJECTORY)),
            _hit(_experience("tip", "tip pytest failure", ExperienceTier.TIP)),
            _hit(_experience("skill", "skill pytest failure", ExperienceTier.SKILL)),
            _hit(_experience("tool", "tool pytest failure", ExperienceTier.TOOL)),
        ]

        result = _selector().select(query="pytest failure", hits=hits, max_tokens=1_000)

        self.assertEqual({candidate.tier for candidate in result.candidates}, set(ExperienceTier))
        self.assertEqual(len(result.selected), 4)
        self.assertEqual(selection_tier_counts(result.candidates)["tool"], 1)

    def test_non_typed_hit_is_rejected_at_selector_boundary(self) -> None:
        plain = MemoryEntry.build(
            id="fact",
            content="ordinary pytest fact",
            type=MemoryType.FACT,
            scope=MemoryScope.PROJECT,
            source="manual",
            token_count=4,
            project_key="/repo",
            created_at=NOW,
        )

        with self.assertRaisesRegex(TypeError, "ExperienceMemory"):
            _selector().select(query="pytest", hits=[_hit(plain)], max_tokens=1_000)

    def test_candidate_retains_original_retrieval_hit(self) -> None:
        hit = _hit(_experience("tip", "pytest fixture cleanup", ExperienceTier.TIP))
        result = _selector().select(query="pytest", hits=[hit], max_tokens=1_000)
        self.assertIs(result.candidates[0].hit, hit)
        self.assertEqual(result.context.hits, [hit])

    def test_deduplicates_within_tier_and_keeps_highest_selection_score(self) -> None:
        low = _experience("low", "same pytest learning", ExperienceTier.TIP)
        high = _experience(
            "high",
            "same pytest learning",
            ExperienceTier.TIP,
            attribution_value=0.5,
            attribution_confidence=1.0,
            candidate_count=2,
        )
        result = _selector().select(
            query="pytest",
            hits=[_hit(low), _hit(high)],
            max_tokens=1_000,
        )
        self.assertEqual([candidate.id for candidate in result.candidates], ["high"])

    def test_same_fingerprint_in_tip_and_skill_both_enter_candidate_pool(self) -> None:
        content = "same cross tier pytest learning"
        result = _selector().select(
            query="pytest",
            hits=[
                _hit(_experience("tip", content, ExperienceTier.TIP)),
                _hit(_experience("skill", content, ExperienceTier.SKILL)),
            ],
            max_tokens=1_000,
        )
        self.assertEqual({candidate.id for candidate in result.candidates}, {"tip", "skill"})

    def test_tier_caps_apply_before_total_limit(self) -> None:
        hits = [
            _hit(_experience("tip-a", "pytest tip a", ExperienceTier.TIP), 1.0),
            _hit(_experience("tip-b", "pytest tip b", ExperienceTier.TIP), 0.9),
            _hit(_experience("skill", "pytest skill", ExperienceTier.SKILL), 0.8),
        ]
        result = _selector(tier_caps={"tip": 1, "skill": 1}).select(
            query="pytest", hits=hits, max_tokens=1_000
        )
        self.assertEqual([item.candidate.id for item in result.selected], ["tip-a", "skill"])
        self.assertEqual(result.metadata["selected_tier_counts"], {"tip": 1, "skill": 1})

    def test_selected_total_does_not_exceed_selected_max_items(self) -> None:
        hits = [
            _hit(_experience("tip", "pytest tip", ExperienceTier.TIP), 1.0),
            _hit(_experience("skill", "pytest skill", ExperienceTier.SKILL), 0.9),
            _hit(_experience("tool", "pytest tool", ExperienceTier.TOOL), 0.8),
        ]
        result = _selector(selected_max_items=2).select(query="pytest", hits=hits, max_tokens=1_000)
        self.assertEqual(len(result.selected), 2)
        self.assertEqual(len(result.context.hits), 2)

    def test_selected_rendering_respects_token_budget_and_skips_oversized_entry(self) -> None:
        big = _experience("big", "oversized pytest memory " * 120, ExperienceTier.TIP)
        small = _experience("small", "small pytest tip", ExperienceTier.TIP)
        result = _selector().select(
            query="pytest", hits=[_hit(big, 1.0), _hit(small, 0.9)], max_tokens=45
        )
        self.assertEqual([item.candidate.id for item in result.selected], ["small"])
        self.assertLessEqual(result.context.estimated_tokens, 45)

    def test_selection_score_matches_legacy_formula_fixture(self) -> None:
        entry = _experience(
            "scored",
            "pytest scored",
            ExperienceTier.SKILL,
            attribution_value=0.25,
            attribution_confidence=0.8,
            candidate_count=4,
        )
        hit = _hit(entry, score=0.9)
        expected = 0.9 * 1.25 * 1.25 * 0.8
        self.assertAlmostEqual(selection_score(hit, tier_weights={"skill": 1.25}), expected)

    def test_writer_confidence_is_used_without_attribution_evidence(self) -> None:
        low = _hit(_experience("low", "pytest low", ExperienceTier.TIP, writer_confidence=0.6))
        high = _hit(_experience("high", "pytest high", ExperienceTier.TIP, writer_confidence=0.9))
        result = _selector().select(query="pytest", hits=[low, high], max_tokens=1_000)
        self.assertEqual(result.candidates[0].id, "high")

    def test_zero_attribution_confidence_is_not_defaulted_when_evidence_exists(self) -> None:
        hit = _hit(
            _experience(
                "lowconf",
                "pytest low confidence",
                ExperienceTier.TIP,
                writer_confidence=1.0,
                attribution_confidence=0.0,
                candidate_count=1,
            )
        )
        self.assertEqual(selection_score(hit, tier_weights={"tip": 1.0}), 0.5)

    def test_render_selected_experiences_uses_prompt_and_trace_contract(self) -> None:
        hit = _hit(_experience("skill", "rerun the exact pytest file", ExperienceTier.SKILL))
        candidate = ExperienceCandidate(
            id="skill",
            hit=hit,
            tier=ExperienceTier.SKILL,
            retrieval_score=1.0,
            selection_score=1.25,
            matched_terms=("pytest",),
            token_count=estimate_tokens(hit.entry.content),
        )
        context = render_selected_experiences(
            [SelectedExperience(candidate=candidate, rank=1, reason="test")],
            max_tokens=1_000,
        )
        summary = selection_candidate_summary(candidate)
        self.assertTrue(context.injected_text.startswith("Relevant selected experience:"))
        self.assertIn("memory_id=skill", context.injected_text)
        self.assertIn("tier=skill", context.injected_text)
        self.assertIn("source_task=task-skill", context.injected_text)
        self.assertEqual(summary["tier"], "skill")
        self.assertEqual(summary["score"], 1.25)
        self.assertEqual(context.hits, [hit])

    def test_policy_name_matches_trace_contract(self) -> None:
        result = _selector().select(
            query="pytest",
            hits=[_hit(_experience("tip", "pytest fixture cleanup", ExperienceTier.TIP))],
            max_tokens=1_000,
        )
        self.assertEqual(result.policy, "rule_tier_weighted_v1")

    def test_source_task_is_sanitized_for_prompt_and_summary(self) -> None:
        path_source = "/home/user/private/project/tests/test_case.py"
        secret_source = "task?api_key=sk-test-token"
        hits = [
            _hit(_experience("path", "pytest path task", ExperienceTier.TIP, source_task=path_source)),
            _hit(_experience("secret", "pytest secret task", ExperienceTier.SKILL, source_task=secret_source)),
        ]
        result = _selector().select(query="pytest", hits=hits, max_tokens=1_000)
        summaries = [selection_candidate_summary(candidate) for candidate in result.candidates]
        self.assertNotIn(path_source, result.context.injected_text)
        self.assertNotIn(secret_source, result.context.injected_text)
        self.assertIn("task_ref_", result.context.injected_text)
        self.assertIn("source_task=[redacted]", result.context.injected_text)
        self.assertNotIn(path_source, str(summaries))


if __name__ == "__main__":
    unittest.main()
