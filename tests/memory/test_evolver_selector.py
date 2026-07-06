from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    ExperienceCandidate,
    ExperienceSelector,
    ExperienceTier,
    SelectedExperience,
    build_experience_entry,
    candidate_tier,
    render_selected_experiences,
    selection_candidate_summary,
    selection_score,
    selection_tier_counts,
)
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType, RetrievalHit


NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


def _experience(
    id: str,
    content: str,
    tier: ExperienceTier | str,
    *,
    extra: dict[str, object] | None = None,
    source_task: str | None = None,
) -> MemoryEntry:
    return build_experience_entry(
        id=id,
        content=content,
        tier=tier,
        project_key="/repo",
        source_task=source_task if source_task is not None else f"task-{id}",
        extra_metadata=extra,
    )


def _plain(id: str, content: str) -> MemoryEntry:
    return MemoryEntry.build(
        id=id,
        content=content,
        type=MemoryType.FACT,
        scope=MemoryScope.PROJECT,
        source="manual",
        token_count=estimate_tokens(content),
        project_key="/repo",
        created_at=NOW,
    )


def _hit(entry: MemoryEntry, score: float = 1.0, terms: tuple[str, ...] = ("pytest",)) -> RetrievalHit:
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
        tier_weights={
            "trajectory": 0.8,
            "tip": 1.0,
            "skill": 1.25,
            "tool": 1.35,
        },
        tier_caps=tier_caps
        or {
            "trajectory": 1,
            "tip": 8,
            "skill": 8,
            "tool": 8,
        },
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

    def test_plain_fact_default_not_candidate(self) -> None:
        tip = _experience("tip", "pytest fixture cleanup", ExperienceTier.TIP)
        plain = _plain("fact", "ordinary pytest fact")

        result = _selector().select(query="pytest", hits=[_hit(plain), _hit(tip)], max_tokens=1_000)

        self.assertEqual([candidate.id for candidate in result.candidates], ["tip"])
        self.assertEqual(candidate_tier(plain), None)

    def test_candidate_retains_original_retrieval_hit(self) -> None:
        hit = _hit(_experience("tip", "pytest fixture cleanup", ExperienceTier.TIP))

        result = _selector().select(query="pytest", hits=[hit], max_tokens=1_000)

        self.assertIs(result.candidates[0].hit, hit)
        self.assertEqual(result.context.hits, [hit])

    def test_deduplicates_by_fingerprint_and_keeps_highest_selection_score(self) -> None:
        low = _experience("low", "same pytest learning", ExperienceTier.TIP)
        high = _experience("high", "same pytest learning", ExperienceTier.TIP, extra={"evolver_value": 0.5})

        result = _selector().select(
            query="pytest",
            hits=[_hit(low, score=1.0), _hit(high, score=1.0)],
            max_tokens=1_000,
        )

        self.assertEqual([candidate.id for candidate in result.candidates], ["high"])

    def test_tier_caps_apply_before_total_limit(self) -> None:
        hits = [
            _hit(_experience("tip-a", "pytest tip a", ExperienceTier.TIP), score=1.0),
            _hit(_experience("tip-b", "pytest tip b", ExperienceTier.TIP), score=0.9),
            _hit(_experience("skill", "pytest skill", ExperienceTier.SKILL), score=0.8),
        ]

        result = _selector(tier_caps={"tip": 1, "skill": 1}).select(
            query="pytest",
            hits=hits,
            max_tokens=1_000,
        )

        self.assertEqual([item.candidate.id for item in result.selected], ["tip-a", "skill"])
        self.assertEqual(result.metadata["selected_tier_counts"], {"tip": 1, "skill": 1})

    def test_selected_total_does_not_exceed_selected_max_items(self) -> None:
        hits = [
            _hit(_experience("tip-a", "pytest tip a", ExperienceTier.TIP), score=1.0),
            _hit(_experience("skill", "pytest skill", ExperienceTier.SKILL), score=0.9),
            _hit(_experience("tool", "pytest tool", ExperienceTier.TOOL), score=0.8),
        ]

        result = _selector(selected_max_items=2).select(query="pytest", hits=hits, max_tokens=1_000)

        self.assertEqual(len(result.selected), 2)
        self.assertEqual(len(result.context.hits), 2)

    def test_selected_rendering_respects_token_budget_and_skips_oversized_entry(self) -> None:
        big = _experience("big", "oversized pytest memory " * 120, ExperienceTier.TIP)
        small = _experience("small", "small pytest tip", ExperienceTier.TIP)

        result = _selector().select(
            query="pytest",
            hits=[_hit(big, score=1.0), _hit(small, score=0.9)],
            max_tokens=45,
        )

        self.assertEqual([item.candidate.id for item in result.selected], ["small"])
        self.assertIn("small pytest tip", result.context.injected_text)
        self.assertNotIn("oversized pytest memory", result.context.injected_text)
        self.assertLessEqual(result.context.estimated_tokens, 45)

    def test_selection_result_context_hits_match_selected_candidate_hits(self) -> None:
        first = _hit(_experience("first", "pytest first", ExperienceTier.TIP), score=1.0)
        second = _hit(_experience("second", "pytest second", ExperienceTier.SKILL), score=0.9)

        result = _selector().select(query="pytest", hits=[first, second], max_tokens=1_000)

        self.assertEqual(
            result.context.hits,
            [item.candidate.hit for item in result.selected],
        )

    def test_value_and_confidence_affect_selection_score(self) -> None:
        normal = _hit(_experience("normal", "pytest normal", ExperienceTier.TIP), score=1.0)
        boosted = _hit(
            _experience(
                "boosted",
                "pytest boosted",
                ExperienceTier.TIP,
                extra={"evolver_value": 0.5, "confidence": 1.2},
            ),
            score=1.0,
        )

        result = _selector().select(query="pytest", hits=[normal, boosted], max_tokens=1_000)

        self.assertEqual(result.candidates[0].id, "boosted")
        self.assertGreater(result.candidates[0].selection_score, result.candidates[1].selection_score)
        self.assertGreater(
            selection_score(boosted, ExperienceTier.TIP, boosted.entry.metadata, tier_weights={"tip": 1.0}),
            selection_score(normal, ExperienceTier.TIP, normal.entry.metadata, tier_weights={"tip": 1.0}),
        )

    def test_render_selected_experiences_uses_prompt_contract(self) -> None:
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
        selected = [SelectedExperience(candidate=candidate, rank=1, reason="test")]

        context = render_selected_experiences(selected, max_tokens=1_000)
        summary = selection_candidate_summary(candidate)

        self.assertTrue(context.injected_text.startswith("Relevant selected experience:"))
        self.assertIn("memory_id=skill", context.injected_text)
        self.assertIn("tier=skill", context.injected_text)
        self.assertIn("source_task=task-skill", context.injected_text)
        self.assertEqual(summary["tier"], "skill")
        self.assertEqual(summary["score"], 1.25)
        self.assertEqual(summary["tokens"], estimate_tokens(hit.entry.content))
        self.assertEqual(summary["retrieval_score"], 1.0)
        self.assertEqual(summary["selection_score"], 1.25)
        self.assertEqual(summary["token_count"], estimate_tokens(hit.entry.content))
        self.assertEqual(context.hits, [hit])

    def test_policy_name_matches_trace_contract(self) -> None:
        hit = _hit(_experience("tip", "pytest fixture cleanup", ExperienceTier.TIP))

        result = _selector().select(query="pytest", hits=[hit], max_tokens=1_000)

        self.assertEqual(result.policy, "rule_tier_weighted_v1")

    def test_source_task_metadata_is_sanitized_for_prompt_and_summary(self) -> None:
        path_source = "/home/user/private/project/tests/test_case.py"
        secret_source = "task?api_key=sk-test-token"
        path_hit = _hit(
            _experience(
                "path",
                "pytest path task",
                ExperienceTier.TIP,
                source_task=path_source,
            )
        )
        secret_hit = _hit(
            _experience(
                "secret",
                "pytest secret task",
                ExperienceTier.SKILL,
                source_task=secret_source,
            )
        )

        result = _selector().select(query="pytest", hits=[path_hit, secret_hit], max_tokens=1_000)
        injected_text = result.context.injected_text
        summaries = [selection_candidate_summary(candidate) for candidate in result.candidates]

        self.assertNotIn(path_source, injected_text)
        self.assertNotIn(secret_source, injected_text)
        self.assertNotIn("api_key", injected_text)
        self.assertNotIn("sk-test-token", injected_text)
        self.assertIn("task_ref_", injected_text)
        self.assertIn("source_task=[redacted]", injected_text)
        self.assertNotIn(path_source, str(summaries))
        self.assertNotIn(secret_source, str(summaries))
        self.assertNotIn("matched_terms", summaries[0])

    def test_bare_secret_like_task_refs_are_redacted(self) -> None:
        secret_refs = [
            "".join(("gh", "p_", "abcdefghijklmnopqrstuvwxyz123456")),
            "".join(("gl", "pat-", "abcdefghijklmnopqrstuvwxyz")),
            "".join(("xo", "xb", "-", "123456789012", "-", "123456789012", "-", "abcdefghijklmnopqrstuvwx")),
            "".join(("A", "KIA", "IOSFODNN7EXAMPLE")),
        ]
        hits = [
            _hit(
                _experience(
                    f"secret-{index}",
                    f"pytest secret ref {index}",
                    ExperienceTier.TIP,
                    source_task=secret_ref,
                )
            )
            for index, secret_ref in enumerate(secret_refs)
        ]

        result = _selector().select(query="pytest", hits=hits, max_tokens=1_000)
        injected_text = result.context.injected_text
        summaries = [selection_candidate_summary(candidate) for candidate in result.candidates]

        for secret_ref in secret_refs:
            self.assertNotIn(secret_ref, injected_text)
            self.assertNotIn(secret_ref, str(summaries))
        self.assertEqual(injected_text.count("source_task=[redacted]"), len(secret_refs))


if __name__ == "__main__":
    unittest.main()
