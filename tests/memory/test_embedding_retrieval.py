from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from my_agent.memory.embedding_retrieval import (
    EmbeddingRetrievalError,
    EmbeddingRetriever,
    cosine_similarity,
)
from my_agent.memory.evolver import ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from tests.memory.experience_fixtures import typed_experience


class _FakeEncoder:
    model_revision = "embed-rev"
    tokenizer_revision = "tokenizer-rev"

    def __init__(self) -> None:
        self.document_batches: list[tuple[str, ...]] = []

    def encode_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return ((1.0, 0.0),) * len(texts)

    def encode_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.document_batches.append(tuple(texts))
        vectors = []
        for text in texts:
            vectors.append((1.0, 0.0) if "alpha" in text or "tie" in text else (0.0, 1.0))
        return tuple(vectors)


class EmbeddingRetrievalTests(unittest.TestCase):
    def test_retrieval_is_per_tier_cosine_and_ties_use_memory_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            for memory in (
                typed_experience("tip-b", "tie alpha b", ExperienceTier.TIP),
                typed_experience("tip-a", "tie alpha a", ExperienceTier.TIP),
                typed_experience("skill-z", "beta skill", ExperienceTier.SKILL),
                typed_experience("tool-a", "alpha tool", ExperienceTier.TOOL),
            ):
                store.add(memory)
            retriever = EmbeddingRetriever(_FakeEncoder())

            results = retriever.retrieve_per_tier(
                "alpha",
                store=store,
                project_key="/repo",
                top_k_per_tier=1,
            )

        self.assertEqual([hit.entry.id for hit in results[ExperienceTier.TIP]], ["tip-a"])
        self.assertEqual([hit.entry.id for hit in results[ExperienceTier.SKILL]], ["skill-z"])
        self.assertEqual([hit.entry.id for hit in results[ExperienceTier.TOOL]], ["tool-a"])
        self.assertEqual(results[ExperienceTier.TIP][0].score, 1.0)
        self.assertEqual(results[ExperienceTier.SKILL][0].score, 0.0)

    def test_new_revision_encodes_only_new_or_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience("tip-a", "alpha tip", ExperienceTier.TIP))
            encoder = _FakeEncoder()
            retriever = EmbeddingRetriever(encoder)
            retriever.retrieve_candidates("alpha", store=store, project_key="/repo")
            first_encoded = sum(len(batch) for batch in encoder.document_batches)

            store.add(typed_experience("skill-a", "alpha skill", ExperienceTier.SKILL))
            retriever.retrieve_candidates("alpha", store=store, project_key="/repo")
            total_encoded = sum(len(batch) for batch in encoder.document_batches)

        self.assertEqual(first_encoded, 1)
        self.assertEqual(total_encoded, 2)
        self.assertGreaterEqual(retriever.last_metrics.cache_hit_count, 1)

    def test_cosine_matches_reference(self) -> None:
        self.assertAlmostEqual(cosine_similarity((3.0, 4.0), (4.0, 3.0)), 24.0 / 25.0)

    def test_repository_failure_does_not_fall_back_to_lexical_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            retriever = EmbeddingRetriever(_FakeEncoder())

            with (
                patch.object(store, "index_snapshot", side_effect=RuntimeError("unavailable")),
                patch.object(store, "all") as lexical_fallback,
                self.assertRaisesRegex(EmbeddingRetrievalError, "formal embedding retrieval failed"),
            ):
                retriever.retrieve_candidates(
                    "alpha",
                    store=store,
                    project_key="/repo",
                )

            lexical_fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
