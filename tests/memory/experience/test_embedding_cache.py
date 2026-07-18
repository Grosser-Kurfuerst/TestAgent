from __future__ import annotations

import unittest

from my_agent.memory.embedding_cache import EmbeddingCache, EmbeddingCacheKey


def _key(*, revision: str, content_hash: str = "sha256:content") -> EmbeddingCacheKey:
    return EmbeddingCacheKey(
        embedding_model_revision="embed-rev",
        tokenizer_revision="tokenizer-rev",
        repository_revision=revision,
        memory_id="mem-1",
        content_hash=content_hash,
        embedding_prompt_version="prompt-v1",
    )


class EmbeddingCacheTests(unittest.TestCase):
    def test_unchanged_content_reuses_vector_in_new_repository_revision(self) -> None:
        cache = EmbeddingCache()
        cache.put(_key(revision="repo-1"), (1.0, 0.0))

        self.assertEqual(cache.get(_key(revision="repo-2")), (1.0, 0.0))
        self.assertEqual(cache.content_entry_count, 1)
        self.assertEqual(cache.revision_entry_count, 2)

    def test_content_change_does_not_reuse_stale_vector(self) -> None:
        cache = EmbeddingCache()
        cache.put(_key(revision="repo-1"), (1.0, 0.0))

        self.assertIsNone(cache.get(_key(revision="repo-2", content_hash="sha256:changed")))


if __name__ == "__main__":
    unittest.main()
