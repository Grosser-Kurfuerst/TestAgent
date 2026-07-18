"""Experience retrieval contracts and lexical/embedding backends."""

from my_agent.memory.experience.retrieval.contracts import (
    ExperienceRetriever,
    PerTierHits,
    RetrievalMetrics,
    flatten_per_tier_hits,
)
from my_agent.memory.experience.retrieval.embedding import (
    EMBEDDING_PROMPT_VERSION,
    EmbeddingEncoder,
    EmbeddingRetrievalError,
    EmbeddingRetrievalMetrics,
    EmbeddingRetriever,
    TransformersEmbeddingEncoder,
    cosine_similarity,
    normalize_vector,
)
from my_agent.memory.experience.retrieval.embedding_cache import (
    EmbeddingCache,
    EmbeddingCacheKey,
)
from my_agent.memory.experience.retrieval.embedding_index import (
    EmbeddingIndexEntry,
    EmbeddingIndexSnapshot,
)
from my_agent.memory.experience.retrieval.lexical import (
    ExperienceRetrievalMetrics,
    LexicalExperienceRetriever,
    LexicalIndexSnapshot,
    build_lexical_index,
)
from my_agent.memory.experience.retrieval.text import (
    experience_index_terms,
    experience_searchable_text,
    tokenize_experience_text,
)

__all__ = [
    "EMBEDDING_PROMPT_VERSION",
    "EmbeddingCache",
    "EmbeddingCacheKey",
    "EmbeddingEncoder",
    "EmbeddingIndexEntry",
    "EmbeddingIndexSnapshot",
    "EmbeddingRetrievalError",
    "EmbeddingRetrievalMetrics",
    "EmbeddingRetriever",
    "ExperienceRetrievalMetrics",
    "ExperienceRetriever",
    "LexicalExperienceRetriever",
    "LexicalIndexSnapshot",
    "PerTierHits",
    "RetrievalMetrics",
    "TransformersEmbeddingEncoder",
    "build_lexical_index",
    "cosine_similarity",
    "experience_index_terms",
    "experience_searchable_text",
    "flatten_per_tier_hits",
    "normalize_vector",
    "tokenize_experience_text",
]
