"""Formal per-tier cosine retrieval using a local embedding model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Protocol, Sequence

from my_agent.config import AgentConfig
from my_agent.memory.embedding_cache import EmbeddingCache, EmbeddingCacheKey
from my_agent.memory.embedding_index import EmbeddingIndexEntry, EmbeddingIndexSnapshot
from my_agent.memory.evolver.types import ExperienceMemory, ExperienceTier
from my_agent.memory.experience_retrieval import experience_searchable_text
from my_agent.memory.experience_store import ExperienceStore, ExperienceStoreIndexSnapshot
from my_agent.memory.types import RetrievalHit
from my_agent.policy.identity import canonical_sha256


EMBEDDING_PROMPT_VERSION = "opd-qwen3-embedding-v1"
QUERY_INSTRUCTION = (
    "Instruct: Retrieve coding-agent experience that is useful for solving the task.\n"
    "Query: "
)


class EmbeddingRetrievalError(RuntimeError):
    pass


class EmbeddingEncoder(Protocol):
    model_revision: str
    tokenizer_revision: str

    def encode_queries(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def encode_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True)
class EmbeddingRetrievalMetrics:
    repository_revision: str = ""
    returned_count: int = 0
    encoded_count: int = 0
    cache_hit_count: int = 0
    per_tier: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "repository_revision": self.repository_revision,
            "returned_count": self.returned_count,
            "embedding_encoded_count": self.encoded_count,
            "embedding_cache_hit_count": self.cache_hit_count,
            "retrieval_per_tier": {key: dict(value) for key, value in self.per_tier.items()},
            "retrieval_backend": "embedding_cosine",
            "retrieval_fallback": "",
        }


class EmbeddingRetriever:
    def __init__(self, encoder: EmbeddingEncoder, *, cache: EmbeddingCache | None = None) -> None:
        if not getattr(encoder, "model_revision", "") or not getattr(encoder, "tokenizer_revision", ""):
            raise ValueError("embedding encoder requires model and tokenizer revisions")
        self.encoder = encoder
        self.cache = cache or EmbeddingCache()
        self.last_metrics = EmbeddingRetrievalMetrics()
        self.last_index: EmbeddingIndexSnapshot | None = None

    def retrieve_per_tier(
        self,
        query: str,
        *,
        store: ExperienceStore,
        project_key: str,
        top_k_per_tier: int = 50,
    ) -> dict[ExperienceTier, tuple[RetrievalHit[ExperienceMemory], ...]]:
        top_k = int(top_k_per_tier)
        if top_k < 1:
            raise ValueError("top_k_per_tier must be >= 1")
        try:
            repository = store.index_snapshot()
            query_vectors = self.encoder.encode_queries((QUERY_INSTRUCTION + str(query),))
            query_vector = _single_vector(query_vectors, "query")
            index, encoded_count, cache_hit_count = self._build_index(
                repository,
                project_key=project_key,
            )
        except Exception as exc:
            if isinstance(exc, (ValueError, EmbeddingRetrievalError)):
                raise
            raise EmbeddingRetrievalError(f"formal embedding retrieval failed: {type(exc).__name__}: {exc}") from exc

        results: dict[ExperienceTier, tuple[RetrievalHit[ExperienceMemory], ...]] = {}
        per_tier: dict[str, dict[str, int]] = {}
        returned_total = 0
        for tier in ExperienceTier:
            entries = index.entries_by_tier[tier]
            hits = [
                RetrievalHit(
                    entry=entry.memory,
                    score=cosine_similarity(query_vector, entry.vector),
                    matched_terms=(),
                    source_weight=1.0,
                    time_decay=1.0,
                )
                for entry in entries
            ]
            hits.sort(key=lambda hit: (-float(hit.score), hit.entry.id))
            returned = tuple(hits[:top_k])
            results[tier] = returned
            returned_total += len(returned)
            per_tier[tier.value] = {
                "visible_count": len(entries),
                "scored_count": len(entries),
                "returned_count": len(returned),
            }
        self.last_index = index
        self.last_metrics = EmbeddingRetrievalMetrics(
            repository_revision=index.repository_revision,
            returned_count=returned_total,
            encoded_count=encoded_count,
            cache_hit_count=cache_hit_count,
            per_tier=per_tier,
        )
        return results

    def retrieve_candidates(
        self,
        query: str,
        *,
        store: ExperienceStore,
        project_key: str,
        top_k_per_tier: int = 50,
    ) -> tuple[RetrievalHit[ExperienceMemory], ...]:
        per_tier = self.retrieve_per_tier(
            query,
            store=store,
            project_key=project_key,
            top_k_per_tier=top_k_per_tier,
        )
        return tuple(hit for tier in ExperienceTier for hit in per_tier[tier])

    def _build_index(
        self,
        repository: ExperienceStoreIndexSnapshot,
        *,
        project_key: str,
    ) -> tuple[EmbeddingIndexSnapshot, int, int]:
        entries_by_tier: dict[ExperienceTier, tuple[EmbeddingIndexEntry, ...]] = {}
        encoded_count = 0
        cache_hit_count = 0
        for tier in ExperienceTier:
            memory_ids = set(repository.global_ids_by_tier.get(tier, ()))
            if project_key:
                memory_ids.update(repository.project_ids_by_tier.get((project_key, tier), ()))
            memories = tuple(
                repository.by_id[memory_id]
                for memory_id in sorted(memory_ids)
                if not repository.by_id[memory_id].invalidated
            )
            vectors: dict[str, tuple[float, ...]] = {}
            pending: list[tuple[ExperienceMemory, EmbeddingCacheKey, str]] = []
            for memory in memories:
                text = experience_searchable_text(memory)
                content_hash = canonical_sha256(text)
                key = EmbeddingCacheKey(
                    embedding_model_revision=self.encoder.model_revision,
                    tokenizer_revision=self.encoder.tokenizer_revision,
                    repository_revision=repository.revision,
                    memory_id=memory.id,
                    content_hash=content_hash,
                    embedding_prompt_version=EMBEDDING_PROMPT_VERSION,
                )
                cached = self.cache.get(key)
                if cached is None:
                    pending.append((memory, key, text))
                else:
                    cache_hit_count += 1
                    vectors[memory.id] = cached
            if pending:
                encoded = self.encoder.encode_documents(tuple(item[2] for item in pending))
                if len(encoded) != len(pending):
                    raise EmbeddingRetrievalError("embedding backend returned the wrong document count")
                for (memory, key, _text), vector in zip(pending, encoded):
                    normalized = normalize_vector(vector)
                    self.cache.put(key, normalized)
                    vectors[memory.id] = normalized
                    encoded_count += 1
            entries_by_tier[tier] = tuple(
                EmbeddingIndexEntry(
                    memory=memory,
                    vector=vectors[memory.id],
                    content_hash=canonical_sha256(experience_searchable_text(memory)),
                )
                for memory in memories
            )
        return EmbeddingIndexSnapshot(repository.revision, entries_by_tier), encoded_count, cache_hit_count


class TransformersEmbeddingEncoder:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        model_revision: str,
        tokenizer_revision: str,
        device: Any,
        max_length: int = 8_192,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.device = device
        self.max_length = max_length

    @classmethod
    def from_config(cls, config: AgentConfig) -> "TransformersEmbeddingEncoder":
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("formal embedding retrieval requires the 'opd-embed' extra") from exc
        snapshot = Path(snapshot_download(
            repo_id=config.embedding_model,
            revision=config.embedding_revision,
        ))
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side="left")
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": _embedding_dtype(torch, config.policy_dtype),
        }
        if config.policy_device == "auto":
            model_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(snapshot, **model_kwargs)
        if config.policy_device != "auto" and hasattr(model, "to"):
            model = model.to(config.policy_device)
        if hasattr(model, "eval"):
            model.eval()
        revision = snapshot.name if snapshot.parent.name == "snapshots" else config.embedding_revision
        return cls(
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            model_revision=revision,
            tokenizer_revision=revision,
            device=getattr(model, "device", config.policy_device),
        )

    def encode_queries(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with self.torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
            pooled = _last_token_pool(hidden, batch["attention_mask"], self.torch)
            normalized = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return tuple(tuple(float(value) for value in row) for row in normalized.float().cpu().tolist())


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(item) for item in vector)
    if not values:
        raise ValueError("embedding vector must not be empty")
    norm = sqrt(sum(item * item for item in values))
    if norm <= 0.0:
        raise ValueError("embedding vector norm must be positive")
    return tuple(item / norm for item in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    normalized_left = normalize_vector(left)
    normalized_right = normalize_vector(right)
    if len(normalized_left) != len(normalized_right):
        raise ValueError("embedding dimensions must match")
    return sum(a * b for a, b in zip(normalized_left, normalized_right))


def _single_vector(vectors: Sequence[Sequence[float]], label: str) -> tuple[float, ...]:
    if len(vectors) != 1:
        raise EmbeddingRetrievalError(f"embedding backend returned the wrong {label} count")
    return normalize_vector(vectors[0])


def _last_token_pool(hidden: Any, attention_mask: Any, torch: Any) -> Any:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden.shape[0]
    return hidden[torch.arange(batch_size, device=hidden.device), sequence_lengths]


def _embedding_dtype(torch: Any, value: str) -> Any:
    dtype = {
        "bfloat16": getattr(torch, "bfloat16", None),
        "float16": getattr(torch, "float16", None),
        "float32": getattr(torch, "float32", None),
    }.get(value)
    if dtype is None:
        raise ValueError(f"unsupported embedding dtype: {value!r}")
    return dtype


__all__ = [
    "EMBEDDING_PROMPT_VERSION",
    "EmbeddingEncoder",
    "EmbeddingRetrievalError",
    "EmbeddingRetrievalMetrics",
    "EmbeddingRetriever",
    "TransformersEmbeddingEncoder",
    "cosine_similarity",
    "normalize_vector",
]
