"""Shared, versioned contracts for memory benchmark adapters and reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import os
import re

from my_agent.policy.identity import canonical_json_bytes, require_sha256


OFFICIAL_RESULT_SCHEMA_VERSION = "memory-benchmark-official-result-v1"
TASK_RESULT_SCHEMA_VERSION = "memory-benchmark-task-result-v1"

_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_HIDDEN_ANSWER_FIELDS = frozenset(
    {
        "answer",
        "expected_answer",
        "expected_output",
        "ground_truth",
        "hidden_answer",
        "hidden_expected_output",
        "reference_answer",
        "reference_solution",
        "solution",
    }
)


@dataclass(frozen=True)
class BenchmarkTask:
    benchmark: str
    subset: str
    task_id: str
    order_index: int
    task_group: str
    instruction: str
    split: str
    source_revision: str
    content_hash: str
    environment_spec: Mapping[str, Any]
    evaluator_spec: Mapping[str, Any]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "benchmark",
            "subset",
            "task_id",
            "task_group",
            "instruction",
            "split",
            "source_revision",
        ):
            _require_non_empty(getattr(self, field_name), field_name=field_name)
        if self.order_index < 1:
            raise ValueError("order_index must be at least 1")
        expected_group = f"{self.benchmark}:{self.subset}"
        if self.task_group != expected_group:
            raise ValueError(f"task_group must be normalized as {expected_group!r}")
        if self.split != "test":
            raise ValueError("benchmark task split must be 'test'")
        if _IMMUTABLE_REVISION.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a full immutable revision")
        require_sha256(self.content_hash, field_name="content_hash")
        environment_spec = dict(self.environment_spec)
        _reject_hidden_answer_fields(environment_spec)
        object.__setattr__(self, "environment_spec", environment_spec)
        object.__setattr__(self, "evaluator_spec", dict(self.evaluator_spec))
        object.__setattr__(self, "tags", _string_tuple(self.tags, field_name="tags"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "subset": self.subset,
            "task_id": self.task_id,
            "order_index": self.order_index,
            "task_group": self.task_group,
            "instruction": self.instruction,
            "split": self.split,
            "source_revision": self.source_revision,
            "content_hash": self.content_hash,
            "environment_spec": dict(self.environment_spec),
            "evaluator_spec": dict(self.evaluator_spec),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkTask":
        return cls(
            benchmark=str(data["benchmark"]),
            subset=str(data["subset"]),
            task_id=str(data["task_id"]),
            order_index=_required_int(data["order_index"], field_name="order_index"),
            task_group=str(data["task_group"]),
            instruction=str(data["instruction"]),
            split=str(data["split"]),
            source_revision=str(data["source_revision"]),
            content_hash=str(data["content_hash"]),
            environment_spec=_required_mapping(
                data["environment_spec"], field_name="environment_spec"
            ),
            evaluator_spec=_required_mapping(
                data["evaluator_spec"], field_name="evaluator_spec"
            ),
            tags=_sequence_of_strings(data.get("tags", ()), field_name="tags"),
        )


@dataclass(frozen=True)
class PreparedBenchmarkTask:
    task: BenchmarkTask
    repo_path: Path
    public_prompt: str
    agent_test_command: tuple[str, ...] | None
    initial_environment_command: tuple[str, ...]
    hidden_evaluator_command: tuple[str, ...]
    env_overrides: Mapping[str, str]
    action_log_path: Path
    runtime_action_log_path: Path
    adapter_state_path: Path
    public_tool_state_path: Path
    official_result_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.task, BenchmarkTask):
            raise ValueError("task must be a BenchmarkTask")
        _require_non_empty(self.public_prompt, field_name="public_prompt")
        for field_name in (
            "repo_path",
            "action_log_path",
            "runtime_action_log_path",
            "adapter_state_path",
            "public_tool_state_path",
            "official_result_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise ValueError(f"{field_name} must be a Path")
        if self.agent_test_command is not None:
            object.__setattr__(
                self,
                "agent_test_command",
                _command_tuple(self.agent_test_command, field_name="agent_test_command"),
            )
        object.__setattr__(
            self,
            "initial_environment_command",
            _command_tuple(
                self.initial_environment_command,
                field_name="initial_environment_command",
            ),
        )
        object.__setattr__(
            self,
            "hidden_evaluator_command",
            _command_tuple(
                self.hidden_evaluator_command,
                field_name="hidden_evaluator_command",
            ),
        )
        object.__setattr__(
            self,
            "env_overrides",
            _string_mapping(self.env_overrides, field_name="env_overrides"),
        )


@dataclass(frozen=True)
class MemoryContextSelection:
    backend: str
    candidate_count: int
    selected_ids: tuple[str, ...]
    selected_texts: tuple[str, ...]
    selected_content_tokens: int
    injected_text: str
    estimated_tokens: int
    retrieval_elapsed_sec: float

    def __post_init__(self) -> None:
        _require_non_empty(self.backend, field_name="backend")
        _require_non_negative(self.candidate_count, field_name="candidate_count")
        _require_non_negative(
            self.selected_content_tokens,
            field_name="selected_content_tokens",
        )
        _require_non_negative(self.estimated_tokens, field_name="estimated_tokens")
        _require_non_negative_float(
            self.retrieval_elapsed_sec,
            field_name="retrieval_elapsed_sec",
        )
        object.__setattr__(
            self,
            "selected_ids",
            _string_tuple(self.selected_ids, field_name="selected_ids"),
        )
        object.__setattr__(
            self,
            "selected_texts",
            _string_tuple(self.selected_texts, field_name="selected_texts"),
        )
        if len(self.selected_ids) != len(self.selected_texts):
            raise ValueError("selected_ids and selected_texts must have equal length")
        if self.candidate_count < len(self.selected_ids):
            raise ValueError("candidate_count cannot be smaller than selected count")
        if len(self.selected_ids) > 20:
            raise ValueError("memory selection cannot exceed 20 items")
        if self.selected_content_tokens > 1800:
            raise ValueError("selected memory content cannot exceed 1800 tokens")
        if not self.selected_ids and (
            self.selected_content_tokens != 0
            or self.estimated_tokens != 0
            or self.injected_text
        ):
            raise ValueError("empty memory selection must not contain rendered context")
        if self.backend == "no_memory" and (
            self.candidate_count != 0
            or self.selected_ids
            or self.selected_texts
            or self.selected_content_tokens != 0
            or self.injected_text
            or self.estimated_tokens != 0
            or self.retrieval_elapsed_sec != 0.0
        ):
            raise ValueError("no_memory selection must be entirely empty and zero-cost")

    @property
    def selected_count(self) -> int:
        return len(self.selected_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "candidate_count": self.candidate_count,
            "selected_ids": list(self.selected_ids),
            "selected_texts": list(self.selected_texts),
            "selected_content_tokens": self.selected_content_tokens,
            "injected_text": self.injected_text,
            "estimated_tokens": self.estimated_tokens,
            "retrieval_elapsed_sec": self.retrieval_elapsed_sec,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryContextSelection":
        return cls(
            backend=str(data["backend"]),
            candidate_count=_required_int(
                data["candidate_count"], field_name="candidate_count"
            ),
            selected_ids=_sequence_of_strings(
                data["selected_ids"], field_name="selected_ids"
            ),
            selected_texts=_sequence_of_strings(
                data["selected_texts"], field_name="selected_texts"
            ),
            selected_content_tokens=_required_int(
                data["selected_content_tokens"],
                field_name="selected_content_tokens",
            ),
            injected_text=str(data["injected_text"]),
            estimated_tokens=_required_int(
                data["estimated_tokens"], field_name="estimated_tokens"
            ),
            retrieval_elapsed_sec=_required_float(
                data["retrieval_elapsed_sec"], field_name="retrieval_elapsed_sec"
            ),
        )


@dataclass(frozen=True)
class PublicEpisode:
    task_id: str
    instruction: str
    actions: tuple[Mapping[str, Any], ...]
    final_response: str
    resolved: bool
    reward: float
    failure_type: str

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, field_name="task_id")
        _require_non_empty(self.instruction, field_name="instruction")
        _require_reward(self.reward)
        if not isinstance(self.resolved, bool):
            raise ValueError("resolved must be a bool")
        normalized_actions: list[Mapping[str, Any]] = []
        for index, action in enumerate(self.actions):
            normalized_actions.append(
                _required_mapping(action, field_name=f"actions[{index}]")
            )
        object.__setattr__(self, "actions", tuple(normalized_actions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "actions": [dict(action) for action in self.actions],
            "final_response": self.final_response,
            "resolved": self.resolved,
            "reward": self.reward,
            "failure_type": self.failure_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublicEpisode":
        raw_actions = data["actions"]
        if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
            raise ValueError("actions must be an array")
        return cls(
            task_id=str(data["task_id"]),
            instruction=str(data["instruction"]),
            actions=tuple(
                _required_mapping(action, field_name="action") for action in raw_actions
            ),
            final_response=str(data["final_response"]),
            resolved=_required_bool(data["resolved"], field_name="resolved"),
            reward=_required_float(data["reward"], field_name="reward"),
            failure_type=str(data["failure_type"]),
        )


@dataclass(frozen=True)
class OfficialEvaluatorResult:
    task_id: str
    evaluator_hash: str
    resolved: bool
    reward: float
    status: str = "ok"

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, field_name="task_id")
        require_sha256(self.evaluator_hash, field_name="evaluator_hash")
        if not isinstance(self.resolved, bool):
            raise ValueError("resolved must be a bool")
        _require_reward(self.reward)
        if self.status != "ok":
            raise ValueError("official evaluator status must be 'ok'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OFFICIAL_RESULT_SCHEMA_VERSION,
            "task_id": self.task_id,
            "evaluator_hash": self.evaluator_hash,
            "resolved": self.resolved,
            "reward": self.reward,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialEvaluatorResult":
        if data.get("schema_version") != OFFICIAL_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported official evaluator result schema")
        return cls(
            task_id=str(data["task_id"]),
            evaluator_hash=str(data["evaluator_hash"]),
            resolved=_required_bool(data["resolved"], field_name="resolved"),
            reward=_required_float(data["reward"], field_name="reward"),
            status=str(data["status"]),
        )


def write_official_result_atomic(
    path: str | Path,
    result: OfficialEvaluatorResult,
) -> Path:
    """Durably replace one official scorer result without exposing partial JSON."""

    if not isinstance(result, OfficialEvaluatorResult):
        raise ValueError("result must be an OfficialEvaluatorResult")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(result.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_official_evaluator_result(
    path: str | Path,
    *,
    expected_task_id: str,
    expected_evaluator_hash: str,
    returncode: int,
) -> OfficialEvaluatorResult:
    """Load a final scorer result and bind it to command return-code semantics."""

    if returncode not in {0, 1}:
        raise ValueError(f"official evaluator returned infrastructure code {returncode}")
    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("official evaluator result must be a JSON object")
    expected_fields = {
        "schema_version",
        "task_id",
        "evaluator_hash",
        "resolved",
        "reward",
        "status",
    }
    if set(payload) != expected_fields:
        raise ValueError("official evaluator result fields do not match the v1 schema")
    official = OfficialEvaluatorResult.from_dict(payload)
    if official.task_id != expected_task_id:
        raise ValueError("official evaluator result task_id mismatch")
    if official.evaluator_hash != expected_evaluator_hash:
        raise ValueError("official evaluator result evaluator_hash mismatch")
    if official.resolved is not (returncode == 0):
        raise ValueError("official evaluator result conflicts with scorer return code")
    return official


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ProviderUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_negative(value, field_name=field_name)

    @property
    def resolved_total_tokens(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None

    @property
    def available(self) -> bool:
        return self.resolved_total_tokens is not None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderUsage":
        return cls(
            prompt_tokens=_optional_int(data.get("prompt_tokens"), field_name="prompt_tokens"),
            completion_tokens=_optional_int(
                data.get("completion_tokens"), field_name="completion_tokens"
            ),
            total_tokens=_optional_int(data.get("total_tokens"), field_name="total_tokens"),
        )


@dataclass(frozen=True)
class ExternalMemoryItem:
    memory_id: str
    text: str
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.memory_id, field_name="memory_id")
        _require_non_empty(self.text, field_name="text")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score must be finite when provided")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "score": self.score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalMemoryItem":
        raw_score = data.get("score")
        return cls(
            memory_id=str(data["memory_id"]),
            text=str(data["text"]),
            score=(
                None
                if raw_score is None
                else _required_float(raw_score, field_name="score")
            ),
            metadata=_required_mapping(data.get("metadata", {}), field_name="metadata"),
        )


@dataclass(frozen=True)
class Mem0SearchResult:
    items: tuple[ExternalMemoryItem, ...]
    llm_usage: ProviderUsage
    embedding_calls: int
    embedding_elapsed_sec: float
    elapsed_sec: float

    def __post_init__(self) -> None:
        if not all(isinstance(item, ExternalMemoryItem) for item in self.items):
            raise ValueError("items must contain ExternalMemoryItem values")
        if not isinstance(self.llm_usage, ProviderUsage):
            raise ValueError("llm_usage must be ProviderUsage")
        _require_non_negative(self.embedding_calls, field_name="embedding_calls")
        _require_non_negative_float(
            self.embedding_elapsed_sec,
            field_name="embedding_elapsed_sec",
        )
        _require_non_negative_float(self.elapsed_sec, field_name="elapsed_sec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "llm_usage": self.llm_usage.to_dict(),
            "embedding_calls": self.embedding_calls,
            "embedding_elapsed_sec": self.embedding_elapsed_sec,
            "elapsed_sec": self.elapsed_sec,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Mem0SearchResult":
        raw_items = data["items"]
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            raise ValueError("items must be an array")
        return cls(
            items=tuple(
                ExternalMemoryItem.from_dict(
                    _required_mapping(item, field_name="memory item")
                )
                for item in raw_items
            ),
            llm_usage=ProviderUsage.from_dict(
                _required_mapping(data["llm_usage"], field_name="llm_usage")
            ),
            embedding_calls=_required_int(
                data["embedding_calls"], field_name="embedding_calls"
            ),
            embedding_elapsed_sec=_required_float(
                data["embedding_elapsed_sec"],
                field_name="embedding_elapsed_sec",
            ),
            elapsed_sec=_required_float(data["elapsed_sec"], field_name="elapsed_sec"),
        )


@dataclass(frozen=True)
class Mem0WriteResult:
    written_ids: tuple[str, ...]
    llm_usage: ProviderUsage
    embedding_calls: int
    embedding_elapsed_sec: float
    elapsed_sec: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "written_ids",
            _string_tuple(self.written_ids, field_name="written_ids"),
        )
        if not isinstance(self.llm_usage, ProviderUsage):
            raise ValueError("llm_usage must be ProviderUsage")
        _require_non_negative(self.embedding_calls, field_name="embedding_calls")
        _require_non_negative_float(
            self.embedding_elapsed_sec,
            field_name="embedding_elapsed_sec",
        )
        _require_non_negative_float(self.elapsed_sec, field_name="elapsed_sec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "written_ids": list(self.written_ids),
            "llm_usage": self.llm_usage.to_dict(),
            "embedding_calls": self.embedding_calls,
            "embedding_elapsed_sec": self.embedding_elapsed_sec,
            "elapsed_sec": self.elapsed_sec,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Mem0WriteResult":
        return cls(
            written_ids=_sequence_of_strings(
                data["written_ids"], field_name="written_ids"
            ),
            llm_usage=ProviderUsage.from_dict(
                _required_mapping(data["llm_usage"], field_name="llm_usage")
            ),
            embedding_calls=_required_int(
                data["embedding_calls"], field_name="embedding_calls"
            ),
            embedding_elapsed_sec=_required_float(
                data["embedding_elapsed_sec"],
                field_name="embedding_elapsed_sec",
            ),
            elapsed_sec=_required_float(data["elapsed_sec"], field_name="elapsed_sec"),
        )


@dataclass(frozen=True)
class BackendFinalizeResult:
    status: str
    written_ids: tuple[str, ...] = ()
    llm_usage: ProviderUsage = ProviderUsage()
    usage_by_role: Mapping[str, ProviderUsage] = field(default_factory=dict)
    embedding_calls: int = 0
    embedding_elapsed_sec: float = 0.0
    elapsed_sec: float = 0.0
    error: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.status, field_name="status")
        object.__setattr__(
            self,
            "written_ids",
            _string_tuple(self.written_ids, field_name="written_ids"),
        )
        if not isinstance(self.llm_usage, ProviderUsage):
            raise ValueError("llm_usage must be ProviderUsage")
        normalized_usage: dict[str, ProviderUsage] = {}
        for role, usage in self.usage_by_role.items():
            _require_non_empty(role, field_name="usage role")
            if not isinstance(usage, ProviderUsage):
                raise ValueError("usage_by_role values must be ProviderUsage")
            normalized_usage[str(role)] = usage
        object.__setattr__(self, "usage_by_role", normalized_usage)
        _require_non_negative(self.embedding_calls, field_name="embedding_calls")
        _require_non_negative_float(
            self.embedding_elapsed_sec,
            field_name="embedding_elapsed_sec",
        )
        _require_non_negative_float(self.elapsed_sec, field_name="elapsed_sec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "written_ids": list(self.written_ids),
            "llm_usage": self.llm_usage.to_dict(),
            "usage_by_role": {
                role: usage.to_dict()
                for role, usage in sorted(self.usage_by_role.items())
            },
            "embedding_calls": self.embedding_calls,
            "embedding_elapsed_sec": self.embedding_elapsed_sec,
            "elapsed_sec": self.elapsed_sec,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendFinalizeResult":
        return cls(
            status=str(data["status"]),
            written_ids=_sequence_of_strings(
                data.get("written_ids", ()), field_name="written_ids"
            ),
            llm_usage=ProviderUsage.from_dict(
                _required_mapping(data.get("llm_usage", {}), field_name="llm_usage")
            ),
            usage_by_role={
                str(role): ProviderUsage.from_dict(
                    _required_mapping(usage, field_name=f"usage_by_role[{role}]")
                )
                for role, usage in _required_mapping(
                    data.get("usage_by_role", {}),
                    field_name="usage_by_role",
                ).items()
            },
            embedding_calls=_required_int(
                data.get("embedding_calls", 0), field_name="embedding_calls"
            ),
            embedding_elapsed_sec=_required_float(
                data.get("embedding_elapsed_sec", 0.0),
                field_name="embedding_elapsed_sec",
            ),
            elapsed_sec=_required_float(
                data.get("elapsed_sec", 0.0), field_name="elapsed_sec"
            ),
            error=str(data.get("error", "")),
        )


@dataclass(frozen=True)
class MemoryRepositorySnapshot:
    revision: str
    entry_count: int
    repository_bytes: int
    tier_counts: Mapping[str, int]
    repository_path: str = ""

    def __post_init__(self) -> None:
        _require_non_negative(self.entry_count, field_name="entry_count")
        _require_non_negative(self.repository_bytes, field_name="repository_bytes")
        normalized: dict[str, int] = {}
        for tier, count in self.tier_counts.items():
            _require_non_empty(tier, field_name="tier")
            _require_non_negative(count, field_name=f"tier_counts[{tier}]")
            normalized[str(tier)] = count
        if sum(normalized.values()) != self.entry_count:
            raise ValueError("tier_counts must sum to entry_count")
        object.__setattr__(self, "tier_counts", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "entry_count": self.entry_count,
            "repository_bytes": self.repository_bytes,
            "tier_counts": dict(sorted(self.tier_counts.items())),
            "repository_path": self.repository_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryRepositorySnapshot":
        raw_counts = _required_mapping(data["tier_counts"], field_name="tier_counts")
        return cls(
            revision=str(data["revision"]),
            entry_count=_required_int(data["entry_count"], field_name="entry_count"),
            repository_bytes=_required_int(
                data["repository_bytes"], field_name="repository_bytes"
            ),
            tier_counts={
                tier: _required_int(count, field_name=f"tier_counts[{tier}]")
                for tier, count in raw_counts.items()
            },
            repository_path=str(data.get("repository_path", "")),
        )


@dataclass(frozen=True)
class MemoryBenchmarkTaskResult:
    run_id: str
    seed: int
    actor_sampling_seed_supported: bool
    actor_sampling_seed_effective: int | None
    arm: str
    benchmark: str
    subset: str
    task_id: str
    order_index: int
    task_content_hash: str
    actor_identity_hash: str
    tools_hash: str
    evaluator_hash: str
    resolved: bool
    reward: float
    outcome_finalized: bool
    failure_type: str
    agent_steps: int
    tool_calls: int
    actor_prompt_tokens: int | None
    actor_completion_tokens: int | None
    actor_total_tokens: int | None
    memory_prompt_tokens: int | None
    memory_completion_tokens: int | None
    memory_total_tokens: int | None
    memory_tokens_by_role: Mapping[str, ProviderUsage]
    actor_usage_available: bool
    memory_usage_available: bool
    memory_usage_unavailable_reason: str
    system_total_tokens: int | None
    embedding_calls: int
    embedding_elapsed_sec: float
    elapsed_sec: float
    memory: Mapping[str, Any]
    trace_path: str
    action_log_path: str
    official_result_path: str
    public_episode_path: str
    protocol_hash: str
    backend_config_hash: str

    def __post_init__(self) -> None:
        for field_name in ("run_id", "arm", "benchmark", "subset", "task_id"):
            _require_non_empty(getattr(self, field_name), field_name=field_name)
        _require_non_negative(self.seed, field_name="seed")
        for field_name in (
            "task_content_hash",
            "actor_identity_hash",
            "tools_hash",
            "evaluator_hash",
            "protocol_hash",
            "backend_config_hash",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.actor_sampling_seed_supported, bool):
            raise ValueError("actor_sampling_seed_supported must be a bool")
        if self.actor_sampling_seed_effective is not None:
            _require_non_negative(
                self.actor_sampling_seed_effective,
                field_name="actor_sampling_seed_effective",
            )
        if not self.actor_sampling_seed_supported and self.actor_sampling_seed_effective is not None:
            raise ValueError("unsupported actor sampling seed must not be marked effective")
        if self.order_index < 1:
            raise ValueError("order_index must be at least 1")
        if not isinstance(self.resolved, bool) or not isinstance(self.outcome_finalized, bool):
            raise ValueError("resolved and outcome_finalized must be bool values")
        _require_reward(self.reward)
        for field_name in ("agent_steps", "tool_calls", "embedding_calls"):
            _require_non_negative(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "actor_prompt_tokens",
            "actor_completion_tokens",
            "actor_total_tokens",
            "memory_prompt_tokens",
            "memory_completion_tokens",
            "memory_total_tokens",
            "system_total_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_negative(value, field_name=field_name)
        _require_non_negative_float(
            self.embedding_elapsed_sec,
            field_name="embedding_elapsed_sec",
        )
        _require_non_negative_float(self.elapsed_sec, field_name="elapsed_sec")
        normalized_roles: dict[str, ProviderUsage] = {}
        for role, usage in self.memory_tokens_by_role.items():
            _require_non_empty(role, field_name="memory role")
            if not isinstance(usage, ProviderUsage):
                raise ValueError("memory_tokens_by_role values must be ProviderUsage")
            normalized_roles[str(role)] = usage
        object.__setattr__(self, "memory_tokens_by_role", normalized_roles)
        object.__setattr__(self, "memory", dict(self.memory))
        if self.actor_usage_available and self.actor_total_tokens is None:
            raise ValueError("available actor usage requires actor_total_tokens")
        if self.memory_usage_available and self.memory_total_tokens is None:
            raise ValueError("available memory usage requires memory_total_tokens")
        if not self.memory_usage_available and not self.memory_usage_unavailable_reason.strip():
            raise ValueError("unavailable memory usage requires a reason")
        if self.memory_usage_available and self.memory_usage_unavailable_reason:
            raise ValueError("available memory usage must not include an unavailable reason")
        if self.actor_usage_available and self.memory_usage_available:
            expected = self.actor_total_tokens + self.memory_total_tokens  # type: ignore[operator]
            if self.system_total_tokens != expected:
                raise ValueError("system_total_tokens must equal actor plus memory tokens")
        elif self.system_total_tokens is not None:
            raise ValueError("system_total_tokens must be null when usage is unavailable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "actor_sampling_seed_supported": self.actor_sampling_seed_supported,
            "actor_sampling_seed_effective": self.actor_sampling_seed_effective,
            "arm": self.arm,
            "benchmark": self.benchmark,
            "subset": self.subset,
            "task_id": self.task_id,
            "order_index": self.order_index,
            "task_content_hash": self.task_content_hash,
            "actor_identity_hash": self.actor_identity_hash,
            "tools_hash": self.tools_hash,
            "evaluator_hash": self.evaluator_hash,
            "resolved": self.resolved,
            "reward": self.reward,
            "outcome_finalized": self.outcome_finalized,
            "failure_type": self.failure_type,
            "agent_steps": self.agent_steps,
            "tool_calls": self.tool_calls,
            "actor_prompt_tokens": self.actor_prompt_tokens,
            "actor_completion_tokens": self.actor_completion_tokens,
            "actor_total_tokens": self.actor_total_tokens,
            "memory_prompt_tokens": self.memory_prompt_tokens,
            "memory_completion_tokens": self.memory_completion_tokens,
            "memory_total_tokens": self.memory_total_tokens,
            "memory_tokens_by_role": {
                role: usage.to_dict()
                for role, usage in sorted(self.memory_tokens_by_role.items())
            },
            "actor_usage_available": self.actor_usage_available,
            "memory_usage_available": self.memory_usage_available,
            "memory_usage_unavailable_reason": self.memory_usage_unavailable_reason,
            "system_total_tokens": self.system_total_tokens,
            "embedding_calls": self.embedding_calls,
            "embedding_elapsed_sec": self.embedding_elapsed_sec,
            "elapsed_sec": self.elapsed_sec,
            "memory": dict(self.memory),
            "trace_path": self.trace_path,
            "action_log_path": self.action_log_path,
            "official_result_path": self.official_result_path,
            "public_episode_path": self.public_episode_path,
            "protocol_hash": self.protocol_hash,
            "backend_config_hash": self.backend_config_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryBenchmarkTaskResult":
        if data.get("schema_version") != TASK_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported memory benchmark task result schema")
        raw_roles = _required_mapping(
            data["memory_tokens_by_role"], field_name="memory_tokens_by_role"
        )
        return cls(
            run_id=str(data["run_id"]),
            seed=_required_int(data["seed"], field_name="seed"),
            actor_sampling_seed_supported=_required_bool(
                data["actor_sampling_seed_supported"],
                field_name="actor_sampling_seed_supported",
            ),
            actor_sampling_seed_effective=_optional_int(
                data.get("actor_sampling_seed_effective"),
                field_name="actor_sampling_seed_effective",
            ),
            arm=str(data["arm"]),
            benchmark=str(data["benchmark"]),
            subset=str(data["subset"]),
            task_id=str(data["task_id"]),
            order_index=_required_int(data["order_index"], field_name="order_index"),
            task_content_hash=str(data["task_content_hash"]),
            actor_identity_hash=str(data["actor_identity_hash"]),
            tools_hash=str(data["tools_hash"]),
            evaluator_hash=str(data["evaluator_hash"]),
            resolved=_required_bool(data["resolved"], field_name="resolved"),
            reward=_required_float(data["reward"], field_name="reward"),
            outcome_finalized=_required_bool(
                data["outcome_finalized"], field_name="outcome_finalized"
            ),
            failure_type=str(data["failure_type"]),
            agent_steps=_required_int(data["agent_steps"], field_name="agent_steps"),
            tool_calls=_required_int(data["tool_calls"], field_name="tool_calls"),
            actor_prompt_tokens=_optional_int(
                data.get("actor_prompt_tokens"), field_name="actor_prompt_tokens"
            ),
            actor_completion_tokens=_optional_int(
                data.get("actor_completion_tokens"),
                field_name="actor_completion_tokens",
            ),
            actor_total_tokens=_optional_int(
                data.get("actor_total_tokens"), field_name="actor_total_tokens"
            ),
            memory_prompt_tokens=_optional_int(
                data.get("memory_prompt_tokens"), field_name="memory_prompt_tokens"
            ),
            memory_completion_tokens=_optional_int(
                data.get("memory_completion_tokens"),
                field_name="memory_completion_tokens",
            ),
            memory_total_tokens=_optional_int(
                data.get("memory_total_tokens"), field_name="memory_total_tokens"
            ),
            memory_tokens_by_role={
                role: ProviderUsage.from_dict(
                    _required_mapping(usage, field_name=f"memory role {role}")
                )
                for role, usage in raw_roles.items()
            },
            actor_usage_available=_required_bool(
                data["actor_usage_available"], field_name="actor_usage_available"
            ),
            memory_usage_available=_required_bool(
                data["memory_usage_available"], field_name="memory_usage_available"
            ),
            memory_usage_unavailable_reason=str(
                data["memory_usage_unavailable_reason"]
            ),
            system_total_tokens=_optional_int(
                data.get("system_total_tokens"), field_name="system_total_tokens"
            ),
            embedding_calls=_required_int(
                data["embedding_calls"], field_name="embedding_calls"
            ),
            embedding_elapsed_sec=_required_float(
                data["embedding_elapsed_sec"], field_name="embedding_elapsed_sec"
            ),
            elapsed_sec=_required_float(data["elapsed_sec"], field_name="elapsed_sec"),
            memory=_required_mapping(data["memory"], field_name="memory"),
            trace_path=str(data["trace_path"]),
            action_log_path=str(data["action_log_path"]),
            official_result_path=str(data["official_result_path"]),
            public_episode_path=str(data["public_episode_path"]),
            protocol_hash=str(data["protocol_hash"]),
            backend_config_hash=str(data["backend_config_hash"]),
        )


def _require_non_empty(value: Any, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_negative(value: Any, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_non_negative_float(value: Any, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    if not isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")


def _require_reward(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reward must be a finite number in [0.0, 1.0]")
    if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("reward must be a finite number in [0.0, 1.0]")


def _required_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, field_name=field_name)


def _required_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _required_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _required_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return normalized


def _sequence_of_strings(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of strings")
    return _string_tuple(value, field_name=field_name)


def _command_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    command = _string_tuple(values, field_name=field_name)
    if not command:
        raise ValueError(f"{field_name} must not be empty")
    return command


def _string_mapping(value: Mapping[str, str], *, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(key, field_name=f"{field_name} key")
        if not isinstance(item, str):
            raise ValueError(f"{field_name} values must be strings")
        normalized[str(key)] = item
    return normalized


def _reject_hidden_answer_fields(value: Any, *, path: str = "environment_spec") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.casefold().replace("-", "_")
            if normalized in _HIDDEN_ANSWER_FIELDS:
                raise ValueError(
                    f"{path} cannot contain hidden answer field {key_text!r}"
                )
            _reject_hidden_answer_fields(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_hidden_answer_fields(item, path=f"{path}[{index}]")


__all__ = [
    "BackendFinalizeResult",
    "BenchmarkTask",
    "ExternalMemoryItem",
    "MemoryBenchmarkTaskResult",
    "MemoryContextSelection",
    "MemoryRepositorySnapshot",
    "Mem0SearchResult",
    "Mem0WriteResult",
    "OFFICIAL_RESULT_SCHEMA_VERSION",
    "OfficialEvaluatorResult",
    "PreparedBenchmarkTask",
    "ProviderUsage",
    "PublicEpisode",
    "TASK_RESULT_SCHEMA_VERSION",
    "load_official_evaluator_result",
    "write_official_result_atomic",
]
