"""Legacy retrieve/select and writer Evolver runtime strategy."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.memory.evolver import (
    ExperienceSelector,
    ExperienceWriteRequest,
    ExperienceWriteResult,
    ExperienceWriter,
    MemoryWriterDatasetLogger,
    SelectionResult,
    build_write_steps_from_tool_history,
    proposal_tier_counts,
    selection_candidate_summary,
    selection_tier_counts,
    writer_policy_for_result,
)
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTier,
)
from my_agent.memory.experience.retrieval.contracts import ExperienceRetriever
from my_agent.memory.experience.retrieval.lexical import ExperienceRetrievalMetrics
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.experience.serialization import experience_payload_to_dict
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.policy.identity import PolicyIdentity

WRITER_METADATA_STRING_CHARS = 1_000
WRITER_DATASET_TASK_CHARS = 2_000
WRITER_DATASET_CONTENT_CHARS = 1_200
WRITER_DATASET_OUTPUT_CHARS = 1_000
WRITER_DATASET_FORBIDDEN_MARKERS = (
    "hidden_test_output",
    "hidden_ok",
    "official_solution",
    "ground_truth",
    "expected_patch",
    "private_key",
    "api_key",
    "apikey",
    "access_key",
    "bearer",
    "cookie",
    "credential",
    "github_pat_",
    "ghp_",
    "glpat-",
    "password",
    "secret",
    "token",
)
WRITER_DATASET_SECRET_PREFIX_RE = re.compile(
    r"(?i)(?:github_pat_|ghp_|glpat-|sk-[A-Za-z0-9_-]{16,}|xox[baprs]-|AKIA|ASIA|AIza|ya29\.|eyJ[A-Za-z0-9_-]{8,})"
)


class LegacyEvolverRuntime:
    def __init__(
        self,
        *,
        mode: str,
        config: AgentConfig,
        context_profile: ContextProfile,
        store: ExperienceStore,
        project_key: str,
        retriever: ExperienceRetriever,
        selector: ExperienceSelector,
        writer: ExperienceWriter,
        selector_factory: Callable[[], ExperienceSelector],
        writer_factory: Callable[[], ExperienceWriter],
        write_enabled: bool,
        trace_sink: Any | None = None,
    ) -> None:
        self.mode = mode
        self.config = config
        self.context_profile = context_profile
        self.store = store
        self.project_key = project_key
        self.retriever = retriever
        self.selector = selector
        self.writer = writer
        self._selector_factory = selector_factory
        self._writer_factory = writer_factory
        self.write_enabled = write_enabled
        self._trace_sink = trace_sink
        self._coordinator: Any | None = None
        self._embedding_retriever: Any | None = None
        self.last_selection: SelectionResult | None = None

    @property
    def coordinator(self) -> Any | None:
        return self._coordinator

    @coordinator.setter
    def coordinator(self, value: Any | None) -> None:
        self._coordinator = value

    @property
    def candidate_retriever(self) -> ExperienceRetriever:
        return self.retriever

    @property
    def experience_retriever(self) -> ExperienceRetriever:
        return self.retriever

    @experience_retriever.setter
    def experience_retriever(self, value: ExperienceRetriever) -> None:
        self.retriever = value

    @property
    def embedding_retriever(self) -> Any | None:
        return self._embedding_retriever

    @embedding_retriever.setter
    def embedding_retriever(self, value: Any | None) -> None:
        self._embedding_retriever = value

    @property
    def selector(self) -> ExperienceSelector:
        return self._selector

    @selector.setter
    def selector(self, value: ExperienceSelector) -> None:
        self._selector = value

    @property
    def writer(self) -> ExperienceWriter:
        return self._writer

    @writer.setter
    def writer(self, value: ExperienceWriter) -> None:
        self._writer = value

    def set_trace_sink(self, trace_sink: Any | None) -> None:
        self._trace_sink = trace_sink

    def build_context(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext[ExperienceMemory]:
        resolved_tokens = (
            max_tokens
            if max_tokens is not None
            else self.context_profile.memory_context_tokens
        )
        resolved_top_k = (
            top_k_per_tier
            if top_k_per_tier is not None
            else self.config.memory_evolver_top_k_per_tier
        )
        resolved_top_k = max(1, int(resolved_top_k))
        visible_count = len(self.store.all(project_key=self.project_key))
        if (
            resolved_tokens <= 0
            or visible_count < self.config.memory_evolver_min_experience_entries
        ):
            index = self.store.index_snapshot()
            visible_entries = self.store.all(project_key=self.project_key)
            tier_metrics = {
                tier.value: {
                    "visible_count": sum(
                        1 for entry in visible_entries if entry.tier == tier
                    ),
                    "indexed_count": sum(
                        1
                        for entry in visible_entries
                        if entry.tier == tier and not entry.invalidated
                    ),
                    "posting_candidate_count": 0,
                    "scored_count": 0,
                    "matched_count": 0,
                    "returned_count": 0,
                }
                for tier in ExperienceTier
            }
            self.retriever.last_metrics = ExperienceRetrievalMetrics(
                repository_revision=index.revision,
                visible_count=len(visible_entries),
                indexed_count=sum(
                    1 for entry in visible_entries if not entry.invalidated
                ),
                per_tier=tier_metrics,
            )
            result = SelectionResult(
                candidates=(),
                selected=(),
                context=MemoryContext(injected_text="", hits=[], estimated_tokens=0),
                policy="rule_tier_weighted_v1",
                estimated_tokens=0,
                metadata={
                    "query_chars": len(query),
                    "candidate_count": 0,
                    "selected_count": 0,
                    "candidate_tier_counts": {},
                    "selected_tier_counts": {},
                    "insufficient_experience_entries": (
                        visible_count
                        < self.config.memory_evolver_min_experience_entries
                    ),
                },
            )
            self.last_selection = result
            self._trace_selection(
                query=query,
                result=result,
                candidates=[],
                max_tokens=resolved_tokens,
                top_k_per_tier=resolved_top_k,
                max_items=max_items,
                visible_experience_count=visible_count,
                insufficient_experience_entries=(
                    visible_count < self.config.memory_evolver_min_experience_entries
                ),
            )
            return result.context

        candidates = self.retrieve_candidates(
            query,
            top_k_per_tier=resolved_top_k,
        )
        try:
            result = self.selector.select(
                query=query,
                hits=candidates,
                max_tokens=resolved_tokens,
                max_items=max_items,
            )
        except Exception as exc:  # noqa: BLE001 - selector failures are fail-closed
            error = f"{type(exc).__name__}: {exc}"
            result = SelectionResult(
                candidates=(),
                selected=(),
                context=MemoryContext(injected_text="", hits=[], estimated_tokens=0),
                policy="rule_tier_weighted_v1",
                estimated_tokens=0,
                metadata={
                    "query_chars": len(query),
                    "candidate_count": 0,
                    "selected_count": 0,
                    "candidate_tier_counts": {},
                    "selected_tier_counts": {},
                    "fallback": True,
                    "error": error,
                },
            )
            self.last_selection = result
            self._trace("memory.evolver_selection_failed", {
                "query_chars": len(query),
                "candidate_count": len(candidates),
                "error": error,
                "fallback": "empty_context",
                "mode": self.mode,
                "memory_project_key": self.project_key,
            })
            self._trace_selection(
                query=query,
                result=result,
                candidates=candidates,
                max_tokens=resolved_tokens,
                top_k_per_tier=resolved_top_k,
                max_items=max_items,
                visible_experience_count=visible_count,
                insufficient_experience_entries=False,
                fallback=True,
            )
            return result.context
        self.last_selection = result
        self._trace_selection(
            query=query,
            result=result,
            candidates=candidates,
            max_tokens=resolved_tokens,
            top_k_per_tier=resolved_top_k,
            max_items=max_items,
            visible_experience_count=visible_count,
            insufficient_experience_entries=False,
        )
        return result.context

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k_per_tier: int | None = None,
    ) -> list[RetrievalHit[ExperienceMemory]]:
        resolved_top_k = (
            top_k_per_tier
            if top_k_per_tier is not None
            else self.config.memory_evolver_top_k_per_tier
        )
        resolved_top_k = max(1, int(resolved_top_k))
        return list(self.retriever.retrieve_candidates(
            query,
            store=self.store,
            project_key=self.project_key,
            top_k_per_tier=resolved_top_k,
        ))

    def begin_task(self, **kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("formal evolver task session is unavailable")

    def finalize_task(self, episode: Any, outcome: Any) -> None:
        del episode, outcome
        return None

    def write_legacy_run(
        self,
        *,
        save_experience: Any,
        task: str,
        run_id: str,
        trace_path: str | Path | None = None,
        stop_reason: str = "",
        final_answer: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        outcome: str = "unknown",
        outcome_source: str = "runtime",
        source_task: str = "",
        stream_id: str = "",
        task_type: str = "",
        memory_mode: str = "",
    ) -> ExperienceWriteResult:
        if not self.write_enabled:
            return ExperienceWriteResult()

        steps = build_write_steps_from_tool_history(
            tool_history,
            max_output_chars=min(
                1_000,
                max(0, self.config.memory_evolver_writer_max_input_chars),
            ),
        )
        resolved_source_task = str(source_task or "").strip() or _task_ref(task)
        selected_ids = _selection_selected_ids(self.last_selection)
        candidate_ids = _selection_candidate_ids(self.last_selection)
        request = ExperienceWriteRequest(
            task=str(task or ""),
            run_id=str(run_id or ""),
            trace_path=Path(trace_path) if trace_path is not None else None,
            stop_reason=str(stop_reason or ""),
            outcome=str(outcome or "unknown"),
            outcome_source=str(outcome_source or "runtime"),
            final_answer=str(final_answer or ""),
            selected_memory_ids=selected_ids,
            candidate_memory_ids=candidate_ids,
            steps=steps,
            source_task=resolved_source_task,
            stream_id=str(stream_id or ""),
            task_type=str(task_type or ""),
            project_key=self.project_key,
            memory_mode=str(memory_mode or ""),
        )
        context_payload = _writer_dataset_context_payload(request)
        try:
            self._trace("memory.evolver_writer_started", {
                **context_payload,
                "mode": self.config.memory_evolver_writer_mode,
                "task_chars": len(request.task),
                "tool_steps": len(request.steps),
                "outcome": request.outcome,
                "outcome_source": request.outcome_source,
                "selected_count": len(request.selected_memory_ids),
                "candidate_count": len(request.candidate_memory_ids),
            })
            proposed = self.writer.propose(
                request,
                mode=self.config.memory_evolver_writer_mode,
            )
            self._trace("memory.evolver_writer_proposed", {
                "proposal_count": len(proposed.proposals),
                "tiers": proposal_tier_counts(proposed.proposals),
                "llm_used": proposed.llm_used,
                "fallback_used": proposed.fallback_used,
                "rejected_count": len(proposed.rejected),
                "rejected_reasons": _rejected_reason_counts(proposed.rejected),
            })
            if not proposed.proposals:
                self._trace("memory.evolver_writer_skipped", {
                    **context_payload,
                    "reason": "no_valid_proposals",
                    "outcome": request.outcome,
                    "tool_steps": len(request.steps),
                })
                self._append_writer_dataset(request, proposed)
                return proposed

            saved: list[ExperienceMemory] = []
            duplicate_ids: list[str] = []
            safe_source_task = _safe_dataset_join_text(request.source_task)
            writer_policy = writer_policy_for_result(
                llm_used=proposed.llm_used,
                fallback_used=proposed.fallback_used,
            )
            for proposal in proposed.proposals:
                entry, created = save_experience(
                    tier=proposal.tier,
                    content=proposal.content,
                    payload=proposal.payload,
                    source_task=safe_source_task,
                    created_by=ExperienceCreatedBy.WRITER,
                    run_id=request.run_id,
                    stream_id=request.stream_id,
                    writer_confidence=proposal.confidence,
                )
                if created:
                    saved.append(entry)
                else:
                    duplicate_ids.append(entry.id)

            result = ExperienceWriteResult(
                proposals=proposed.proposals,
                saved=tuple(saved),
                duplicate_ids=tuple(duplicate_ids),
                rejected=proposed.rejected,
                llm_used=proposed.llm_used,
                fallback_used=proposed.fallback_used,
            )
            self._trace("memory.evolver_writer_saved", {
                **context_payload,
                "saved_count": len(saved),
                "duplicate_count": len(duplicate_ids),
                "saved_ids": [entry.id for entry in saved],
                "saved_records": _saved_records(saved),
                "tiers": _saved_tier_counts(saved),
                "writer_policy": writer_policy,
            })
            self._append_writer_dataset(request, result)
            return result
        except Exception as exc:  # noqa: BLE001 - writer must not affect agent loop
            error = f"{type(exc).__name__}: {exc}"
            self._trace("memory.evolver_writer_failed", {
                **context_payload,
                "error": _safe_error_text(error),
                "phase": "unknown",
                "fallback_attempted": True,
            })
            return ExperienceWriteResult(error=error)

    def require_formal_binding(
        self,
        *,
        config: AgentConfig,
        policy_identity: PolicyIdentity,
        repo_path: Path | None,
        manager_llm: Any | None,
        manager_store: Any,
        manager_project_key: str,
    ) -> None:
        del config, policy_identity, repo_path, manager_llm, manager_store, manager_project_key
        raise ValueError("formal OPD runtime cannot use a non-formal MemoryManager")

    def fork(self) -> "LegacyEvolverRuntime":
        return LegacyEvolverRuntime(
            mode=self.mode,
            config=self.config,
            context_profile=self.context_profile,
            store=self.store,
            project_key=self.project_key,
            retriever=self.retriever.fork(),
            selector=self._selector_factory(),
            writer=self._writer_factory(),
            selector_factory=self._selector_factory,
            writer_factory=self._writer_factory,
            write_enabled=self.write_enabled,
            trace_sink=self._trace_sink,
        )

    def _trace_selection(
        self,
        *,
        query: str,
        result: SelectionResult,
        candidates: list[RetrievalHit[ExperienceMemory]],
        max_tokens: int,
        top_k_per_tier: int,
        max_items: int | None,
        visible_experience_count: int,
        insufficient_experience_entries: bool,
        fallback: bool = False,
    ) -> None:
        retrieved_tiers = _hit_tier_counts(candidates)
        candidate_tiers = selection_tier_counts(result.candidates)
        selected_candidates = [item.candidate for item in result.selected]
        selected_tiers = selection_tier_counts(selected_candidates)
        candidate_payload = {
            "query_chars": len(query),
            "candidate_count": len(result.candidates),
            "top_k_per_tier": top_k_per_tier,
            "tiers": candidate_tiers,
            "retrieved_candidate_count": len(candidates),
            "retrieved_tiers": retrieved_tiers,
            "candidate_ids": [candidate.id for candidate in result.candidates],
            "candidate_summaries": [
                selection_candidate_summary(candidate)
                for candidate in result.candidates
            ],
            "selection_policy": result.policy,
            "mode": self.mode,
            "insufficient_experience_entries": insufficient_experience_entries,
            "visible_experience_count": visible_experience_count,
            "memory_project_key": self.project_key,
            **self.retriever.last_metrics.to_trace_payload(),
        }
        selected_payload = {
            "selected_count": len(result.selected),
            "selected_ids": [item.candidate.id for item in result.selected],
            "omitted_ids": list(result.omitted_ids),
            "tiers": selected_tiers,
            "estimated_tokens": result.context.estimated_tokens,
            "max_tokens": max_tokens,
            "max_items": (
                max_items
                if max_items is not None
                else self.config.memory_evolver_selected_max_items
            ),
            "selection_policy": result.policy,
            "selection_reasons": [
                {
                    "id": item.candidate.id,
                    "reason": (
                        f"selected: score={item.candidate.selection_score:.2f} "
                        f"tier={item.candidate.tier.value} rank={item.rank}"
                    ),
                }
                for item in result.selected
            ],
            "fallback": fallback,
            "memory_project_key": self.project_key,
        }
        self._trace("memory.evolver_candidates", candidate_payload)
        self._trace("memory.evolver_selected", selected_payload)
        self._trace("memory.retrieved", {
            "query_chars": len(query),
            "hits": len(result.context.hits),
            "tokens": result.context.estimated_tokens,
            "include_short_term": False,
            "mode": self.mode,
            "selection_policy": result.policy,
            **self.retriever.last_metrics.to_trace_payload(),
        })

    def _append_writer_dataset(
        self,
        request: ExperienceWriteRequest,
        result: ExperienceWriteResult,
    ) -> None:
        dataset_path = self.config.memory_evolver_writer_dataset_path
        if dataset_path is None:
            return
        try:
            MemoryWriterDatasetLogger(dataset_path).append(
                _writer_dataset_record(
                    request=request,
                    result=result,
                    selection=self.last_selection,
                )
            )
        except Exception as exc:  # noqa: BLE001 - logging must not affect writes
            self._trace("memory.evolver_writer_failed", {
                **_writer_dataset_context_payload(request),
                "error": _safe_error_text(f"{type(exc).__name__}: {exc}"),
                "phase": "dataset",
                "fallback_attempted": result.fallback_used,
            })

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(event, payload)
        except Exception:
            pass


def _task_ref(task: str) -> str:
    return f"task_ref_{hashlib.sha256(str(task or '').encode('utf-8')).hexdigest()[:12]}"


def _selection_candidate_ids(selection: SelectionResult | None) -> tuple[str, ...]:
    if selection is None:
        return ()
    return tuple(candidate.id for candidate in selection.candidates)


def _selection_selected_ids(selection: SelectionResult | None) -> tuple[str, ...]:
    if selection is None:
        return ()
    return tuple(item.candidate.id for item in selection.selected)


def _writer_dataset_context_payload(request: ExperienceWriteRequest) -> dict[str, Any]:
    return {
        "source_task": _safe_dataset_join_text(request.source_task),
        "stream_id": _safe_dataset_join_text(request.stream_id),
        "task_type": _safe_dataset_join_text(request.task_type),
        "memory_project_key": _safe_dataset_join_text(request.project_key),
        "memory_mode": _safe_dataset_join_text(request.memory_mode),
    }


def _writer_dataset_record(
    *,
    request: ExperienceWriteRequest,
    result: ExperienceWriteResult,
    selection: SelectionResult | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "task": _safe_dataset_text(request.task, WRITER_DATASET_TASK_CHARS),
        "run_id": request.run_id,
        "trace_path": _safe_dataset_join_text(str(request.trace_path or "")),
        "source_task": _safe_dataset_join_text(request.source_task),
        "task_id": _safe_dataset_join_text(request.source_task),
        "task_type": _safe_dataset_join_text(request.task_type),
        "stream_id": _safe_dataset_join_text(request.stream_id),
        "memory_project_key": _safe_dataset_join_text(request.project_key),
        "memory_mode": _safe_dataset_join_text(request.memory_mode),
        "outcome": request.outcome,
        "outcome_source": request.outcome_source,
        "stop_reason": request.stop_reason,
        "selected_memory_ids": list(request.selected_memory_ids),
        "candidate_memory_ids": list(request.candidate_memory_ids),
        "selected_memory_ids_by_tier": _selection_selected_ids_by_tier(selection),
        "candidate_memory_ids_by_tier": _selection_candidate_ids_by_tier(selection),
        "steps": [_writer_dataset_step(step) for step in request.steps],
        "proposals": [
            _writer_dataset_proposal(proposal) for proposal in result.proposals
        ],
        "saved_ids": [entry.id for entry in result.saved],
        "saved_records": _saved_records(list(result.saved)),
        "duplicate_ids": list(result.duplicate_ids),
        "rejected": [_safe_dataset_value(dict(item)) for item in result.rejected],
        "llm_used": result.llm_used,
        "fallback_used": result.fallback_used,
    }
    if result.error:
        record["error"] = _safe_error_text(result.error)
    return record


def _writer_dataset_step(step: Any) -> dict[str, Any]:
    output = str(getattr(step, "output", "") or "")
    redacted = _unsafe_dataset_text(output)
    return {
        "step_num": int(getattr(step, "step_num", 0) or 0),
        "tool": str(getattr(step, "tool", "") or ""),
        "arguments": _safe_dataset_value(dict(getattr(step, "arguments", {}) or {})),
        "ok": bool(getattr(step, "ok", False)),
        "output": ""
        if redacted
        else _safe_dataset_text(output, WRITER_DATASET_OUTPUT_CHARS),
        "output_redacted": redacted,
        "blocked": bool(getattr(step, "blocked", False)),
        "error_code": str(getattr(step, "error_code", "") or ""),
    }


def _writer_dataset_proposal(proposal: Any) -> dict[str, Any]:
    tier = getattr(proposal, "tier", "")
    payload = getattr(proposal, "payload", None)
    try:
        serialized_payload = experience_payload_to_dict(payload)
    except (TypeError, ValueError):
        serialized_payload = {}
    return {
        "tier": tier.value if isinstance(tier, ExperienceTier) else str(tier),
        "content": _safe_dataset_text(
            str(getattr(proposal, "content", "") or ""),
            WRITER_DATASET_CONTENT_CHARS,
        ),
        "payload": _safe_dataset_value(serialized_payload),
        "confidence": float(getattr(proposal, "confidence", 0.0) or 0.0),
        "reason": _safe_dataset_text(
            str(getattr(proposal, "reason", "") or ""),
            WRITER_METADATA_STRING_CHARS,
        ),
    }


def _safe_dataset_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe_items: dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            if _unsafe_dataset_text(raw_key):
                safe_items[_safe_dataset_redaction(raw_key)] = ""
                continue
            safe_items[_safe_dataset_text(
                raw_key,
                WRITER_METADATA_STRING_CHARS,
            )] = _safe_dataset_value(item)
        return safe_items
    if isinstance(value, list):
        return [_safe_dataset_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_dataset_value(item) for item in value]
    if isinstance(value, str):
        return _safe_dataset_text(value, WRITER_METADATA_STRING_CHARS)
    return value


def _safe_dataset_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if _unsafe_dataset_text(text):
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3].rstrip() + "..."


def _safe_dataset_join_text(value: str) -> str:
    text = str(value or "")
    if _unsafe_dataset_text(text):
        return _safe_dataset_redaction(text)
    return _safe_dataset_text(text, WRITER_METADATA_STRING_CHARS)


def _safe_error_text(value: str) -> str:
    text = str(value or "")
    if _unsafe_dataset_text(text):
        return _safe_dataset_redaction(text)
    return _safe_dataset_text(text, WRITER_METADATA_STRING_CHARS)


def _safe_dataset_redaction(value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:12]
    return f"redacted_{digest}"


def _unsafe_dataset_text(value: str) -> bool:
    text = str(value or "")
    lower = text.casefold()
    if WRITER_DATASET_SECRET_PREFIX_RE.search(text):
        return True
    return any(marker in lower for marker in WRITER_DATASET_FORBIDDEN_MARKERS)


def _selection_candidate_ids_by_tier(
    selection: SelectionResult | None,
) -> dict[str, list[str]]:
    if selection is None:
        return {}
    grouped: dict[str, list[str]] = {}
    for candidate in selection.candidates:
        tier = candidate.tier.value
        grouped.setdefault(tier, []).append(candidate.id)
    return grouped


def _selection_selected_ids_by_tier(
    selection: SelectionResult | None,
) -> dict[str, list[str]]:
    if selection is None:
        return {}
    grouped: dict[str, list[str]] = {}
    for item in selection.selected:
        tier = item.candidate.tier.value
        grouped.setdefault(tier, []).append(item.candidate.id)
    return grouped


def _rejected_reason_counts(rejected: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejected:
        reason = str(item.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _saved_records(entries: list[ExperienceMemory]) -> list[dict[str, str]]:
    return [
        {
            "id": entry.id,
            "tier": entry.tier.value,
            "source_task": entry.source_task,
        }
        for entry in entries
    ]


def _saved_tier_counts(entries: list[ExperienceMemory]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        tier = entry.tier.value
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _hit_tier_counts(
    hits: list[RetrievalHit[ExperienceMemory]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        tier = hit.entry.tier.value
        counts[tier] = counts.get(tier, 0) + 1
    return counts


__all__ = ["LegacyEvolverRuntime"]
