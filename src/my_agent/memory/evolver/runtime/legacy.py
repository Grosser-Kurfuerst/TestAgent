"""Legacy retrieve/select and writer Evolver runtime strategy."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.memory.evolver.selection.contracts import SelectionResult
from my_agent.memory.evolver.selection.legacy import (
    ExperienceSelector,
    selection_candidate_summary,
    selection_tier_counts,
)
from my_agent.memory.evolver.writing.contracts import (
    ExperienceWriteRequest,
    ExperienceWriteResult,
)
from my_agent.memory.evolver.writing.dataset import (
    MemoryWriterDatasetLogger,
    safe_dataset_join_text,
    safe_error_text,
    saved_records,
    writer_dataset_context_payload,
    writer_dataset_record,
)
from my_agent.memory.evolver.writing.legacy import (
    ExperienceWriter,
    build_write_steps_from_tool_history,
    proposal_tier_counts,
    writer_policy_for_result,
)
from my_agent.memory.evolver.writing.persistence import ExperienceRepositoryWriter
from my_agent.memory.experience.models import (
    ExperienceMemory,
    ExperienceTier,
)
from my_agent.memory.experience.retrieval.contracts import ExperienceRetriever
from my_agent.memory.experience.retrieval.lexical import ExperienceRetrievalMetrics
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.policy.identity import PolicyIdentity


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
        self.repository_writer = ExperienceRepositoryWriter(
            store=store,
            project_key=project_key,
        )
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
        context_payload = writer_dataset_context_payload(request)
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

            safe_source_task = safe_dataset_join_text(request.source_task)
            writer_policy = writer_policy_for_result(
                llm_used=proposed.llm_used,
                fallback_used=proposed.fallback_used,
            )
            persisted = self.repository_writer.write(
                proposed.proposals,
                source_task=safe_source_task,
                run_id=request.run_id,
                stream_id=request.stream_id,
            )
            if persisted.error:
                self._trace("memory.evolver_writer_failed", {
                    **context_payload,
                    "error": safe_error_text(persisted.error),
                    "phase": "unknown",
                    "fallback_attempted": True,
                })
                return ExperienceWriteResult(error=persisted.error)
            result = ExperienceWriteResult(
                proposals=proposed.proposals,
                saved=persisted.saved,
                duplicate_ids=persisted.duplicate_ids,
                rejected=(*proposed.rejected, *persisted.rejected),
                llm_used=proposed.llm_used,
                fallback_used=proposed.fallback_used,
                error=persisted.error,
            )
            self._trace_persisted_entries(result)
            self._trace("memory.evolver_writer_saved", {
                **context_payload,
                "saved_count": len(result.saved),
                "duplicate_count": len(result.duplicate_ids),
                "saved_ids": [entry.id for entry in result.saved],
                "saved_records": saved_records(result.saved),
                "tiers": _saved_tier_counts(list(result.saved)),
                "writer_policy": writer_policy,
            })
            self._append_writer_dataset(request, result)
            return result
        except Exception as exc:  # noqa: BLE001 - writer must not affect agent loop
            error = f"{type(exc).__name__}: {exc}"
            self._trace("memory.evolver_writer_failed", {
                **context_payload,
                "error": safe_error_text(error),
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
                writer_dataset_record(
                    request=request,
                    result=result,
                    selection=self.last_selection,
                )
            )
        except Exception as exc:  # noqa: BLE001 - logging must not affect writes
            self._trace("memory.evolver_writer_failed", {
                **writer_dataset_context_payload(request),
                "error": safe_error_text(f"{type(exc).__name__}: {exc}"),
                "phase": "dataset",
                "fallback_attempted": result.fallback_used,
            })

    def _trace_persisted_entries(self, result: ExperienceWriteResult) -> None:
        for entry in result.saved:
            self._trace_experience_saved(entry, created=True)
        for duplicate_id in result.duplicate_ids:
            duplicate = self.store.get(duplicate_id)
            if duplicate is not None:
                self._trace_experience_saved(duplicate, created=False)

    def _trace_experience_saved(
        self,
        entry: ExperienceMemory,
        *,
        created: bool,
    ) -> None:
        self._trace("memory.evolver_saved", {
            "id": entry.id,
            "created": created,
            "tier": entry.tier.value,
            "scope": entry.scope.value,
            "tokens": entry.token_count,
            "source_task": entry.source_task,
            "created_by": entry.created_by.value,
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


def _rejected_reason_counts(rejected: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejected:
        reason = str(item.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


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
