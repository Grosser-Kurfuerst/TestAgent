"""Factory for disabled, legacy, and formal Evolver runtime strategies."""

from __future__ import annotations

from typing import Any

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.llm import AgentLLM
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.runtime.contracts import EvolverRuntime
from my_agent.memory.evolver.runtime.disabled import DisabledEvolverRuntime
from my_agent.memory.evolver.runtime.formal import FormalEvolverRuntime
from my_agent.memory.evolver.runtime.legacy import LegacyEvolverRuntime
from my_agent.memory.evolver.selection.formal import SimilarityTaskSelectionPolicy
from my_agent.memory.evolver.selection.legacy import ExperienceSelector
from my_agent.memory.evolver.writing.legacy import ExperienceWriter
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.experience.retrieval.embedding import (
    EmbeddingRetriever,
    TransformersEmbeddingEncoder,
)
from my_agent.memory.experience.retrieval.lexical import LexicalExperienceRetriever
from my_agent.policy.runtime_validation import require_formal_policy


def build_evolver_runtime(
    *,
    config: AgentConfig,
    llm: AgentLLM | None,
    context_profile: ContextProfile,
    store: ExperienceStore,
    project_key: str,
    trace_sink: Any | None,
) -> EvolverRuntime:
    mode = config.memory_evolver_mode
    if mode == "off":
        return DisabledEvolverRuntime(trace_sink=trace_sink)
    if mode in {"retrieve_select", "full"}:
        def selector_factory() -> ExperienceSelector:
            return ExperienceSelector(
                tier_weights=config.memory_evolver_tier_weights,
                tier_caps=config.memory_evolver_tier_caps,
                selected_max_items=config.memory_evolver_selected_max_items,
                min_score=config.memory_evolver_min_score,
            )

        def writer_factory() -> ExperienceWriter:
            return ExperienceWriter(
                llm=llm,
                min_confidence=config.memory_evolver_writer_min_confidence,
                max_records=config.memory_evolver_writer_max_records,
                max_input_chars=config.memory_evolver_writer_max_input_chars,
                max_content_chars=config.memory_evolver_writer_max_content_chars,
            )
        return LegacyEvolverRuntime(
            mode=mode,
            config=config,
            context_profile=context_profile,
            store=store,
            project_key=project_key,
            retriever=LexicalExperienceRetriever(),
            selector=selector_factory(),
            writer=writer_factory(),
            selector_factory=selector_factory,
            writer_factory=writer_factory,
            write_enabled=mode == "full" and config.memory_evolver_writer_enabled,
            trace_sink=trace_sink,
        )
    if mode == "formal":
        policy_identity = require_formal_policy(config, llm)
        if policy_identity is None:
            raise ValueError("formal memory evolver requires a validated policy identity")
        retriever = (
            LexicalExperienceRetriever()
            if config.memory_evolver_retrieval_backend == "lexical_ablation"
            else EmbeddingRetriever(TransformersEmbeddingEncoder.from_config(config))
        )
        coordinator = EvolverCoordinator(
            store=store,
            project_key=project_key,
            policy_identity=policy_identity,
            retriever=retriever,
            selector=(
                SimilarityTaskSelectionPolicy()
                if config.memory_evolver_selection_backend == "similarity_ablation"
                else None
            ),
            policy=llm,
            dataset_dir=config.memory_evolver_dataset_dir,
            trace_sink=trace_sink,
            top_k_per_tier=config.memory_evolver_candidate_top_k_per_tier,
            selected_max_items=config.memory_evolver_selected_max_items,
            selection_token_budget=config.memory_evolver_selection_prompt_tokens,
            maintenance_interval_tasks=config.memory_evolver_maintenance_interval_tasks,
            maintenance_max_turns=config.memory_evolver_maintenance_max_turns,
            collection_round=config.memory_evolver_collection_round,
            dataset_split=config.memory_evolver_dataset_split,
            maintenance_enabled=config.memory_evolver_maintenance_enabled,
        )
        return FormalEvolverRuntime(
            coordinator=coordinator,
            candidate_retriever=retriever,
        )
    raise ValueError(f"unsupported memory evolver mode: {mode!r}")


__all__ = ["build_evolver_runtime"]
