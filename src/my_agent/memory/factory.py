"""Dependency assembly for the concrete MemoryManager facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.llm import AgentLLM
from my_agent.memory.evolver.runtime.factory import build_evolver_runtime
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.short_term import MemoryCompressor, ShortTermMemory

MemoryManagerT = TypeVar("MemoryManagerT")


def build_memory_manager(
    manager_type: type[MemoryManagerT],
    *,
    config: AgentConfig,
    llm: AgentLLM | None,
    repo_path: Path,
    session_id: str | None = None,
    trace_sink: Any | None = None,
) -> MemoryManagerT:
    memory_dir = Path(config.memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    experience_store = ExperienceStore.from_dir(memory_dir, trace_sink=trace_sink)
    experience_store.load()
    context_profile = ContextProfile.resolve(config, model_name(llm, config))
    short_term = ShortTermMemory(
        max_tokens=context_profile.short_term_storage_token_limit,
        max_entries=config.memory_short_term_entries,
    )
    compressor = MemoryCompressor(
        llm=llm,
        chunk_size=config.memory_map_chunk_size,
        retain_recent_turns=config.memory_retain_recent_turns,
        max_input_chars=config.max_summary_input_chars,
    )
    project_key = str(getattr(config, "memory_project_key", "") or "").strip()
    if not project_key:
        project_key = normalize_project_key(repo_path)
    evolver_runtime = build_evolver_runtime(
        config=config,
        llm=llm,
        context_profile=context_profile,
        store=experience_store,
        project_key=project_key,
        trace_sink=trace_sink,
    )
    return manager_type(
        config=config,
        llm=llm,
        repo_path=Path(repo_path),
        short_term=short_term,
        experience_store=experience_store,
        compressor=compressor,
        evolver_runtime=evolver_runtime,
        project_key=project_key,
        session_id=session_id or "",
        trace_sink=trace_sink,
        context_profile=context_profile,
    )


def model_name(llm: AgentLLM | None, config: AgentConfig) -> str:
    value = getattr(llm, "model", "") if llm is not None else ""
    if isinstance(value, str) and value.strip():
        return value
    return config.model


def normalize_project_key(repo_path: Path) -> str:
    try:
        resolved = Path(repo_path).expanduser().resolve()
    except (OSError, RuntimeError):
        resolved = Path(repo_path).expanduser().absolute()
    return str(resolved)


__all__ = ["build_memory_manager", "model_name", "normalize_project_key"]
