"""Deterministic selected-memory rendering for both selection strategies."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from my_agent.memory.evolver.selection.contracts import SelectedExperience
from my_agent.memory.experience.models import ExperienceMemory
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.training.role_views import SELECTED_MEMORY_CONTEXT_HEADER

LEGACY_SELECTED_CONTEXT_HEADER = "Relevant selected experience:"
_TASK_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,80}")
_KNOWN_TASK_REF_RE = re.compile(
    r"(?:"
    r"(?:task|run|stream|case|eval|bench|benchmark|humaneval|mbpp|problem|sample)[A-Za-z0-9_.:-]{0,72}"
    r"|[A-Za-z0-9_.:-]{0,48}(?:task|case|eval|bench|humaneval|mbpp)[A-Za-z0-9_.:-]{0,48}"
    r"|\d{1,12}"
    r")"
)
_SECRET_PREFIX_RE = re.compile(
    r"(?i)(?:ghp_|github_pat_|glpat-|xox[baprs]-|AKIA|ASIA|AIza|ya29\.|eyJ[A-Za-z0-9_-]{8,})"
)
_SECRET_MARKERS = (
    "secret",
    "token",
    "password",
    "api_key",
    "apikey",
    "credential",
    "bearer",
    "private_key",
    "access_key",
)


def render_selected_experiences(
    selected: Sequence[SelectedExperience],
    *,
    max_tokens: int,
) -> MemoryContext[ExperienceMemory]:
    if not selected or max_tokens <= 0:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    header_tokens = estimate_tokens(LEGACY_SELECTED_CONTEXT_HEADER)
    if header_tokens > max_tokens:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    lines = [LEGACY_SELECTED_CONTEXT_HEADER]
    kept: list[SelectedExperience] = []
    used_tokens = header_tokens
    for item in selected:
        entry_text = render_legacy_selected_entry(item)
        entry_tokens = estimate_tokens(entry_text)
        separator_tokens = 1 if kept else 0
        if used_tokens + separator_tokens + entry_tokens > max_tokens:
            continue
        lines.append(entry_text)
        kept.append(item)
        used_tokens += separator_tokens + entry_tokens
    if not kept:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    injected_text = "\n".join(lines)
    return MemoryContext(
        injected_text=injected_text,
        hits=[item.candidate.hit for item in kept],
        estimated_tokens=estimate_tokens(injected_text),
    )


def render_legacy_selected_entry(item: SelectedExperience) -> str:
    candidate = item.candidate
    entry = candidate.hit.entry
    header = (
        f"- [memory_id={entry.id} tier={candidate.tier.value} "
        f"score={candidate.selection_score:.2f} source_task={safe_task_ref(entry.source_task)}]"
    )
    return f"{header}\n{_indent_content(entry.content.strip())}"


def render_formal_selected_context(
    selected_ids: tuple[str, ...],
    hits: tuple[RetrievalHit[ExperienceMemory], ...],
) -> MemoryContext[ExperienceMemory]:
    by_id = {hit.entry.id: hit for hit in hits}
    selected_hits = [by_id[memory_id] for memory_id in selected_ids]
    if not selected_hits:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    blocks = [SELECTED_MEMORY_CONTEXT_HEADER]
    for hit in selected_hits:
        blocks.append(f"[{hit.entry.id} | {hit.entry.tier.value}]\n{hit.entry.content}")
    rendered = "\n\n".join(blocks)
    return MemoryContext(rendered, selected_hits, estimate_tokens(rendered))


def safe_task_ref(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if _looks_secret_like(text):
        return "[redacted]"
    if _TASK_ID_RE.fullmatch(text) and _KNOWN_TASK_REF_RE.fullmatch(text):
        return text
    return f"task_ref_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _indent_content(content: str) -> str:
    return "\n".join(f"  {line}" for line in (content.splitlines() or [""]))


def _looks_secret_like(text: str) -> bool:
    lower = text.casefold()
    if _SECRET_PREFIX_RE.search(text):
        return True
    if any(marker in lower for marker in _SECRET_MARKERS):
        return True
    if lower.startswith("sk-") or " sk-" in lower or "=sk-" in lower or ":sk-" in lower:
        return True
    if "key=" in lower or "key:" in lower or "key-" in lower or "key_" in lower:
        return True
    if "?" in text or "&" in text or "=" in text or "://" in lower:
        return True
    return False


__all__ = [
    "LEGACY_SELECTED_CONTEXT_HEADER",
    "render_formal_selected_context",
    "render_legacy_selected_entry",
    "render_selected_experiences",
    "safe_task_ref",
]
