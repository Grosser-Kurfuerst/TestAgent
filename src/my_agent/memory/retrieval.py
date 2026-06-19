from __future__ import annotations

import re
from datetime import datetime, timezone

from my_agent.memory.long_term import LongTermMemoryStore
from my_agent.memory.short_term import ShortTermMemory
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import (
    MemoryContext,
    MemoryEntry,
    RetrievalHit,
    normalize_content,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[一-鿿]+")
# Decay floor / horizon from plan §7.
_TIME_DECAY_FLOOR = 0.35
_TIME_DECAY_HORIZON_DAYS = 30.0
_LONG_TERM_SOURCE_WEIGHT = 1.2
_SHORT_TERM_SOURCE_WEIGHT = 1.0


def tokenize(text: str) -> list[str]:
    """Tokenize a query/content for keyword matching.

    English/ASCII runs match as whole words; CJK runs match as contiguous
    spans. The text is casefolded so matching is case-insensitive.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.casefold())


def _match_terms(content: str, query_terms: list[str]) -> tuple[int, tuple[str, ...]]:
    normalized = normalize_content(content)
    matched: list[str] = []
    for term in query_terms:
        if term and term in normalized:
            matched.append(term)
    return len(matched), tuple(matched)


class MemoryRetriever:
    """Score and rank memory entries for a query (plan §7).

    Scoring: ``base * time_decay * source_weight`` where ``base`` is 1.0 on a
    full normalized-query match, otherwise the fraction of query terms found;
    ``time_decay`` decays from 1.0 over 30 days to a floor of 0.35; long-term
    entries get a 1.2 source weight so durable facts outrank an otherwise-equal
    short-term candidate.
    """

    def __init__(self, *, now: datetime | None = None) -> None:
        self._now = now

    def retrieve(
        self,
        query: str,
        *,
        short_term: ShortTermMemory | None,
        long_term: LongTermMemoryStore,
        project_key: str,
        limit: int,
        include_short_term: bool = False,
    ) -> list[RetrievalHit]:
        now = self._resolved_now()
        query_terms = tokenize(query)
        normalized_query = normalize_content(query)

        hits: list[RetrievalHit] = []

        long_entries = long_term.search_candidates(project_key=project_key)
        for entry in long_entries:
            hit = self._score(entry, query, normalized_query, query_terms, now, is_long_term=True)
            if hit is not None:
                hits.append(hit)

        if include_short_term and short_term is not None:
            for entry in short_term.all():
                hit = self._score(entry, query, normalized_query, query_terms, now, is_long_term=False)
                if hit is not None:
                    hits.append(hit)

        hits.sort(key=lambda hit: hit.score, reverse=True)
        if limit > 0:
            return hits[:limit]
        return hits

    def build_context(self, hits: list[RetrievalHit], *, max_tokens: int) -> MemoryContext:
        """Render hits into a fixed-format, token-bounded injection block.

        The default injection path (long-term only) uses the fixed header
        ``Relevant long-term memory:`` from plan §7. When a caller mixes in
        short-term hits via ``retrieve(include_short_term=True)`` (e.g. the
        ``/memory`` debug view), the header generalizes to ``Relevant memory:``
        so it is not factually wrong.

        The returned text never exceeds ``max_tokens``. If even the header plus
        one hit cannot fit, or there are no hits, an empty context is returned
        so the caller injects no memory message at all (plan §10: "没有检索到
        长期记忆，不注入空 memory message").
        """
        if not hits or max_tokens <= 0:
            return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

        has_short_term = any(hit.source_weight != _LONG_TERM_SOURCE_WEIGHT for hit in hits)
        header = "Relevant memory:" if has_short_term else "Relevant long-term memory:"
        header_tokens = estimate_tokens(header)
        if header_tokens > max_tokens:
            # The header alone would exceed the budget; inject nothing rather
            # than emit a truncated/empty memory block.
            return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

        lines: list[str] = [header]
        used_tokens = header_tokens
        kept: list[RetrievalHit] = []
        for hit in hits:
            line = _render_hit(hit)
            line_tokens = estimate_tokens(line)
            # Each appended line is joined with a "\n" separator that
            # estimate_tokens counts, so account for one separator per line to
            # keep the running total consistent with the final estimate.
            separator_tokens = 1 if kept else 0
            if used_tokens + separator_tokens + line_tokens > max_tokens:
                # This hit is too large to fit. Skip it and keep trying the
                # remaining (possibly shorter) hits so a single oversized
                # entry does not starve the context of otherwise relevant,
                # fittable memory (plan §7: inject relevant memory up to the
                # token budget).
                continue
            lines.append(line)
            kept.append(hit)
            used_tokens += separator_tokens + line_tokens

        if not kept:
            # Header fits but no hit fits; per the "no empty memory message"
            # rule, return empty rather than a lone header.
            return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

        injected_text = "\n".join(lines)
        return MemoryContext(
            injected_text=injected_text,
            hits=kept,
            estimated_tokens=estimate_tokens(injected_text),
        )

    def _score(
        self,
        entry: MemoryEntry,
        query: str,
        normalized_query: str,
        query_terms: list[str],
        now: datetime,
        *,
        is_long_term: bool,
    ) -> RetrievalHit | None:
        if not query_terms and not normalized_query:
            return None
        normalized_content_text = normalize_content(entry.content)
        matched_count, matched_terms = _match_terms(entry.content, query_terms)

        if normalized_query and normalized_query in normalized_content_text:
            base = 1.0
        elif matched_count == 0:
            return None
        else:
            base = matched_count / max(1, len(query_terms))

        time_decay = _time_decay(entry.created_at, now)
        source_weight = _LONG_TERM_SOURCE_WEIGHT if is_long_term else _SHORT_TERM_SOURCE_WEIGHT
        score = base * time_decay * source_weight
        return RetrievalHit(
            entry=entry,
            score=score,
            matched_terms=matched_terms,
            source_weight=source_weight,
            time_decay=time_decay,
        )

    def _resolved_now(self) -> datetime:
        if self._now is not None:
            return self._now
        return datetime.now(timezone.utc)


def _time_decay(created_at: datetime, now: datetime) -> float:
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return max(_TIME_DECAY_FLOOR, 1.0 - age_days / _TIME_DECAY_HORIZON_DAYS)


def _render_hit(hit: RetrievalHit) -> str:
    entry = hit.entry
    date_str = entry.created_at.date().isoformat()
    scope = entry.scope.value
    return f"- [{entry.type.value} {scope} {date_str} score={hit.score:.2f}] {entry.content}"


__all__ = ["MemoryRetriever", "tokenize"]
