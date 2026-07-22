"""Formal selection request construction."""

from __future__ import annotations

from my_agent.policy.contracts import DecisionRequest
from my_agent.policy.identity import canonical_json_bytes
from my_agent.training.role_views import (
    CandidateSnapshotEntry,
    CanonicalMessage,
    SelectionPublic,
)


def build_selection_request(
    *,
    task: str,
    candidates: tuple[CandidateSnapshotEntry, ...],
    token_budget: int,
    max_items: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> DecisionRequest:
    public = SelectionPublic(
        task=task,
        candidates=candidates,
        token_budget=token_budget,
    )
    allowed_labels = {
        tier: [candidate.label for candidate in candidates if candidate.tier == tier]
        for tier in ("skill", "tip", "tool", "trajectory")
    }
    schema = {
        "selected_skills": [],
        "selected_tips": [],
        "selected_tools": [],
        "selected_trajectories": [],
        "reasoning": "brief selection reason",
    }
    return DecisionRequest(
        role="selection",
        purpose="fast_loop_evidence",
        messages=(
            CanonicalMessage(
                "system",
                "Select zero or more useful memories from the provided candidate labels. "
                "Use only labels listed in allowed_labels_by_tier, place each label in its "
                "matching tier field, and never invent or duplicate labels. Respect max_items "
                "and token_budget. If no candidate is useful, return empty arrays. Return "
                "exactly one JSON object matching output_schema and no prose.",
            ),
            CanonicalMessage(
                "user",
                canonical_json_bytes({
                    "public_view": public.to_dict(),
                    "max_items": max_items,
                    "allowed_labels_by_tier": allowed_labels,
                    "output_schema": schema,
                }).decode("utf-8"),
            ),
        ),
        tools=(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


__all__ = ["build_selection_request"]
