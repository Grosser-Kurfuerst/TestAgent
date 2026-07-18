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
    schema = {
        "selected_skills": ["RETRIEVED_SKILL_01"],
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
                "Select only useful candidate labels. Return exactly one JSON object and no prose.",
            ),
            CanonicalMessage(
                "user",
                canonical_json_bytes({
                    "public_view": public.to_dict(),
                    "max_items": max_items,
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
