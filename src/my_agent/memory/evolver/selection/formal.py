"""Pure-LLM selector prompt, parser, and task policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import json

from my_agent.memory.evolver.selection.contracts import SelectionPolicyCandidate
from my_agent.memory.evolver.selection.prompt import build_selection_request
from my_agent.memory.evolver.selection.service import limit_selected_ids
from my_agent.policy.contracts import DecisionResponse, GenerationPolicy
from my_agent.training.decision_log import (
    DecisionAttemptError,
    DecisionEventContext,
    DecisionEventRecorder,
)
from my_agent.training.role_views import CandidateSnapshotEntry


_OUTPUT_FIELDS = (
    "selected_skills",
    "selected_tips",
    "selected_tools",
    "selected_trajectories",
    "reasoning",
)
_FIELD_TIERS = (
    ("selected_skills", "skill"),
    ("selected_tips", "tip"),
    ("selected_tools", "tool"),
    ("selected_trajectories", "trajectory"),
)


class LLMTaskSelectionPolicy:
    def __init__(
        self,
        *,
        policy: GenerationPolicy,
        recorder: DecisionEventRecorder,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> None:
        self.policy = policy
        self.recorder = recorder
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def select(
        self,
        *,
        task: str,
        candidates: tuple[SelectionPolicyCandidate, ...],
        token_budget: int,
        max_items: int,
        context: DecisionEventContext | None,
    ) -> tuple[str, ...]:
        if context is None:
            raise ValueError("formal selection requires a decision event context")
        formal_candidates = _formal_candidates(candidates)
        request = build_selection_request(
            task=task,
            candidates=formal_candidates,
            token_budget=token_budget,
            max_items=max_items,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        selected_ids: list[tuple[str, ...]] = []

        def parse_response(response: DecisionResponse) -> Mapping[str, Any]:
            parsed, memory_ids = parse_selection_response(
                _response_content(self.policy, response),
                candidates=formal_candidates,
            )
            selected_ids.append(memory_ids)
            return parsed

        try:
            self.recorder.generate(
                request,
                context=context,
                parse_response=parse_response,
            )
        except DecisionAttemptError:
            return ()
        return limit_selected_ids(
            selected_ids[0],
            candidates=formal_candidates,
            token_budget=token_budget,
            max_items=max_items,
        )


class EmptyTaskSelectionPolicy:
    def select(
        self,
        *,
        task: str,
        candidates: tuple[SelectionPolicyCandidate, ...],
        token_budget: int,
        max_items: int,
        context: DecisionEventContext | None,
    ) -> tuple[str, ...]:
        del task, candidates, token_budget, max_items, context
        return ()


class SimilarityTaskSelectionPolicy:
    """Paper ablation that selects by retrieval score without an LLM."""

    def select(
        self,
        *,
        task: str,
        candidates: tuple[SelectionPolicyCandidate, ...],
        token_budget: int,
        max_items: int,
        context: DecisionEventContext | None,
    ) -> tuple[str, ...]:
        del task, context
        formal_candidates = _formal_candidates(candidates)
        ordered = sorted(
            formal_candidates,
            key=lambda item: (-item.retrieval_score, item.memory_id),
        )
        selected: list[str] = []
        used_tokens = 0
        for candidate in ordered:
            candidate_tokens = candidate.token_count
            if (
                len(selected) >= max_items
                or used_tokens + candidate_tokens > token_budget
            ):
                continue
            selected.append(candidate.memory_id)
            used_tokens += candidate_tokens
        return tuple(selected)


def parse_selection_response(
    content: str,
    *,
    candidates: tuple[CandidateSnapshotEntry, ...],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("selector output must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("selector output must be a JSON object")
    if set(payload) != set(_OUTPUT_FIELDS):
        raise ValueError("selector output fields do not match the formal schema")
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str):
        raise ValueError("selector reasoning must be a string")

    by_label = {candidate.label: candidate for candidate in candidates}
    selected_labels: list[str] = []
    normalized: dict[str, Any] = {}
    for field_name, expected_tier in _FIELD_TIERS:
        labels = payload[field_name]
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError(f"selector {field_name} must be an array of labels")
        for label in labels:
            candidate = by_label.get(label)
            if candidate is None:
                raise ValueError(f"selector referenced unknown candidate label: {label}")
            if candidate.tier != expected_tier:
                raise ValueError(f"selector label {label} does not belong to tier {expected_tier}")
            if label in selected_labels:
                raise ValueError(f"selector returned duplicate candidate label: {label}")
            selected_labels.append(label)
        normalized[field_name] = list(labels)
    normalized["reasoning"] = reasoning
    return normalized, tuple(by_label[label].memory_id for label in selected_labels)


def _formal_candidates(
    candidates: tuple[SelectionPolicyCandidate, ...],
) -> tuple[CandidateSnapshotEntry, ...]:
    if any(not isinstance(candidate, CandidateSnapshotEntry) for candidate in candidates):
        raise TypeError("formal selector requires CandidateSnapshotEntry candidates")
    return tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, CandidateSnapshotEntry)
    )


def _response_content(policy: GenerationPolicy, response: DecisionResponse) -> str:
    chat_response = policy.chat_response_from_decision(response)
    content = getattr(chat_response, "content", None)
    if not isinstance(content, str):
        raise ValueError("formal selector response conversion did not produce text content")
    return content.strip()


__all__ = [
    "EmptyTaskSelectionPolicy",
    "LLMTaskSelectionPolicy",
    "SimilarityTaskSelectionPolicy",
    "build_selection_request",
    "parse_selection_response",
]
