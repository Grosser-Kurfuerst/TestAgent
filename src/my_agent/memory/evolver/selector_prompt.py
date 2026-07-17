"""Pure-LLM selector prompt, parser, and task policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import json

from my_agent.policy.contracts import DecisionRequest, DecisionResponse, GenerationPolicy
from my_agent.policy.identity import canonical_json_bytes
from my_agent.training.decision_log import (
    DecisionAttemptError,
    DecisionEventContext,
    DecisionEventRecorder,
)
from my_agent.training.role_views import CandidateSnapshotEntry, CanonicalMessage, SelectionPublic


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
        candidates: tuple[CandidateSnapshotEntry, ...],
        token_budget: int,
        max_items: int,
        context: DecisionEventContext,
    ) -> tuple[str, ...]:
        request = build_selection_request(
            task=task,
            candidates=candidates,
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
                candidates=candidates,
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
        return selected_ids[0]


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
    public = SelectionPublic(task=task, candidates=candidates, token_budget=token_budget)
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


def _response_content(policy: GenerationPolicy, response: DecisionResponse) -> str:
    chat_response = policy.chat_response_from_decision(response)
    content = getattr(chat_response, "content", None)
    if not isinstance(content, str):
        raise ValueError("formal selector response conversion did not produce text content")
    return content.strip()


__all__ = [
    "LLMTaskSelectionPolicy",
    "build_selection_request",
    "parse_selection_response",
]
