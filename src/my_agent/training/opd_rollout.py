"""On-policy learner regeneration from typed public and hindsight views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from my_agent.opd_data.export import PreparedLearnerDecision
from my_agent.opd_data.schema import LearnerSample
from my_agent.policy.contracts import DecisionRequest, DecisionResponse, TrainablePolicy
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, require_matching_policy_identity
from my_agent.training.role_views import (
    ActionPublic,
    CanonicalMessage,
    CanonicalTool,
    MaintenancePublic,
)


_ROLE_SYSTEM_PROMPTS = {
    "selection": "Select useful candidate memories for the public task. Return the deployment-format decision.",
    "action": "Solve the public task with the available tools. Do not assume any external memory context.",
    "writing": "Decide which reusable memories to write from the public trajectory and outcome.",
    "maintenance": "Maintain the public repository snapshot using only the provided maintenance tools.",
}
_FORBIDDEN_PUBLIC_MARKERS = (
    "hidden_test_output",
    "official_solution",
    "ground_truth",
    "expected_patch",
    "private_key",
    "api_key",
    "access_token",
    "password",
    "secret",
)


def generate_learner_sample(
    prepared: PreparedLearnerDecision,
    *,
    policy: TrainablePolicy,
    max_new_tokens: int = 1_024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int | None = None,
) -> LearnerSample:
    sample, _response = _generate_learner_sample_with_response(
        prepared,
        policy=policy,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    return sample


def _generate_learner_sample_with_response(
    prepared: PreparedLearnerDecision,
    *,
    policy: TrainablePolicy,
    max_new_tokens: int = 1_024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int | None = None,
) -> tuple[LearnerSample, DecisionResponse]:
    if not isinstance(policy, TrainablePolicy):
        raise ValueError("learner regeneration requires a local TrainablePolicy")
    identity = policy.identity()
    student_messages = render_public_messages(prepared)
    teacher_messages = render_teacher_messages(prepared, public_messages=student_messages)
    tools = _tools_for_public(prepared.public_view)
    _validate_public_safety(prepared, student_messages=student_messages, tools=tools)
    student_request = DecisionRequest(
        role=prepared.role,
        purpose="opd_learner",
        messages=student_messages,
        tools=tools,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    teacher_request = DecisionRequest(
        role=prepared.role,
        purpose="opd_learner",
        messages=teacher_messages,
        tools=tools,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    token_batch = policy.tokenize(student_request)
    response = policy.generate_decision(student_request)
    require_matching_policy_identity(identity, response.identity)
    require_matching_policy_identity(identity, policy.identity())
    if not policy.verify_completion_round_trip(response):
        raise ValueError("learner raw completion does not round-trip from token IDs")
    tokenized_prompt_ids = _first_row_ids(token_batch.input_ids)
    if tokenized_prompt_ids != response.prompt_token_ids:
        raise ValueError("learner prompt token IDs do not match policy.tokenize()")
    student_prompt_hash = policy.render_prompt_hash(student_request)
    teacher_prompt_hash = policy.render_prompt_hash(teacher_request)
    public_prefix_hash = canonical_sha256({
        "messages": [item.to_dict() for item in student_messages],
        "tools": [item.to_dict() for item in tools],
    })
    sample = LearnerSample(
        role=prepared.role,
        collection_round=prepared.collection_round,
        split=prepared.split,
        task_group=prepared.task_group,
        stream_id=prepared.stream_id,
        memory_project_key=prepared.memory_project_key,
        source_evidence_ids=prepared.source_evidence_ids,
        evidence_refs=prepared.evidence_refs,
        policy_identity=identity,
        student_public_view=prepared.public_view.to_dict(),
        teacher_hindsight_view=prepared.hindsight_view.to_dict(),
        canonical_student_messages=student_messages,
        canonical_teacher_messages=teacher_messages,
        canonical_tools=tools,
        student_raw_completion=response.raw_completion,
        student_prompt_token_ids=response.prompt_token_ids,
        student_completion_token_ids=response.completion_token_ids,
        assistant_loss_mask=response.assistant_loss_mask,
        public_prefix_hash=public_prefix_hash,
        student_prompt_hash=student_prompt_hash,
        teacher_prompt_hash=teacher_prompt_hash,
    )
    return sample, response


def generate_action_rollout_samples(
    decisions: Sequence[PreparedLearnerDecision],
    *,
    policy: TrainablePolicy,
    max_new_tokens: int = 1_024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int | None = None,
) -> tuple[LearnerSample, ...]:
    ordered = tuple(sorted(decisions, key=lambda item: item.action_turn_index))
    if not ordered or any(item.role != "action" for item in ordered):
        raise ValueError("action rollout requires action learner decisions")
    rollout_id = ordered[0].action_rollout_id
    if not rollout_id or any(item.action_rollout_id != rollout_id for item in ordered):
        raise ValueError("action rollout decisions must share one rollout ID")
    if tuple(item.action_turn_index for item in ordered) != tuple(range(len(ordered))):
        raise ValueError("action rollout turn indexes must be contiguous from zero")
    first_public = ordered[0].public_view
    if not isinstance(first_public, ActionPublic):
        raise ValueError("action rollout requires ActionPublic")
    current_messages = first_public.prefix_messages
    samples: list[LearnerSample] = []
    for index, decision in enumerate(ordered):
        public = decision.public_view
        if not isinstance(public, ActionPublic):
            raise ValueError("action rollout requires ActionPublic")
        active = replace(
            decision,
            public_view=replace(public, prefix_messages=current_messages),
        )
        sample, response = _generate_learner_sample_with_response(
            active,
            policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=None if seed is None else seed + index,
        )
        samples.append(sample)
        expected_calls = decision.action_expected_tool_calls
        if not expected_calls:
            break
        if not _same_tool_calls(expected_calls, response.parsed_tool_calls):
            break
        observations = _remap_observations(
            decision.action_observation_messages,
            expected_calls=expected_calls,
            actual_calls=response.parsed_tool_calls,
        )
        if not observations:
            break
        chat_response = policy.chat_response_from_decision(response)
        content = getattr(chat_response, "content", None)
        if not isinstance(content, str):
            raise ValueError("action learner response conversion did not produce text content")
        current_messages = (
            *current_messages,
            CanonicalMessage(
                "assistant",
                content,
                tool_calls=response.parsed_tool_calls,
            ),
            *observations,
        )
    return tuple(samples)


def render_public_messages(
    prepared: PreparedLearnerDecision,
) -> tuple[CanonicalMessage, ...]:
    if isinstance(prepared.public_view, ActionPublic):
        return prepared.public_view.prefix_messages
    return (
        CanonicalMessage("system", _ROLE_SYSTEM_PROMPTS[prepared.role]),
        CanonicalMessage(
            "user",
            canonical_json_bytes({"public_view": prepared.public_view.to_dict()}).decode("utf-8"),
        ),
    )


def render_teacher_messages(
    prepared: PreparedLearnerDecision,
    *,
    public_messages: tuple[CanonicalMessage, ...] | None = None,
) -> tuple[CanonicalMessage, ...]:
    public_prefix = public_messages or render_public_messages(prepared)
    return (
        *public_prefix,
        CanonicalMessage(
            "user",
            canonical_json_bytes({
                "privileged_hindsight": prepared.hindsight_view.to_dict(),
                "instruction": (
                    "Use this hindsight only to score the same student completion prefix; "
                    "do not generate a separate target sequence."
                ),
            }).decode("utf-8"),
        ),
    )


def _tools_for_public(public: Any) -> tuple[CanonicalTool, ...]:
    if isinstance(public, (ActionPublic, MaintenancePublic)):
        return public.tools
    return ()


def _validate_public_safety(
    prepared: PreparedLearnerDecision,
    *,
    student_messages: tuple[CanonicalMessage, ...],
    tools: tuple[CanonicalTool, ...],
) -> None:
    serialized = canonical_json_bytes({
        "public_view": prepared.public_view.to_dict(),
        "messages": [item.to_dict() for item in student_messages],
        "tools": [item.to_dict() for item in tools],
    }).decode("utf-8")
    lowered = serialized.lower()
    for marker in _FORBIDDEN_PUBLIC_MARKERS:
        if marker in lowered:
            raise ValueError(f"student public view contains forbidden marker: {marker}")
    if prepared.role != "action":
        return
    for memory_id, content in prepared.forbidden_action_memories:
        if memory_id and memory_id in serialized:
            raise ValueError(f"action student public view leaks selected memory ID: {memory_id}")
        if content and content in serialized:
            raise ValueError(f"action student public view leaks selected memory content: {memory_id}")


def _first_row_ids(value: Any) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("policy.tokenize() input_ids must be list-like")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("learner regeneration expects batch size one")
        value = list(value[0])
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise ValueError("policy.tokenize() returned invalid token IDs")
    return tuple(value)


def _same_tool_calls(
    expected: tuple[Any, ...],
    actual: tuple[Any, ...],
) -> bool:
    return tuple((item.name, item.arguments_json) for item in expected) == tuple(
        (item.name, item.arguments_json) for item in actual
    )


def _remap_observations(
    observations: tuple[CanonicalMessage, ...],
    *,
    expected_calls: tuple[Any, ...],
    actual_calls: tuple[Any, ...],
) -> tuple[CanonicalMessage, ...]:
    call_ids = {
        expected.call_id: actual.call_id
        for expected, actual in zip(expected_calls, actual_calls, strict=True)
    }
    remapped: list[CanonicalMessage] = []
    for observation in observations:
        actual_id = call_ids.get(observation.tool_call_id)
        if actual_id is None:
            raise ValueError("action observation references an unknown fast-loop tool call")
        remapped.append(replace(observation, tool_call_id=actual_id))
    return tuple(remapped)


__all__ = [
    "generate_learner_sample",
    "generate_action_rollout_samples",
    "render_public_messages",
    "render_teacher_messages",
]
