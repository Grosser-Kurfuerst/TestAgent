from __future__ import annotations

from types import SimpleNamespace

from my_agent.evaluation.memory_benchmark.api_config import ApiEndpoint
from my_agent.evaluation.memory_benchmark.api_policy import MemoryBenchmarkApiPolicy
from my_agent.policy.contracts import (
    DecisionRequest,
    GenerationPolicy,
    TrainablePolicy,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.training.role_views import (
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
)


def _endpoint() -> ApiEndpoint:
    return ApiEndpoint(
        api_key="secret",
        base_url="https://example.test/v1",
        model="qwen-plus",
        endpoint_hash=canonical_sha256("actor-endpoint"),
    )


class _Completions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _client(*responses: object) -> SimpleNamespace:
    completions = _Completions(list(responses))
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions=completions,
    )


def _response(
    *,
    content: str | None = "{}",
    tool_calls: list[object] | None = None,
    usage: object | None = None,
) -> SimpleNamespace:
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
        finish_reason="tool_calls" if tool_calls else "stop",
    )
    response = SimpleNamespace(choices=[choice])
    if usage is not None:
        response.usage = usage
    return response


def _usage(prompt: int = 10, completion: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _tool_call(call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name="apply_memory_ops", arguments='{"keep":[]}'),
    )


def _tool() -> CanonicalTool:
    parameters = {"type": "object", "properties": {"keep": {"type": "array"}}}
    return CanonicalTool(
        name="apply_memory_ops",
        description="Apply maintenance operations.",
        parameters_json=canonical_json_bytes(parameters).decode("utf-8"),
        schema_hash=canonical_sha256(parameters),
    )


def _request(
    role: str,
    *,
    messages: tuple[CanonicalMessage, ...] | None = None,
    tools: tuple[CanonicalTool, ...] = (),
) -> DecisionRequest:
    return DecisionRequest(
        role=role,
        purpose="fast_loop_evidence",
        messages=messages or (CanonicalMessage("system", "Return JSON."),),
        tools=tools,
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9,
    )


def test_policy_is_generation_only_and_identity_is_stable() -> None:
    first = MemoryBenchmarkApiPolicy(_endpoint(), client=_client())
    second = MemoryBenchmarkApiPolicy(_endpoint(), client=_client())

    assert isinstance(first, GenerationPolicy)
    assert not isinstance(first, TrainablePolicy)
    assert first.identity() == second.identity()
    assert "secret" not in first.identity().identity_hash


def test_policy_converts_messages_tools_and_generation_parameters() -> None:
    client = _client(_response(content='{"selected":[]}', usage=_usage()))
    policy = MemoryBenchmarkApiPolicy(_endpoint(), client=client)
    prior_call = CanonicalToolCall("prior-call", "lookup", "{}")
    messages = (
        CanonicalMessage("system", "system"),
        CanonicalMessage("user", "user", name="requester"),
        CanonicalMessage("assistant", "", tool_calls=(prior_call,)),
        CanonicalMessage("tool", "result", tool_call_id="prior-call"),
    )

    result = policy.generate_decision(
        _request("selection", messages=messages, tools=(_tool(),))
    )
    request = client.completions.requests[0]

    assert result.raw_completion == '{"selected":[]}'
    assert result.prompt_token_ids == ()
    assert request["model"] == "qwen-plus"
    assert request["max_tokens"] == 200
    assert request["temperature"] == 0.2
    assert request["top_p"] == 0.9
    assert request["messages"][1]["name"] == "requester"  # type: ignore[index]
    assert request["messages"][2]["tool_calls"][0]["id"] == "prior-call"  # type: ignore[index]
    assert request["messages"][3]["tool_call_id"] == "prior-call"  # type: ignore[index]
    assert request["tools"][0]["function"]["parameters"]["type"] == "object"  # type: ignore[index]
    assert "tool_choice" not in request
    assert "parallel_tool_calls" not in request


def test_selection_and_writing_json_responses_are_preserved() -> None:
    client = _client(
        _response(content='{"selected_labels":["M1"]}', usage=_usage()),
        _response(content='{"records":[]}', usage=_usage()),
    )
    policy = MemoryBenchmarkApiPolicy(_endpoint(), client=client)

    selection = policy.generate_decision(_request("selection"))
    writing = policy.generate_decision(_request("writing"))

    assert selection.raw_completion == '{"selected_labels":["M1"]}'
    assert writing.raw_completion == '{"records":[]}'


def test_maintenance_tool_calls_round_trip_across_two_turns() -> None:
    client = _client(
        _response(content=None, tool_calls=[_tool_call()], usage=_usage()),
        _response(content="done", usage=_usage()),
    )
    policy = MemoryBenchmarkApiPolicy(_endpoint(), client=client)

    first = policy.generate_decision(_request("maintenance", tools=(_tool(),)))
    second_messages = (
        CanonicalMessage("system", "Maintain memory."),
        CanonicalMessage("assistant", "", tool_calls=first.parsed_tool_calls),
        CanonicalMessage("tool", "applied", tool_call_id="call-1"),
    )
    second = policy.generate_decision(
        _request("maintenance", messages=second_messages, tools=(_tool(),))
    )

    assert first.parsed_tool_calls == (
        CanonicalToolCall("call-1", "apply_memory_ops", '{"keep":[]}'),
    )
    assert second.raw_completion == "done"
    for request in client.completions.requests:
        assert request["tool_choice"] == "required"
        assert request["parallel_tool_calls"] is False
    second_payload = client.completions.requests[1]["messages"]  # type: ignore[index]
    assert second_payload[1]["tool_calls"][0]["id"] == "call-1"
    assert second_payload[2]["tool_call_id"] == "call-1"


def test_chat_response_uses_raw_content_and_provider_tool_calls() -> None:
    policy = MemoryBenchmarkApiPolicy(
        _endpoint(),
        client=_client(_response(content="", tool_calls=[_tool_call()], usage=_usage())),
    )

    response = policy.chat_response_from_decision(
        policy.generate_decision(_request("maintenance", tools=(_tool(),)))
    )

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "apply_memory_ops"
    assert response.usage.total_tokens == 13


def test_usage_is_accumulated_by_role_and_missing_usage_is_not_zero_cost() -> None:
    client = _client(
        _response(usage=_usage(7, 2)),
        _response(),
        _response(usage=_usage(5, 1)),
    )
    policy = MemoryBenchmarkApiPolicy(_endpoint(), client=client)

    policy.generate_decision(_request("selection"))
    snapshot = policy.metrics_snapshot()
    policy.generate_decision(_request("writing"))
    after_missing = policy.metrics_since(snapshot)
    snapshot = policy.metrics_snapshot()
    policy.generate_decision(_request("writing"))
    after_known = policy.metrics_since(snapshot)

    assert policy.metrics_snapshot().by_role["selection"].total_tokens == 9
    assert after_missing.by_role["writing"].usage_available is False
    assert after_missing.by_role["writing"].usage_unavailable_calls == 1
    assert after_known.by_role["writing"].usage_available is True
    assert after_known.by_role["writing"].total_tokens == 6


def test_render_prompt_hash_changes_with_messages() -> None:
    policy = MemoryBenchmarkApiPolicy(_endpoint(), client=_client())

    assert policy.render_prompt_hash(_request("selection")) != policy.render_prompt_hash(
        _request(
            "selection",
            messages=(CanonicalMessage("system", "Different prompt."),),
        )
    )
