"""Evaluation-only OpenAI-compatible generation policy for memory roles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable
import json

from my_agent.evaluation.memory_benchmark.api_config import ApiEndpoint
from my_agent.llm.types import ChatResponse, ChatUsage, LLMToolCall, MessageLike
from my_agent.policy.chat_template import canonicalize_messages, canonicalize_tools
from my_agent.policy.contracts import DecisionOutputError, DecisionRequest, DecisionResponse
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    require_matching_policy_identity,
)
from my_agent.training.role_views import CanonicalMessage, CanonicalTool, CanonicalToolCall


@dataclass(frozen=True)
class ApiPolicyRoleMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    elapsed_sec: float = 0.0
    usage_available: bool = True
    usage_unavailable_calls: int = 0


@dataclass(frozen=True)
class ApiPolicyMetrics:
    by_role: Mapping[str, ApiPolicyRoleMetrics] = field(default_factory=dict)


class MemoryBenchmarkApiPolicy:
    supports_tools = True

    def __init__(
        self,
        endpoint: ApiEndpoint,
        *,
        client: Any | None = None,
        default_temperature: float = 1.0,
        default_top_p: float = 0.95,
        default_max_new_tokens: int = 1_024,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.endpoint = endpoint
        self.model = endpoint.model
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_max_new_tokens = default_max_new_tokens
        self._client = client or _openai_client(endpoint)
        self._clock = clock
        self._identity = _api_policy_identity(endpoint)
        self._metrics: dict[str, ApiPolicyRoleMetrics] = {}
        self._last_usage = ChatUsage()

    def identity(self) -> PolicyIdentity:
        return self._identity

    def render_prompt_hash(self, request: DecisionRequest) -> str:
        return canonical_sha256(
            {
                "protocol": "openai-compatible-message-protocol-v1",
                "messages": _openai_messages(request.messages),
                "tools": _openai_tools(request.tools),
            }
        )

    def generate_decision(self, request: DecisionRequest) -> DecisionResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _openai_messages(request.messages),
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        tools = _openai_tools(request.tools)
        if tools:
            kwargs["tools"] = tools
            if request.role == "maintenance":
                kwargs["tool_choice"] = "required"
                kwargs["parallel_tool_calls"] = False
        started = self._clock()
        usage_available = False
        usage = ChatUsage()
        try:
            provider_response = self._client.chat.completions.create(**kwargs)
            message, finish_reason = _first_message(provider_response)
            content = _content(message)
            usage, usage_available = _usage(provider_response)
            self._last_usage = usage
            response = DecisionResponse(
                raw_completion=content,
                prompt_token_ids=(),
                completion_token_ids=(),
                assistant_loss_mask=(),
                parsed_tool_calls=(),
                identity=self._identity,
            )
            try:
                parsed_calls = _parsed_tool_calls(message)
            except ValueError as exc:
                raise DecisionOutputError(response, exc) from exc
            return DecisionResponse(
                raw_completion=content,
                prompt_token_ids=(),
                completion_token_ids=(),
                assistant_loss_mask=(),
                parsed_tool_calls=parsed_calls,
                identity=self._identity,
            )
        finally:
            elapsed = max(0.0, self._clock() - started)
            self._record_metrics(
                request.role,
                usage=usage,
                usage_available=usage_available,
                elapsed_sec=elapsed,
            )

    def chat_response_from_decision(self, response: DecisionResponse) -> ChatResponse:
        require_matching_policy_identity(self._identity, response.identity)
        calls = [
            LLMToolCall(
                id=call.call_id,
                name=call.name,
                arguments=json.loads(call.arguments_json),
                arguments_json=call.arguments_json,
            )
            for call in response.parsed_tool_calls
        ]
        return ChatResponse(
            content=response.raw_completion.strip(),
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=self._last_usage,
            raw={
                "policy_identity_hash": response.identity.identity_hash,
                "raw_completion": response.raw_completion,
                "parsed_tool_calls": [call.to_dict() for call in response.parsed_tool_calls],
            },
        )

    def chat(
        self,
        messages: list[MessageLike],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        request = DecisionRequest(
            role="action",
            purpose="fast_loop_evidence",
            messages=canonicalize_messages(messages),
            tools=canonicalize_tools(tools),
            max_new_tokens=self.default_max_new_tokens,
            temperature=self.default_temperature,
            top_p=self.default_top_p,
        )
        return self.chat_response_from_decision(self.generate_decision(request))

    def metrics_snapshot(self) -> ApiPolicyMetrics:
        return ApiPolicyMetrics(by_role=dict(self._metrics))

    def metrics_since(self, snapshot: ApiPolicyMetrics) -> ApiPolicyMetrics:
        roles = set(self._metrics) | set(snapshot.by_role)
        return ApiPolicyMetrics(
            by_role={
                role: _subtract_metrics(
                    self._metrics.get(role, ApiPolicyRoleMetrics()),
                    snapshot.by_role.get(role, ApiPolicyRoleMetrics()),
                )
                for role in sorted(roles)
            }
        )

    def _record_metrics(
        self,
        role: str,
        *,
        usage: ChatUsage,
        usage_available: bool,
        elapsed_sec: float,
    ) -> None:
        current = self._metrics.get(role, ApiPolicyRoleMetrics())
        self._metrics[role] = ApiPolicyRoleMetrics(
            prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
            completion_tokens=current.completion_tokens + usage.completion_tokens,
            total_tokens=current.total_tokens + usage.total_tokens,
            calls=current.calls + 1,
            elapsed_sec=current.elapsed_sec + elapsed_sec,
            usage_available=current.usage_available and usage_available,
            usage_unavailable_calls=(
                current.usage_unavailable_calls + (0 if usage_available else 1)
            ),
        )


def _api_policy_identity(endpoint: ApiEndpoint) -> PolicyIdentity:
    return PolicyIdentity(
        base_model=endpoint.model,
        base_revision=f"api:{endpoint.model}",
        checkpoint_hash=canonical_sha256(
            {
                "provider": "openai_compatible",
                "model": endpoint.model,
                "endpoint_hash": endpoint.endpoint_hash,
            }
        ),
        adapter_hash=None,
        tokenizer_revision=f"api-managed:{endpoint.model}",
        tokenizer_hash=canonical_sha256(
            {"provider": "openai_compatible", "tokenizer": "provider_managed"}
        ),
        chat_template_hash=canonical_sha256(
            "openai-compatible-message-protocol-v1"
        ),
    )


def _openai_messages(messages: Sequence[CanonicalMessage]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json,
                    },
                }
                for call in message.tool_calls
            ]
        rendered.append(payload)
    return rendered


def _openai_tools(tools: Sequence[CanonicalTool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": json.loads(tool.parameters_json),
            },
        }
        for tool in tools
    ]


def _first_message(response: Any) -> tuple[Any, str]:
    choices = _field(response, "choices", None)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ValueError("chat API response must contain choices[0]")
    choice = choices[0]
    message = _field(choice, "message", None)
    if message is None:
        raise ValueError("chat API response must contain choices[0].message")
    finish_reason = _field(choice, "finish_reason", "")
    return message, finish_reason if isinstance(finish_reason, str) else ""


def _content(message: Any) -> str:
    value = _field(message, "content", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("chat API message content must be a string or null")
    return value


def _parsed_tool_calls(message: Any) -> tuple[CanonicalToolCall, ...]:
    raw_calls = _field(message, "tool_calls", None) or ()
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise ValueError("chat API tool_calls must be a sequence")
    calls: list[CanonicalToolCall] = []
    for raw_call in raw_calls:
        function = _field(raw_call, "function", None)
        if function is None:
            raise ValueError("chat API tool call is missing function")
        call_id = _field(raw_call, "id", "")
        name = _field(function, "name", "")
        arguments = _field(function, "arguments", "{}")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("chat API tool call is missing id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("chat API tool call is missing function name")
        if not isinstance(arguments, str):
            raise ValueError("chat API tool call arguments must be JSON text")
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("chat API tool call arguments must be valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("chat API tool call arguments must decode to an object")
        calls.append(
            CanonicalToolCall(
                call_id=call_id.strip(),
                name=name.strip(),
                arguments_json=canonical_json_bytes(dict(parsed)).decode("utf-8"),
            )
        )
    return tuple(calls)


def _usage(response: Any) -> tuple[ChatUsage, bool]:
    raw = _field(response, "usage", None)
    if raw is None:
        return ChatUsage(), False
    values: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _field(raw, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return ChatUsage(), False
        values[name] = value
    return ChatUsage(**values), True


def _subtract_metrics(
    current: ApiPolicyRoleMetrics, previous: ApiPolicyRoleMetrics
) -> ApiPolicyRoleMetrics:
    calls = current.calls - previous.calls
    unavailable_calls = (
        current.usage_unavailable_calls - previous.usage_unavailable_calls
    )
    return ApiPolicyRoleMetrics(
        prompt_tokens=current.prompt_tokens - previous.prompt_tokens,
        completion_tokens=current.completion_tokens - previous.completion_tokens,
        total_tokens=current.total_tokens - previous.total_tokens,
        calls=calls,
        elapsed_sec=max(0.0, current.elapsed_sec - previous.elapsed_sec),
        usage_available=unavailable_calls == 0,
        usage_unavailable_calls=unavailable_calls,
    )


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _openai_client(endpoint: ApiEndpoint) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "memory benchmark API clients require the 'memory-benchmark' extra"
        ) from exc
    return OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)


__all__ = [
    "ApiPolicyMetrics",
    "ApiPolicyRoleMetrics",
    "MemoryBenchmarkApiPolicy",
]
