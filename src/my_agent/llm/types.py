from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, payload: Mapping[str, Any]) -> "ChatUsage":
        usage = payload.get("usage", {})
        if not isinstance(usage, Mapping):
            return cls()
        return cls(
            prompt_tokens=_int_usage_value(usage.get("prompt_tokens")),
            completion_tokens=_int_usage_value(usage.get("completion_tokens")),
            total_tokens=_int_usage_value(usage.get("total_tokens")),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_json: str = "{}"
    type: str = "function"
    arguments_error: str = ""

    @classmethod
    def from_openai(cls, payload: Mapping[str, Any]) -> "LLMToolCall":
        function = payload.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool_calls entry is missing function payload.")

        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool_calls entry is missing function name.")

        raw_arguments = function.get("arguments", "{}")
        if raw_arguments is None:
            raw_arguments = "{}"
        if not isinstance(raw_arguments, str):
            raise ValueError("tool_calls function.arguments must be a JSON string.")

        arguments: dict[str, Any] = {}
        arguments_error = ""
        try:
            parsed = json.loads(raw_arguments or "{}")
            if isinstance(parsed, dict):
                arguments = dict(parsed)
            else:
                arguments_error = "function.arguments JSON must decode to an object."
        except json.JSONDecodeError as exc:
            arguments_error = str(exc)

        raw_id = payload.get("id")
        call_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else f"call_{name.strip()}"
        raw_type = payload.get("type")
        call_type = raw_type if isinstance(raw_type, str) and raw_type.strip() else "function"
        return cls(
            id=call_id,
            name=name.strip(),
            arguments=arguments,
            arguments_json=raw_arguments,
            type=call_type,
            arguments_error=arguments_error,
        )

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return payload


@dataclass(frozen=True)
class ChatResponse:
    role: str = "assistant"
    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: ChatUsage = field(default_factory=ChatUsage)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai_payload(cls, payload: Mapping[str, Any]) -> "ChatResponse":
        try:
            choice = payload["choices"][0]  # type: ignore[index]
            message = choice["message"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LLM response did not contain choices[0].message.") from exc
        if not isinstance(message, Mapping):
            raise ValueError("LLM response message must be an object.")

        content = message.get("content")
        if content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        else:
            raise ValueError("LLM response message.content must be a string or null.")

        tool_calls_payload = message.get("tool_calls") or []
        if not isinstance(tool_calls_payload, list):
            raise ValueError("LLM response message.tool_calls must be an array.")
        tool_calls = [LLMToolCall.from_openai(item) for item in tool_calls_payload if isinstance(item, Mapping)]

        role = message.get("role", "assistant")
        finish_reason = choice.get("finish_reason", "") if isinstance(choice, Mapping) else ""
        return cls(
            role=role if isinstance(role, str) and role else "assistant",
            content=content_text.strip(),
            tool_calls=tool_calls,
            finish_reason=finish_reason if isinstance(finish_reason, str) else "",
            usage=ChatUsage.from_openai(payload),
            raw=dict(payload),
        )


MessageLike = Message | dict[str, Any]


def messages_to_openai(messages: list[MessageLike]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, Message):
            rendered.append(message.to_openai())
        elif isinstance(message, dict):
            rendered.append(dict(message))
        else:
            raise TypeError("messages must contain Message or dict objects.")
    return rendered


def _int_usage_value(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
