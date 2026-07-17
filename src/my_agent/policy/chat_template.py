"""Canonical chat-template conversion for local white-box policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import json

from my_agent.llm.types import MessageLike, messages_to_openai
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.training.role_views import (
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
)


@dataclass(frozen=True)
class RenderedChat:
    text: str
    prompt_hash: str


class CanonicalChatTemplate:
    def __init__(self, tokenizer: Any, *, configured_template: str = "model_default") -> None:
        self.tokenizer = tokenizer
        if configured_template == "model_default":
            template = getattr(tokenizer, "chat_template", None)
        else:
            template = configured_template
        if not isinstance(template, str) or not template.strip():
            raise ValueError("formal policy requires a non-empty chat template")
        self.template_text = template
        self.template_hash = canonical_sha256(template)

    def render(
        self,
        messages: tuple[CanonicalMessage, ...],
        tools: tuple[CanonicalTool, ...],
    ) -> RenderedChat:
        rendered = self._apply(messages, tools, tokenize=False, return_tensors=None)
        if not isinstance(rendered, str):
            raise TypeError("tokenizer.apply_chat_template must return text when tokenize=false")
        return RenderedChat(text=rendered, prompt_hash=canonical_sha256(rendered))

    def tokenize(
        self,
        messages: tuple[CanonicalMessage, ...],
        tools: tuple[CanonicalTool, ...],
        *,
        return_tensors: str | None = None,
    ) -> Any:
        return self._apply(messages, tools, tokenize=True, return_tensors=return_tensors)

    def _apply(
        self,
        messages: tuple[CanonicalMessage, ...],
        tools: tuple[CanonicalTool, ...],
        *,
        tokenize: bool,
        return_tensors: str | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": True,
            "chat_template": self.template_text,
        }
        if tools:
            kwargs["tools"] = canonical_tools_to_hf(tools)
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors
        return self.tokenizer.apply_chat_template(
            canonical_messages_to_hf(messages),
            **kwargs,
        )


def canonicalize_messages(messages: Sequence[MessageLike]) -> tuple[CanonicalMessage, ...]:
    canonical: list[CanonicalMessage] = []
    for payload in messages_to_openai(list(messages)):
        role = _required_string(payload.get("role"), "message role")
        content_value = payload.get("content", "")
        if content_value is None:
            content_value = ""
        if not isinstance(content_value, str):
            raise ValueError("message content must be a string or null")
        tool_calls: list[CanonicalToolCall] = []
        raw_tool_calls = payload.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ValueError("message tool_calls must be an array")
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, Mapping):
                raise ValueError("message tool_calls entries must be objects")
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                raise ValueError("message tool call requires function payload")
            name = _required_string(function.get("name"), "tool call name")
            arguments = function.get("arguments", "{}")
            arguments_json = _canonical_arguments(arguments)
            call_id = raw_call.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"call_{canonical_sha256({'name': name, 'arguments': json.loads(arguments_json)})[7:19]}"
            tool_calls.append(CanonicalToolCall(call_id, name, arguments_json))
        name_value = payload.get("name")
        tool_call_id_value = payload.get("tool_call_id")
        canonical.append(
            CanonicalMessage(
                role=role,
                content=content_value,
                name=name_value if isinstance(name_value, str) else "",
                tool_call_id=tool_call_id_value if isinstance(tool_call_id_value, str) else "",
                tool_calls=tuple(tool_calls),
            )
        )
    return tuple(canonical)


def canonicalize_tools(tools: Sequence[Mapping[str, Any]] | None) -> tuple[CanonicalTool, ...]:
    canonical: list[CanonicalTool] = []
    for raw_tool in tools or ():
        function = raw_tool.get("function") if raw_tool.get("type") == "function" else raw_tool
        if not isinstance(function, Mapping):
            raise ValueError("tool definition requires a function object")
        name = _required_string(function.get("name"), "tool name")
        description = function.get("description", "")
        if not isinstance(description, str):
            raise ValueError("tool description must be a string")
        parameters = function.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("tool parameters must be a JSON object")
        parameters_json = canonical_json_bytes(dict(parameters)).decode("utf-8")
        canonical.append(
            CanonicalTool(
                name=name,
                description=description,
                parameters_json=parameters_json,
                schema_hash=canonical_sha256(dict(parameters)),
            )
        )
    return tuple(canonical)


def canonical_messages_to_hf(messages: tuple[CanonicalMessage, ...]) -> list[dict[str, Any]]:
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
                        "arguments": json.loads(call.arguments_json),
                    },
                }
                for call in message.tool_calls
            ]
        rendered.append(payload)
    return rendered


def canonical_tools_to_hf(tools: tuple[CanonicalTool, ...]) -> list[dict[str, Any]]:
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


def _canonical_arguments(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("tool call arguments must contain valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("tool call arguments must be a JSON object")
    return canonical_json_bytes(dict(value)).decode("utf-8")


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


__all__ = [
    "CanonicalChatTemplate",
    "RenderedChat",
    "canonical_messages_to_hf",
    "canonical_tools_to_hf",
    "canonicalize_messages",
    "canonicalize_tools",
]
