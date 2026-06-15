from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, Sequence

from my_agent.config import AgentConfig
from my_agent.llm.types import ChatResponse, ChatUsage, LLMToolCall, MessageLike, messages_to_openai


class AgentLLM(Protocol):
    supports_tools: bool

    def chat(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        ...


def build_llm(config: AgentConfig) -> AgentLLM:
    config.require_valid_provider()
    if config.use_fake_llm:
        return FakeLLM()
    if config.provider == "openai":
        config.require_api_key()
        return OpenAICompatibleLLM(config)
    raise RuntimeError(f"Unsupported LLM provider: {config.provider}")


class FakeLLM:
    """Deterministic local model used by tests and smoke runs."""

    supports_tools = True

    def __init__(
        self,
        chat_responses: Sequence[ChatResponse | str] | None = None,
    ):
        self._chat_responses = list(chat_responses or [])
        self._chat_turn = 0

    def chat(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        if self._chat_responses:
            response = self._chat_responses.pop(0)
            if isinstance(response, ChatResponse):
                return response
            return ChatResponse(content=response, finish_reason="stop")
        self._chat_turn += 1
        if self._chat_turn == 1:
            return _tool_chat_response("retrieve_context", {"query": "subtract", "top_k": 3})
        if self._chat_turn == 2:
            return _tool_chat_response("read_file", {"path": "calculator.py", "limit": 12000})
        if self._chat_turn == 3:
            old = 'def subtract(a: int, b: int) -> int:\n    """Return a minus b."""\n    return a + b'
            new = 'def subtract(a: int, b: int) -> int:\n    """Return a minus b."""\n    return a - b'
            return _tool_chat_response("replace_in_file", {"path": "calculator.py", "old": old, "new": new})
        if self._chat_turn == 4:
            return _tool_chat_response("run_tests", {})
        if self._chat_turn == 5:
            return _tool_chat_response("git_diff", {})
        return _tool_chat_response(
            "finish",
            {"summary": "Updated subtract to return the first number minus the second number and ran tests."},
        )


class OpenAICompatibleLLM:
    """Small OpenAI-compatible chat-completions client implemented with stdlib HTTP."""

    supports_tools = True

    def __init__(self, config: AgentConfig, timeout: int = 60, max_retries: int = 2):
        config.require_api_key()
        self.api_key = config.api_key
        self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = config.model
        self.temperature = config.temperature
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def chat(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_openai(messages),
            "temperature": self.temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        body = self._post_chat_completion(payload)
        try:
            parsed = json.loads(body)
            return ChatResponse.from_openai_payload(parsed)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM response could not be parsed: {_excerpt(body)}") from exc

    def _chat(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> str:
        response = self.chat(messages, tools=tools)
        if not response.content:
            raise RuntimeError("LLM response message content was empty.")
        return response.content

    def _post_chat_completion(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        attempts = self.max_retries + 1
        last_error = ""
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = f"LLM request failed with HTTP {exc.code}: {_excerpt(error_body)}"
                if not _is_retryable_http_error(exc.code) or attempt == attempts - 1:
                    raise RuntimeError(last_error) from exc
            except urllib.error.URLError as exc:
                last_error = f"LLM request failed: {exc.reason}"
                if attempt == attempts - 1:
                    raise RuntimeError(last_error) from exc
            time.sleep(min(0.25 * (2**attempt), 1.0))
        raise RuntimeError(last_error or "LLM request failed.")


def _is_retryable_http_error(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _tool_chat_response(tool: str, arguments: dict[str, object]) -> ChatResponse:
    arguments_json = json.dumps(arguments, ensure_ascii=False)
    return ChatResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            LLMToolCall(
                id=f"call_{tool}",
                name=tool,
                arguments=dict(arguments),
                arguments_json=arguments_json,
            )
        ],
    )


def _excerpt(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... truncated"


__all__ = [
    "AgentLLM",
    "ChatResponse",
    "ChatUsage",
    "FakeLLM",
    "MessageLike",
    "OpenAICompatibleLLM",
    "build_llm",
]
