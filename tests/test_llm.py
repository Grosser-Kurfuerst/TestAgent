from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import urllib.error

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.llm import ChatResponse, FakeLLM, OpenAICompatibleLLM, build_llm


class _FakeResponse:
    def __init__(self, body: dict[str, object]):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class LLMTests(unittest.TestCase):
    def test_build_llm_uses_fake_provider_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MY_AGENT_LLM_PROVIDER=fake\n", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        self.assertIsInstance(build_llm(config), FakeLLM)

    def test_openai_compatible_client_posts_chat_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "MY_AGENT_LLM_PROVIDER=openai",
                        "MY_AGENT_API_KEY=test-key",
                        "MY_AGENT_BASE_URL=https://example.test/v1",
                        "MY_AGENT_MODEL=test-model",
                        "MY_AGENT_TEMPERATURE=0.2",
                    ]
                ),
                encoding="utf-8",
            )
            config = AgentConfig.from_env(env_file=env_file)
        client = OpenAICompatibleLLM(config, timeout=3)

        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse({"choices": [{"message": {"content": "planned"}}]})

            output = client._chat([{"role": "user", "content": "plan"}])

        self.assertEqual(output, "planned")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "plan"}])
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3)

    def test_openai_chat_sends_tools_and_parses_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "MY_AGENT_LLM_PROVIDER=openai\n"
                "MY_AGENT_API_KEY=test-key\n"
                "MY_AGENT_BASE_URL=https://example.test/v1\n"
                "MY_AGENT_MODEL=test-model\n",
                encoding="utf-8",
            )
            config = AgentConfig.from_env(env_file=env_file)
        client = OpenAICompatibleLLM(config, timeout=3)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            }
        ]

        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_x",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": "{\"path\":\"src/a.py\"}",
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                }
            )

            response = client.chat([{"role": "user", "content": "read"}], tools=tools)

        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.content, "")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.usage.total_tokens, 14)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].id, "call_x")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "src/a.py"})
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["tools"], tools)

    def test_openai_chat_retries_retryable_http_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MY_AGENT_LLM_PROVIDER=openai\nMY_AGENT_API_KEY=test-key\n", encoding="utf-8")
            config = AgentConfig.from_env(env_file=env_file)
        client = OpenAICompatibleLLM(config, timeout=3, max_retries=1)
        error = urllib.error.HTTPError(
            url="https://example.test/v1/chat/completions",
            code=429,
            msg="rate limit",
            hdrs={},
            fp=io.BytesIO(b"limited"),
        )

        with mock.patch("urllib.request.urlopen", side_effect=[error, _FakeResponse({"choices": [{"message": {"content": "ok"}}]})]):
            with mock.patch("time.sleep"):
                output = client._chat([{"role": "user", "content": "hello"}])

        self.assertEqual(output, "ok")

    def test_fake_llm_implements_standard_chat_interface(self) -> None:
        llm = FakeLLM(chat_responses=[ChatResponse(content="done", finish_reason="stop")])

        response = llm.chat([{"role": "user", "content": "hello"}], tools=[])

        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.content, "done")
        self.assertEqual(response.finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
