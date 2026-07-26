from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import urllib.error

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.llm import ChatResponse, FakeLLM, OpenAICompatibleLLM, build_llm, build_policy
from my_agent.policy.identity import PolicyIdentity, policy_identity_manifest_payload
from my_agent.policy.transformers_policy import TransformersPolicy


class _FakeResponse:
    def __init__(self, body: dict[str, object]):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _policy_identity(base_revision: str) -> PolicyIdentity:
    return PolicyIdentity(
        base_model="model",
        base_revision=base_revision,
        checkpoint_hash="sha256:" + "1" * 64,
        adapter_hash=None,
        tokenizer_revision="tokenizer-revision-1",
        tokenizer_hash="sha256:" + "2" * 64,
        chat_template_hash="sha256:" + "3" * 64,
    )


def _formal_config(manifest_path: Path) -> AgentConfig:
    return AgentConfig.from_env(
        {
            "MY_AGENT_LLM_PROVIDER": "fake",
            "AGENTCLI_MEMORY_EVOLVER_MODE": "formal",
            "AGENTCLI_POLICY_BASE_REVISION": "model-revision-1",
            "AGENTCLI_POLICY_TOKENIZER_REVISION": "tokenizer-revision-1",
            "AGENTCLI_POLICY_IDENTITY_MANIFEST": str(manifest_path),
            "AGENTCLI_EMBEDDING_REVISION": "embedding-revision-1",
        },
        env_file=manifest_path.parent / ".env.test",
        require_env_file=False,
    )


class LLMTests(unittest.TestCase):
    def test_build_llm_uses_fake_provider_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MY_AGENT_LLM_PROVIDER=fake\n", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        self.assertIsInstance(build_llm(config), FakeLLM)
        self.assertIsInstance(build_policy(config), FakeLLM)

    def test_build_llm_delegates_formal_mode_to_transformers_policy(self) -> None:
        identity = _policy_identity("model-revision-1")
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "identity.json"
            manifest_path.write_text(
                json.dumps(policy_identity_manifest_payload(identity)),
                encoding="utf-8",
            )
            config = _formal_config(manifest_path)
            sentinel = mock.create_autospec(TransformersPolicy, instance=True)
            sentinel.identity.return_value = identity
            with mock.patch(
                "my_agent.policy.transformers_policy.TransformersPolicy.from_config",
                return_value=sentinel,
            ) as from_config:
                built = build_llm(config)

        self.assertIs(built, sentinel)
        from_config.assert_called_once_with(config)

    def test_formal_builder_rejects_checkpoint_identity_mismatch(self) -> None:
        expected = _policy_identity("expected-revision")
        actual = _policy_identity("actual-revision")
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "identity.json"
            manifest_path.write_text(
                json.dumps(policy_identity_manifest_payload(expected)),
                encoding="utf-8",
            )
            config = _formal_config(manifest_path)
            policy = mock.create_autospec(TransformersPolicy, instance=True)
            policy.identity.return_value = actual
            with mock.patch(
                "my_agent.policy.transformers_policy.TransformersPolicy.from_config",
                return_value=policy,
            ):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    build_policy(config)

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
                        "MY_AGENT_REASONING_EFFORT=high",
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
        self.assertEqual(payload["reasoning_effort"], "high")
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
        self.assertNotIn("reasoning_effort", payload)

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

    def test_openai_chat_retries_socket_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "MY_AGENT_LLM_PROVIDER=openai\nMY_AGENT_API_KEY=test-key\n",
                encoding="utf-8",
            )
            config = AgentConfig.from_env(env_file=env_file)
        client = OpenAICompatibleLLM(config, timeout=3, max_retries=1)

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                TimeoutError("timed out"),
                _FakeResponse({"choices": [{"message": {"content": "ok"}}]}),
            ],
        ) as urlopen:
            with mock.patch("time.sleep") as retry_sleep:
                output = client._chat([{"role": "user", "content": "hello"}])

        self.assertEqual(output, "ok")
        self.assertEqual(urlopen.call_count, 2)
        retry_sleep.assert_called_once_with(0.25)

    def test_openai_chat_wraps_exhausted_socket_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "MY_AGENT_LLM_PROVIDER=openai\nMY_AGENT_API_KEY=test-key\n",
                encoding="utf-8",
            )
            config = AgentConfig.from_env(env_file=env_file)
        client = OpenAICompatibleLLM(config, timeout=3, max_retries=1)

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ) as urlopen:
            with mock.patch("time.sleep"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "LLM request timed out: timed out",
                ):
                    client._chat([{"role": "user", "content": "hello"}])

        self.assertEqual(urlopen.call_count, 2)

    def test_fake_llm_implements_standard_chat_interface(self) -> None:
        llm = FakeLLM(chat_responses=[ChatResponse(content="done", finish_reason="stop")])

        response = llm.chat([{"role": "user", "content": "hello"}], tools=[])

        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.content, "done")
        self.assertEqual(response.finish_reason, "stop")

    def test_fake_llm_does_not_route_team_planner_from_user_content(self) -> None:
        response = FakeLLM().chat(
            [
                {"role": "system", "content": "请压缩以下 Agent 对话片段。"},
                {"role": "user", "content": "We discussed a Multi-Agent coding team design."},
            ],
            tools=None,
        )

        self.assertEqual(response.content, "Compressed map summary generated by FakeLLM.")
        self.assertNotIn('"steps"', response.content)


if __name__ == "__main__":
    unittest.main()
