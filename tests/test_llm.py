from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.llm import FakeLLM, OpenAICompatibleLLM, build_llm


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


if __name__ == "__main__":
    unittest.main()
