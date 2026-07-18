from __future__ import annotations

import json
import unittest

from my_agent.policy.chat_template import (
    CanonicalChatTemplate,
    canonicalize_messages,
    canonicalize_tools,
)
from my_agent.policy.identity import canonical_sha256


class _RecordingTokenizer:
    chat_template = "{{ messages }}{{ tools }}"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> object:
        self.calls.append((messages, dict(kwargs)))
        if kwargs["tokenize"]:
            return [[11, 12]] if kwargs.get("return_tensors") == "pt" else [11, 12]
        return json.dumps(
            {"messages": messages, "tools": kwargs.get("tools", [])},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class ChatTemplateTests(unittest.TestCase):
    def test_canonical_render_and_tokenize_share_messages_tools_and_template(self) -> None:
        tokenizer = _RecordingTokenizer()
        template = CanonicalChatTemplate(tokenizer)
        messages = canonicalize_messages([{"role": "user", "content": "read"}])
        tools = canonicalize_tools([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ])

        rendered = template.render(messages, tools)
        token_ids = template.tokenize(messages, tools, return_tensors="pt")

        self.assertEqual(token_ids, [[11, 12]])
        self.assertEqual(rendered.prompt_hash, canonical_sha256(rendered.text))
        self.assertEqual(template.template_hash, canonical_sha256(tokenizer.chat_template))
        self.assertEqual(tokenizer.calls[0][0], tokenizer.calls[1][0])
        self.assertEqual(tokenizer.calls[0][1]["tools"], tokenizer.calls[1][1]["tools"])

    def test_tool_arguments_are_canonicalized_before_template_rendering(self) -> None:
        messages = canonicalize_messages([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"b": 2, "a": 1}'},
                    }
                ],
            }
        ])
        self.assertEqual(messages[0].tool_calls[0].arguments_json, '{"a":1,"b":2}')

    def test_missing_call_ids_use_the_shared_index_aware_rule(self) -> None:
        messages = canonicalize_messages([{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "read_file", "arguments": {"path": "a"}}},
                {"type": "function", "function": {"name": "read_file", "arguments": {"path": "a"}}},
            ],
        }])

        self.assertNotEqual(messages[0].tool_calls[0].call_id, messages[0].tool_calls[1].call_id)
        self.assertEqual(
            messages[0].tool_calls[0].call_id,
            "call_" + canonical_sha256({
                "index": 0,
                "name": "read_file",
                "arguments": {"path": "a"},
            })[7:19],
        )


if __name__ == "__main__":
    unittest.main()
