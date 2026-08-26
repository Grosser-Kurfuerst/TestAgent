from __future__ import annotations

import json
import unittest

from my_agent.policy.chat_template import (
    CanonicalChatTemplate,
    QWEN35_NOTHINK_TEMPLATE,
    canonical_messages_to_hf,
    canonicalize_messages,
    canonicalize_tools,
)
from my_agent.policy.identity import canonical_sha256
from my_agent.training.role_views import CanonicalMessage


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

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _QwenLikeTokenizer(_RecordingTokenizer):
    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> object:
        for index, message in enumerate(messages):
            if message["role"] == "system" and index != 0:
                raise ValueError("System message must be at the beginning.")
        if kwargs["tokenize"] and not kwargs.get("add_generation_prompt", True):
            self.calls.append((messages, dict(kwargs)))
            return [11, 12, 13]
        return super().apply_chat_template(messages, **kwargs)


class ChatTemplateTests(unittest.TestCase):
    def test_qwen35_nothink_mode_disables_thinking_and_binds_identity(self) -> None:
        tokenizer = _RecordingTokenizer()
        template = CanonicalChatTemplate(
            tokenizer,
            configured_template=QWEN35_NOTHINK_TEMPLATE,
        )
        messages = canonicalize_messages([{"role": "user", "content": "read"}])

        template.render(messages, ())

        self.assertIs(tokenizer.calls[0][1]["enable_thinking"], False)
        self.assertEqual(
            template.template_hash,
            canonical_sha256({
                "template": tokenizer.chat_template,
                "enable_thinking": False,
            }),
        )

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

    def test_leading_system_contexts_are_merged_before_qwen_rendering(self) -> None:
        tokenizer = _QwenLikeTokenizer()
        template = CanonicalChatTemplate(
            tokenizer,
            configured_template=QWEN35_NOTHINK_TEMPLATE,
        )
        messages = canonicalize_messages([
            {"role": "system", "content": "coding agent instructions"},
            {
                "role": "system",
                "content": "[Selected evolver memory - frozen for this task]\nuse focused tests",
            },
            {"role": "user", "content": "fix the task"},
        ])

        rendered = template.render(messages, ())
        token_ids = template.tokenize(messages, ())

        self.assertEqual(token_ids, [11, 12])
        self.assertEqual(rendered.prompt_hash, canonical_sha256(rendered.text))
        self.assertEqual(
            [message["role"] for message in tokenizer.calls[0][0]],
            ["system", "user"],
        )
        self.assertEqual(
            tokenizer.calls[0][0][0]["content"],
            "coding agent instructions\n\n"
            "[Selected evolver memory - frozen for this task]\nuse focused tests",
        )
        self.assertEqual(tokenizer.calls[0][0], tokenizer.calls[1][0])

    def test_training_turn_hash_uses_the_same_merged_system_context(self) -> None:
        tokenizer = _QwenLikeTokenizer()
        template = CanonicalChatTemplate(tokenizer)
        messages = canonicalize_messages([
            {"role": "system", "content": "coding agent instructions"},
            {"role": "system", "content": "selected memory"},
            {"role": "user", "content": "fix the task"},
        ])
        target = CanonicalMessage("assistant", "done")

        turn = template.render_training_turn(messages, (), target)

        expected_messages = [
            {"role": "system", "content": "coding agent instructions\n\nselected memory"},
            {"role": "user", "content": "fix the task"},
        ]
        self.assertEqual(tokenizer.calls[0][0], expected_messages)
        self.assertEqual(tokenizer.calls[1][0][:-1], expected_messages)
        self.assertEqual(
            turn.normalized_template_input_hash,
            canonical_sha256({
                "messages": expected_messages,
                "target": canonical_messages_to_hf((target,))[0],
                "tools": [],
            }),
        )

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
