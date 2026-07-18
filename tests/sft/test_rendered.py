from __future__ import annotations

import json
import unittest

from my_agent.policy.chat_template import CanonicalChatTemplate
from my_agent.policy.identity import canonical_sha256
from my_agent.sft.rendered import RenderedSFTManifest, RenderedSFTSample
from tests.sft.test_semantic import (
    _action_sample,
    _assistant_text_sample,
    _maintenance_sample,
    _selection_sample,
    _writing_sample,
)


TOKENIZER_HASH = canonical_sha256("fake-tokenizer-v1")


class _NativeToolTokenizer:
    chat_template = "fake-native-tool-template-v1"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        text = self._render(
            messages,
            tools=kwargs.get("tools", []),
            add_generation_prompt=kwargs["add_generation_prompt"],
        )
        if kwargs["tokenize"]:
            return [ord(char) for char in text]
        return text

    def decode(self, token_ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token) for token in token_ids)

    @staticmethod
    def _render(messages, *, tools, add_generation_prompt):
        prefix = "<tools>" + json.dumps(
            tools,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "</tools>"
        parts = [prefix]
        for message in messages:
            role = message["role"]
            if role == "assistant":
                parts.append("<assistant>")
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        function = tool_call["function"]
                        parts.append("<tool_call>" + json.dumps(
                            {
                                "name": function["name"],
                                "arguments": function["arguments"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ) + "</tool_call>")
                else:
                    parts.append(message["content"])
                parts.append("<end>")
            elif role == "tool":
                parts.append(
                    f"<tool id={message['tool_call_id']}>{message['content']}</tool><end>"
                )
            else:
                parts.append(f"<{role}>{message['content']}</{role}><end>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


class RenderedSFTTests(unittest.TestCase):
    def test_all_role_fixtures_render_with_completion_only_labels(self) -> None:
        tokenizer = _NativeToolTokenizer()
        template = CanonicalChatTemplate(tokenizer)
        samples = (
            _action_sample(),
            _assistant_text_sample(),
            _selection_sample(),
            _writing_sample(),
            _maintenance_sample(),
        )

        rendered = tuple(
            RenderedSFTSample.from_semantic(
                sample,
                chat_template=template,
                tokenizer_revision="tokenizer-commit",
                tokenizer_hash=TOKENIZER_HASH,
                cutoff_len=20_000,
            )
            for sample in samples
        )

        for semantic, item in zip(samples, rendered):
            self.assertEqual(item.sample_id, semantic.sample_id)
            self.assertEqual(item.input_ids, item.prompt_token_ids + item.completion_token_ids)
            self.assertEqual(
                tokenizer.decode(item.completion_token_ids, skip_special_tokens=False),
                item.raw_completion,
            )
            self.assertEqual(
                item.full_sequence_loss_mask,
                (0,) * len(item.prompt_token_ids) + (1,) * len(item.completion_token_ids),
            )
            self.assertTrue(all(label == -100 for label in item.labels[: len(item.prompt_token_ids)]))
            self.assertEqual(item.labels[len(item.prompt_token_ids):], item.completion_token_ids)
            self.assertEqual(RenderedSFTSample.from_dict(item.to_dict()), item)

        self.assertIn("<tool_call>", rendered[0].raw_completion)
        self.assertIn("<tool_call>", rendered[-1].raw_completion)
        self.assertEqual(
            [call["add_generation_prompt"] for call in tokenizer.calls[:2]],
            [True, False],
        )

    def test_cutoff_fails_closed_instead_of_truncating(self) -> None:
        tokenizer = _NativeToolTokenizer()
        template = CanonicalChatTemplate(tokenizer)
        sample = _action_sample()
        turn = template.render_training_turn(sample.messages, sample.tools, sample.target)
        with self.assertRaisesRegex(ValueError, "cannot be truncated"):
            RenderedSFTSample.from_semantic(
                sample,
                chat_template=template,
                tokenizer_revision="tokenizer-commit",
                tokenizer_hash=TOKENIZER_HASH,
                cutoff_len=len(turn.input_ids) - 1,
            )

    def test_rendered_manifest_binds_semantic_manifest_and_sample_hashes(self) -> None:
        tokenizer = _NativeToolTokenizer()
        template = CanonicalChatTemplate(tokenizer)
        sample = RenderedSFTSample.from_semantic(
            _action_sample(),
            chat_template=template,
            tokenizer_revision="tokenizer-commit",
            tokenizer_hash=TOKENIZER_HASH,
            cutoff_len=20_000,
        )
        manifest = RenderedSFTManifest.create(
            semantic_dataset_manifest_hash=canonical_sha256({"dataset": "canonical"}),
            tokenizer_revision="tokenizer-commit",
            tokenizer_hash=TOKENIZER_HASH,
            chat_template_hash=template.template_hash,
            cutoff_len=20_000,
            split_rendered_sample_hashes={"train": (sample.rendered_sample_hash,)},
        )

        self.assertEqual(RenderedSFTManifest.from_dict(manifest.to_dict()), manifest)
        self.assertEqual(
            manifest.split_rendered_sample_hashes["train"],
            (sample.rendered_sample_hash,),
        )


if __name__ == "__main__":
    unittest.main()
