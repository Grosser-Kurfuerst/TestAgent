from __future__ import annotations

from types import SimpleNamespace
import unittest

from my_agent.policy.contracts import DecisionRequest
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.policy.transformers_policy import TransformersPolicy, parse_tool_calls
from my_agent.training.role_views import CanonicalMessage


RAW_TOOL_CALL = '<tool_call>{"name":"read_file","arguments":{"path":"src/a.py"}}</tool_call>'


class _TinyTokenizer:
    chat_template = "tiny-template-v1"

    def __init__(self) -> None:
        self.rendered_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> object:
        self.rendered_calls.append({"messages": messages, **kwargs})
        if kwargs["tokenize"]:
            return [[101, 102]]
        return "<user>task</user><assistant>"

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.last_decode = (token_ids, skip_special_tokens)
        return RAW_TOOL_CALL


class _TinyModel:
    device = None

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_kwargs = dict(kwargs)
        return [[101, 102, 201, 202]]

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.forward_kwargs = dict(kwargs)
        return SimpleNamespace(logits="tiny-logits")


def _identity(tokenizer: _TinyTokenizer) -> PolicyIdentity:
    return PolicyIdentity(
        base_model="tiny-model",
        base_revision="tiny-revision",
        checkpoint_hash="sha256:" + "1" * 64,
        adapter_hash=None,
        tokenizer_revision="tiny-tokenizer-revision",
        tokenizer_hash="sha256:" + "2" * 64,
        chat_template_hash=canonical_sha256(tokenizer.chat_template),
    )


class TransformersPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _TinyTokenizer()
        self.model = _TinyModel()
        self.policy = TransformersPolicy(
            model=self.model,
            tokenizer=self.tokenizer,
            identity=_identity(self.tokenizer),
            default_max_new_tokens=8,
        )

    def test_generate_decision_preserves_exact_token_spans_and_identity(self) -> None:
        request = DecisionRequest(
            role="action",
            purpose="opd_learner",
            messages=(CanonicalMessage("user", "task"),),
            tools=(),
            max_new_tokens=8,
            temperature=1.0,
            top_p=0.95,
            seed=None,
        )

        response = self.policy.generate_decision(request)

        self.assertEqual(response.prompt_token_ids, (101, 102))
        self.assertEqual(response.completion_token_ids, (201, 202))
        self.assertEqual(response.assistant_loss_mask, (1, 1))
        self.assertEqual(response.raw_completion, RAW_TOOL_CALL)
        self.assertEqual(response.identity, self.policy.identity())
        self.assertTrue(self.policy.verify_completion_round_trip(response))
        self.assertEqual(response.parsed_tool_calls[0].name, "read_file")

    def test_chat_remains_agent_llm_compatible_and_returns_tool_call(self) -> None:
        response = self.policy.chat(
            [{"role": "user", "content": "read"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "read",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "src/a.py"})
        self.assertEqual(response.usage.prompt_tokens, 2)
        self.assertEqual(response.usage.completion_tokens, 2)

    def test_tokenize_and_forward_logits_expose_white_box_interface(self) -> None:
        request = DecisionRequest(
            role="selection",
            purpose="opd_learner",
            messages=(CanonicalMessage("user", "task"),),
            tools=(),
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
        batch = self.policy.tokenize(request)

        self.assertEqual(batch.input_ids, [[101, 102]])
        self.assertEqual(batch.attention_mask, [[1, 1]])
        self.assertEqual(batch.assistant_loss_mask, [[0, 0]])
        self.assertEqual(self.policy.forward_logits(batch), "tiny-logits")

    def test_tool_parser_accepts_qwen_style_tool_call_block(self) -> None:
        calls = parse_tool_calls(RAW_TOOL_CALL)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments_json, '{"path":"src/a.py"}')


if __name__ == "__main__":
    unittest.main()
