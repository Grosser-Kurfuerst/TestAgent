from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest

from my_agent.policy.contracts import DecisionOutputError, DecisionRequest
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.policy import transformers_policy
from my_agent.policy.transformers_policy import (
    TransformersPolicy,
    hash_adapter_artifacts,
    parse_tool_calls,
)
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


class _MalformedTokenizer(_TinyTokenizer):
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.last_decode = (token_ids, skip_special_tokens)
        return '<tool_call>{"name":"read_file","arguments":</tool_call>'


class _TruncatedToolTokenizer(_TinyTokenizer):
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.last_decode = (token_ids, skip_special_tokens)
        return '<tool_call>{"name":"read_file","arguments":'


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
        self.assertEqual(response.raw["raw_completion"], RAW_TOOL_CALL)
        self.assertEqual(response.raw["assistant_loss_mask"], [1, 1])
        self.assertEqual(response.raw["policy_identity_hash"], self.policy.identity().identity_hash)

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

    def test_tool_parser_accepts_legacy_sft_tool_json(self) -> None:
        calls = parse_tool_calls(json.dumps({
            "tool": "read_file",
            "arguments": {"path": "src/a.py"},
            "reason": "inspect",
        }))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments_json, '{"path":"src/a.py"}')

    def test_tool_parser_rejects_conflicting_native_and_legacy_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            parse_tool_calls(json.dumps({
                "name": "read_file",
                "tool": "run_tests",
                "arguments": {},
            }))

    def test_tool_parser_rejects_legacy_non_object_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an object"):
            parse_tool_calls(json.dumps({
                "tool": "read_file",
                "arguments": ["src/a.py"],
            }))

    def test_invalid_tool_output_retains_exact_generated_token_span(self) -> None:
        tokenizer = _MalformedTokenizer()
        policy = TransformersPolicy(
            model=_TinyModel(),
            tokenizer=tokenizer,
            identity=_identity(tokenizer),
        )
        request = DecisionRequest(
            role="action",
            purpose="fast_loop_evidence",
            messages=(CanonicalMessage("user", "task"),),
            tools=(),
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )

        with self.assertRaises(DecisionOutputError) as captured:
            policy.generate_decision(request)

        self.assertEqual(captured.exception.response.prompt_token_ids, (101, 102))
        self.assertEqual(captured.exception.response.completion_token_ids, (201, 202))
        self.assertEqual(captured.exception.response.assistant_loss_mask, (1, 1))
        self.assertIn("<tool_call>", captured.exception.response.raw_completion)

    def test_unclosed_tool_marker_is_invalid_and_retains_generated_tokens(self) -> None:
        tokenizer = _TruncatedToolTokenizer()
        policy = TransformersPolicy(
            model=_TinyModel(),
            tokenizer=tokenizer,
            identity=_identity(tokenizer),
        )
        request = DecisionRequest(
            role="action",
            purpose="fast_loop_evidence",
            messages=(CanonicalMessage("user", "task"),),
            tools=(),
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )

        with self.assertRaises(DecisionOutputError) as captured:
            policy.generate_decision(request)

        self.assertEqual(captured.exception.response.completion_token_ids, (201, 202))
        self.assertIn("unmatched marker", str(captured.exception.cause))

    def test_nested_shared_adapter_hash_excludes_trainer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "shared"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            first = hash_adapter_artifacts(root)
            (root / "optimizer.pt").write_bytes(b"optimizer")
            (root / "opd_checkpoint_manifest.json").write_text("{}", encoding="utf-8")
            second = hash_adapter_artifacts(root)
            load_path = transformers_policy._adapter_load_path(root)

        self.assertEqual(first, second)
        self.assertEqual(load_path.name, "shared")


if __name__ == "__main__":
    unittest.main()
