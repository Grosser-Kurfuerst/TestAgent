from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest

from my_agent.policy.contracts import DecisionOutputError, DecisionRequest
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.policy import transformers_policy
from my_agent.policy.chat_template import CanonicalChatTemplate, QWEN35_NOTHINK_TEMPLATE
from my_agent.policy.transformers_policy import (
    TransformersPolicy,
    hash_adapter_artifacts,
    parse_tool_calls,
)
from my_agent.training.role_views import CanonicalMessage, CanonicalTool


RAW_TOOL_CALL = '<tool_call>{"name":"read_file","arguments":{"path":"src/a.py"}}</tool_call>'


class _TinyTokenizer:
    chat_template = "tiny-template-v1"
    eos_token_id = 248046
    pad_token_id = 248044

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


class _BatchEncodingTokenizer(_TinyTokenizer):
    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> object:
        self.rendered_calls.append({"messages": messages, **kwargs})
        if kwargs["tokenize"]:
            return {
                "input_ids": [[101, 102]],
                "attention_mask": [[1, 0]],
            }
        return "<user>task</user><assistant>"


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


class _ScriptedGenerationModel(_TinyModel):
    def __init__(self, completion_ids: list[int]) -> None:
        super().__init__()
        self.completion_ids = list(completion_ids)

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_kwargs = dict(kwargs)
        prompt_ids = kwargs["input_ids"]
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        return [list(prompt_ids) + self.completion_ids]


class _RejectingModelLoader:
    @classmethod
    def from_pretrained(cls, model_path: object, **kwargs: object) -> object:
        raise ValueError("unsupported causal LM config")


class _ImageTextModelLoader:
    @classmethod
    def from_pretrained(cls, model_path: object, **kwargs: object) -> object:
        return {"model_path": model_path, "kwargs": kwargs}


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
        self.assertEqual(self.model.generate_kwargs["eos_token_id"], 248046)
        self.assertEqual(self.model.generate_kwargs["pad_token_id"], 248044)

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

    def test_generation_stops_at_first_tokenizer_or_model_eos(self) -> None:
        model = _ScriptedGenerationModel([201, 248044, 202, 248046])
        model.generation_config = SimpleNamespace(eos_token_id=248044)
        policy = TransformersPolicy(
            model=model,
            tokenizer=self.tokenizer,
            identity=_identity(self.tokenizer),
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

        response = policy.generate_decision(request)

        self.assertEqual(model.generate_kwargs["eos_token_id"], [248046, 248044])
        self.assertEqual(response.completion_token_ids, (201, 248044))

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

    def test_batch_encoding_uses_input_ids_and_preserves_attention_mask(self) -> None:
        tokenizer = _BatchEncodingTokenizer()
        model = _TinyModel()
        policy = TransformersPolicy(
            model=model,
            tokenizer=tokenizer,
            identity=_identity(tokenizer),
            default_max_new_tokens=8,
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

        response = policy.generate_decision(request)
        batch = policy.tokenize(request)

        self.assertEqual(response.prompt_token_ids, (101, 102))
        self.assertEqual(model.generate_kwargs["input_ids"], [[101, 102]])
        self.assertEqual(model.generate_kwargs["attention_mask"], [[1, 0]])
        self.assertEqual(batch.input_ids, [[101, 102]])
        self.assertEqual(batch.attention_mask, [[1, 0]])
        self.assertEqual(batch.assistant_loss_mask, [[0, 0]])

    def test_tool_parser_accepts_qwen_style_tool_call_block(self) -> None:
        calls = parse_tool_calls(RAW_TOOL_CALL)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments_json, '{"path":"src/a.py"}')

    def test_tool_parser_accepts_qwen35_xml_tool_call_block(self) -> None:
        calls = parse_tool_calls(
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>\nsrc/a.py\n</parameter>\n"
            "<parameter=limit>\n12000\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments_json, '{"limit":12000,"path":"src/a.py"}')

    def test_tool_parser_ignores_content_after_first_assistant_turn(self) -> None:
        calls = parse_tool_calls(
            RAW_TOOL_CALL
            + "<|im_end|>\n<|im_start|>user\nignored<|im_end|>\n"
            + '<|im_start|>assistant\n<tool_call>{"name":"run_tests"}</tool_call>'
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")

    def test_legacy_tool_json_ignores_content_after_first_assistant_turn(self) -> None:
        calls = parse_tool_calls(
            json.dumps({
                "tool": "read_file",
                "arguments": {"path": "src/a.py"},
            })
            + "<|im_end|><|im_start|>assistant\n"
            + json.dumps({"tool": "run_tests", "arguments": {}})
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")

    def test_tool_parser_rejects_more_than_four_calls_in_one_decision(self) -> None:
        completion = "".join(
            '<tool_call>{"name":"read_file","arguments":{"path":"file-'
            + str(index)
            + '.py"}}</tool_call>'
            for index in range(5)
        )

        with self.assertRaisesRegex(ValueError, "per-decision tool-call limit"):
            parse_tool_calls(completion)

    def test_tool_parser_accepts_four_calls_in_one_decision(self) -> None:
        completion = "".join(
            '<tool_call>{"name":"read_file","arguments":{"path":"file-'
            + str(index)
            + '.py"}}</tool_call>'
            for index in range(4)
        )

        self.assertEqual(len(parse_tool_calls(completion)), 4)

    def test_local_qwen35_tokenizer_stops_at_first_im_end(self) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            self.skipTest(f"Qwen3.5 tokenizer integration dependencies unavailable: {exc}")
        try:
            snapshot = snapshot_download(
                repo_id="Qwen/Qwen3.5-4B",
                revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001 - local optional integration fixture
            self.skipTest(f"local Qwen3.5 tokenizer snapshot unavailable: {exc}")

        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        template = CanonicalChatTemplate(
            tokenizer,
            configured_template=QWEN35_NOTHINK_TEMPLATE,
        )
        tool_call_ids = tokenizer.encode(RAW_TOOL_CALL, add_special_tokens=False)
        fake_next_turn_ids = tokenizer.encode(
            "<|im_start|>user\nthis must not be parsed<|im_end|>",
            add_special_tokens=False,
        )
        model = _ScriptedGenerationModel([
            *tool_call_ids,
            tokenizer.eos_token_id,
            *fake_next_turn_ids,
        ])
        model.generation_config = SimpleNamespace(eos_token_id=tokenizer.pad_token_id)
        identity = PolicyIdentity(
            base_model="Qwen/Qwen3.5-4B",
            base_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            checkpoint_hash="sha256:" + "1" * 64,
            adapter_hash=None,
            tokenizer_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            tokenizer_hash="sha256:" + "2" * 64,
            chat_template_hash=template.template_hash,
        )
        policy = TransformersPolicy(
            model=model,
            tokenizer=tokenizer,
            identity=identity,
            configured_template=QWEN35_NOTHINK_TEMPLATE,
            torch_module=torch,
        )
        request = DecisionRequest(
            role="action",
            purpose="fast_loop_evidence",
            messages=(CanonicalMessage("user", "read the file"),),
            tools=(CanonicalTool(
                name="read_file",
                description="Read a repository file.",
                parameters_json='{"properties":{"path":{"type":"string"}},"type":"object"}',
                schema_hash=canonical_sha256({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                }),
            ),),
            max_new_tokens=1_024,
            temperature=0.0,
            top_p=1.0,
        )

        response = policy.generate_decision(request)

        self.assertEqual(
            model.generate_kwargs["eos_token_id"],
            [tokenizer.eos_token_id, tokenizer.pad_token_id],
        )
        self.assertEqual(model.generate_kwargs["pad_token_id"], tokenizer.pad_token_id)
        self.assertEqual(response.completion_token_ids[-1], tokenizer.eos_token_id)
        self.assertNotIn("<|im_start|>user", response.raw_completion)
        self.assertTrue(policy.verify_completion_round_trip(response))
        self.assertEqual(len(response.parsed_tool_calls), 1)
        self.assertEqual(response.parsed_tool_calls[0].name, "read_file")

    def test_generation_model_loader_prefers_conditional_qwen35_architecture(self) -> None:
        transformers = SimpleNamespace(
            AutoModelForCausalLM=_RejectingModelLoader,
            AutoModelForImageTextToText=_ImageTextModelLoader,
        )

        model = transformers_policy._load_generation_model(
            transformers,
            "Qwen/Qwen3.5-4B",
            local_files_only=True,
        )

        self.assertEqual(model["model_path"], "Qwen/Qwen3.5-4B")
        self.assertEqual(model["kwargs"], {"local_files_only": True})

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
