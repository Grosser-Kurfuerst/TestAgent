"""Local Hugging Face Transformers implementation of the white-box policy."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping
import json
import re
import warnings

from my_agent.config import AgentConfig
from my_agent.llm.types import ChatResponse, ChatUsage, LLMToolCall, MessageLike
from my_agent.policy.chat_template import (
    CanonicalChatTemplate,
    canonicalize_messages,
    canonicalize_tools,
)
from my_agent.policy.contracts import (
    DecisionOutputError,
    DecisionRequest,
    DecisionResponse,
    TokenBatch,
)
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    hash_artifact_path,
    require_matching_policy_identity,
)
from my_agent.training.role_views import CanonicalToolCall


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_QWEN35_FUNCTION_RE = re.compile(
    r"^\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*$",
    re.DOTALL,
)
_QWEN35_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_MODEL_ARTIFACT_NAMES = frozenset({
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
})
_MODEL_ARTIFACT_SUFFIXES = (".safetensors", ".bin")
_ADAPTER_ARTIFACT_NAMES = frozenset({
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
})
_TOKENIZER_PREFIXES = (
    "tokenizer",
    "special_tokens_map",
    "added_tokens",
    "vocab",
    "merges",
    "spiece",
)


class TransformersPolicy:
    supports_tools = True

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        identity: PolicyIdentity,
        configured_template: str = "model_default",
        torch_module: Any | None = None,
        default_temperature: float = 1.0,
        default_top_p: float = 0.95,
        default_max_new_tokens: int = 1_024,
    ) -> None:
        if not isinstance(identity, PolicyIdentity):
            raise ValueError("TransformersPolicy requires PolicyIdentity")
        self.model = model
        self.tokenizer = tokenizer
        self._identity = identity
        self._torch = torch_module
        self.chat_template = CanonicalChatTemplate(
            tokenizer,
            configured_template=configured_template,
        )
        if self.chat_template.template_hash != identity.chat_template_hash:
            raise ValueError("chat template hash does not match PolicyIdentity")
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_max_new_tokens = default_max_new_tokens

    @classmethod
    def from_config(cls, config: AgentConfig) -> "TransformersPolicy":
        if config.policy_backend != "transformers":
            raise ValueError("TransformersPolicy requires policy_backend=transformers")
        torch, transformers, snapshot_download = _load_transformers_dependencies()
        base_snapshot = Path(snapshot_download(
            repo_id=config.policy_base_model,
            revision=config.policy_base_revision,
        ))
        tokenizer_snapshot = Path(snapshot_download(
            repo_id=config.policy_base_model,
            revision=config.policy_tokenizer_revision,
        ))
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_snapshot,
            local_files_only=True,
        )
        chat_template = CanonicalChatTemplate(
            tokenizer,
            configured_template=config.policy_chat_template,
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": _torch_dtype(torch, config.policy_dtype),
        }
        if config.policy_device == "auto":
            model_kwargs["device_map"] = "auto"
        model = _load_generation_model(
            transformers,
            base_snapshot,
            **model_kwargs,
        )
        if config.policy_device != "auto" and hasattr(model, "to"):
            model = model.to(config.policy_device)

        adapter_hash: str | None = None
        if config.policy_adapter_path is not None:
            adapter_path = config.policy_adapter_path.expanduser().resolve()
            if not adapter_path.exists():
                raise FileNotFoundError(f"policy adapter path does not exist: {adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("policy adapters require the 'opd-train' extra") from exc
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    message=r"Found missing adapter keys while loading the checkpoint.*",
                    category=UserWarning,
                )
                try:
                    model = PeftModel.from_pretrained(
                        model,
                        _adapter_load_path(adapter_path),
                        is_trainable=True,
                    )
                except UserWarning as exc:
                    raise ValueError(
                        "policy adapter is incompatible with the loaded base-model architecture"
                    ) from exc
            adapter_hash = hash_adapter_artifacts(adapter_path)

        identity = PolicyIdentity(
            base_model=config.policy_base_model,
            base_revision=_resolved_snapshot_revision(base_snapshot, config.policy_base_revision),
            checkpoint_hash=hash_artifact_path(base_snapshot, include=_is_model_artifact),
            adapter_hash=adapter_hash,
            tokenizer_revision=_resolved_snapshot_revision(
                tokenizer_snapshot,
                config.policy_tokenizer_revision,
            ),
            tokenizer_hash=hash_artifact_path(tokenizer_snapshot, include=_is_tokenizer_artifact),
            chat_template_hash=chat_template.template_hash,
        )
        if hasattr(model, "eval"):
            model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            identity=identity,
            configured_template=config.policy_chat_template,
            torch_module=torch,
            default_temperature=config.temperature,
        )

    def identity(self) -> PolicyIdentity:
        return self._identity

    def render_prompt_hash(self, request: DecisionRequest) -> str:
        return self.chat_template.render(request.messages, request.tools).prompt_hash

    def chat(
        self,
        messages: list[MessageLike],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        request = DecisionRequest(
            role="action",
            purpose="fast_loop_evidence",
            messages=canonicalize_messages(messages),
            tools=canonicalize_tools(tools),
            max_new_tokens=self.default_max_new_tokens,
            temperature=self.default_temperature,
            top_p=self.default_top_p,
        )
        response = self.generate_decision(request)
        return self.chat_response_from_decision(response)

    def chat_response_from_decision(self, response: DecisionResponse) -> ChatResponse:
        require_matching_policy_identity(self._identity, response.identity)
        tool_calls = [_to_llm_tool_call(call) for call in response.parsed_tool_calls]
        decoded_content = self.tokenizer.decode(
            list(response.completion_token_ids),
            skip_special_tokens=True,
        )
        if not isinstance(decoded_content, str):
            raise TypeError("tokenizer.decode must return a string")
        content = _content_without_tool_calls(decoded_content) if tool_calls else decoded_content
        return ChatResponse(
            content=content.strip(),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=ChatUsage(
                prompt_tokens=len(response.prompt_token_ids),
                completion_tokens=len(response.completion_token_ids),
                total_tokens=len(response.prompt_token_ids) + len(response.completion_token_ids),
            ),
            raw={
                "policy_identity": response.identity.to_dict(),
                "policy_identity_hash": response.identity.identity_hash,
                "raw_completion": response.raw_completion,
                "prompt_token_ids": list(response.prompt_token_ids),
                "completion_token_ids": list(response.completion_token_ids),
                "assistant_loss_mask": list(response.assistant_loss_mask),
                "parsed_tool_calls": [call.to_dict() for call in response.parsed_tool_calls],
            },
        )

    def generate_decision(self, request: DecisionRequest) -> DecisionResponse:
        input_ids = self.chat_template.tokenize(
            request.messages,
            request.tools,
            return_tensors="pt",
        )
        input_ids = _move_to_device(input_ids, _model_device(self.model))
        attention_mask = _ones_like(input_ids, self._torch)
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0.0,
            "top_p": request.top_p,
        }
        if request.temperature > 0.0:
            generate_kwargs["temperature"] = request.temperature
        generator = _seeded_generator(self._torch, request.seed, _model_device(self.model))
        if generator is not None:
            generate_kwargs["generator"] = generator
        with _no_grad(self._torch):
            output_ids = self.model.generate(**generate_kwargs)

        prompt_ids = tuple(_first_row_ids(input_ids))
        all_ids = tuple(_first_row_ids(output_ids))
        if all_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("generated token sequence does not preserve the prompt prefix")
        completion_ids = all_ids[len(prompt_ids):]
        raw_completion = self.tokenizer.decode(
            list(completion_ids),
            skip_special_tokens=False,
        )
        if not isinstance(raw_completion, str):
            raise TypeError("tokenizer.decode must return a string")
        try:
            parsed_calls = parse_tool_calls(raw_completion)
        except ValueError as exc:
            response = DecisionResponse(
                raw_completion=raw_completion,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                assistant_loss_mask=(1,) * len(completion_ids),
                parsed_tool_calls=(),
                identity=self._identity,
            )
            raise DecisionOutputError(response, exc) from exc
        return DecisionResponse(
            raw_completion=raw_completion,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            assistant_loss_mask=(1,) * len(completion_ids),
            parsed_tool_calls=parsed_calls,
            identity=self._identity,
        )

    def tokenize(self, request: DecisionRequest) -> TokenBatch:
        input_ids = self.chat_template.tokenize(
            request.messages,
            request.tools,
            return_tensors="pt",
        )
        input_ids = _move_to_device(input_ids, _model_device(self.model))
        return TokenBatch(
            input_ids=input_ids,
            attention_mask=_ones_like(input_ids, self._torch),
            assistant_loss_mask=_zeros_like(input_ids, self._torch),
        )

    def forward_logits(self, batch: TokenBatch) -> Any:
        output = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        logits = getattr(output, "logits", None)
        if logits is None:
            raise RuntimeError("transformers model forward did not return logits")
        return logits

    def forward_hidden_states(
        self,
        batch: TokenBatch,
        *,
        model: Any | None = None,
    ) -> Any:
        causal_model = _causal_lm_model(self.model if model is None else model)
        prefix = getattr(causal_model, "base_model_prefix", None)
        backbone = getattr(causal_model, prefix, None) if isinstance(prefix, str) else None
        if backbone is None or backbone is causal_model:
            raise RuntimeError("causal LM does not expose a separate hidden-state backbone")
        output = backbone(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            return_dict=True,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("transformers backbone did not return last_hidden_state")
        return hidden

    def output_projection(
        self,
        *,
        model: Any | None = None,
    ) -> tuple[Any, Any | None]:
        causal_model = _causal_lm_model(self.model if model is None else model)
        projection = causal_model.get_output_embeddings()
        weight = getattr(projection, "weight", None)
        if weight is None:
            raise RuntimeError("causal LM output projection does not expose weight")
        return weight, getattr(projection, "bias", None)

    def verify_completion_round_trip(self, response: DecisionResponse) -> bool:
        decoded = self.tokenizer.decode(
            list(response.completion_token_ids),
            skip_special_tokens=False,
        )
        return decoded == response.raw_completion


def parse_tool_calls(raw_completion: str) -> tuple[CanonicalToolCall, ...]:
    payloads: list[Mapping[str, Any]] = []
    matches = _TOOL_CALL_RE.findall(raw_completion)
    if matches:
        for match in matches:
            payloads.append(_tool_call_mapping(match))
        residual = _TOOL_CALL_RE.sub("", raw_completion)
        if "<tool_call" in residual or "</tool_call" in residual:
            raise ValueError("generated tool call contains an unmatched marker")
    else:
        stripped = raw_completion.strip()
        if "<tool_call" in stripped or "</tool_call" in stripped:
            raise ValueError("generated tool call contains an unmatched marker")
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                raw_calls = parsed.get("tool_calls")
                if isinstance(raw_calls, list):
                    payloads.extend(item for item in raw_calls if isinstance(item, Mapping))
                elif "name" in parsed or "tool" in parsed:
                    payloads.append(parsed)

    calls: list[CanonicalToolCall] = []
    for index, payload in enumerate(payloads):
        function = payload.get("function") if isinstance(payload.get("function"), Mapping) else payload
        name = function.get("name") if isinstance(function, Mapping) else None
        legacy_name = function.get("tool") if isinstance(function, Mapping) else None
        if (
            isinstance(name, str)
            and name.strip()
            and isinstance(legacy_name, str)
            and legacy_name.strip()
            and name.strip() != legacy_name.strip()
        ):
            raise ValueError("generated tool call name conflicts with legacy tool field")
        if not isinstance(name, str) or not name.strip():
            name = legacy_name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("generated tool call is missing name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("generated tool call arguments are invalid JSON") from exc
        if not isinstance(arguments, Mapping):
            raise ValueError("generated tool call arguments must be an object")
        arguments_json = canonical_json_bytes(dict(arguments)).decode("utf-8")
        raw_id = payload.get("id")
        call_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else (
            f"call_{canonical_sha256({'index': index, 'name': name, 'arguments': dict(arguments)})[7:19]}"
        )
        calls.append(CanonicalToolCall(call_id, name, arguments_json))
    return tuple(calls)


def _load_transformers_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "formal transformers policy requires the 'opd-train' optional dependency"
        ) from exc
    return torch, transformers, snapshot_download


def _torch_dtype(torch: Any, value: str) -> Any:
    mapping = {
        "bfloat16": getattr(torch, "bfloat16", None),
        "float16": getattr(torch, "float16", None),
        "float32": getattr(torch, "float32", None),
    }
    dtype = mapping.get(value)
    if dtype is None:
        raise ValueError(f"unsupported policy dtype: {value!r}")
    return dtype


def _resolved_snapshot_revision(path: Path, fallback: str) -> str:
    if path.parent.name == "snapshots" and path.name:
        return path.name
    return fallback


def _is_model_artifact(path: Path) -> bool:
    return path.name in _MODEL_ARTIFACT_NAMES or path.name.endswith(_MODEL_ARTIFACT_SUFFIXES)


def _is_adapter_artifact(path: Path) -> bool:
    return path.name in _ADAPTER_ARTIFACT_NAMES


def hash_adapter_artifacts(path: str | Path) -> str:
    """Hash deployable adapter files without including trainer/checkpoint metadata."""

    adapter_root = _adapter_load_path(Path(path))
    return hash_artifact_path(
        adapter_root,
        include=lambda item: item.parent == adapter_root and _is_adapter_artifact(item),
    )


def _adapter_load_path(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    candidates = tuple(sorted(
        item.parent for item in path.rglob("adapter_config.json") if item.is_file()
    ))
    if len(candidates) != 1:
        raise ValueError(
            "adapter checkpoint must contain exactly one loadable shared adapter directory"
        )
    return candidates[0]


def _causal_lm_model(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _is_tokenizer_artifact(path: Path) -> bool:
    return path.name.startswith(_TOKENIZER_PREFIXES) or path.name == "chat_template.jinja"


def _move_to_device(value: Any, device: Any) -> Any:
    if device is not None and hasattr(value, "to"):
        return value.to(device)
    return value


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _first_row_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError("token tensor must be convertible to a list")
    if value and isinstance(value[0], list):
        value = value[0]
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise TypeError("token tensor contains non-integer values")
    return list(value)


def _ones_like(value: Any, torch: Any | None) -> Any:
    if torch is not None:
        return torch.ones_like(value)
    return [[1 for _ in row] for row in value] if value and isinstance(value[0], list) else [1 for _ in value]


def _zeros_like(value: Any, torch: Any | None) -> Any:
    if torch is not None:
        return torch.zeros_like(value)
    return [[0 for _ in row] for row in value] if value and isinstance(value[0], list) else [0 for _ in value]


def _no_grad(torch: Any | None) -> Any:
    return torch.no_grad() if torch is not None else nullcontext()


def _seeded_generator(torch: Any | None, seed: int | None, device: Any) -> Any | None:
    if torch is None or seed is None:
        return None
    generator = torch.Generator(device=device) if device is not None else torch.Generator()
    return generator.manual_seed(seed)


def _tool_call_mapping(text: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _qwen35_tool_call_mapping(text)
    if not isinstance(payload, Mapping):
        raise ValueError("generated <tool_call> payload must be an object")
    return payload


def _qwen35_tool_call_mapping(text: str) -> Mapping[str, Any]:
    match = _QWEN35_FUNCTION_RE.fullmatch(text)
    if match is None:
        raise ValueError("generated <tool_call> payload is neither JSON nor Qwen3.5 XML")
    name = match.group(1).strip()
    if not name:
        raise ValueError("generated Qwen3.5 tool call is missing function name")
    arguments: dict[str, Any] = {}
    body = match.group(2)
    for parameter in _QWEN35_PARAMETER_RE.finditer(body):
        parameter_name = parameter.group(1).strip()
        if not parameter_name:
            raise ValueError("generated Qwen3.5 tool call has an empty parameter name")
        raw_value = parameter.group(2).strip()
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        arguments[parameter_name] = value
    residual = _QWEN35_PARAMETER_RE.sub("", body).strip()
    if residual:
        raise ValueError("generated Qwen3.5 tool call contains malformed parameters")
    return {"name": name, "arguments": arguments}


def _load_generation_model(transformers: Any, model_path: Any, **model_kwargs: Any) -> Any:
    loaders = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModelForCausalLM", None),
    ]
    errors: list[Exception] = []
    for loader in loaders:
        if loader is None:
            continue
        try:
            return loader.from_pretrained(model_path, **model_kwargs)
        except ValueError as exc:
            errors.append(exc)
    if errors:
        raise errors[-1]
    raise RuntimeError("transformers does not expose a compatible generation model loader")


def _to_llm_tool_call(call: CanonicalToolCall) -> LLMToolCall:
    arguments = json.loads(call.arguments_json)
    return LLMToolCall(
        id=call.call_id,
        name=call.name,
        arguments=dict(arguments),
        arguments_json=call.arguments_json,
    )


def _content_without_tool_calls(raw_completion: str) -> str:
    return _TOOL_CALL_RE.sub("", raw_completion)


__all__ = ["TransformersPolicy", "hash_adapter_artifacts", "parse_tool_calls"]
