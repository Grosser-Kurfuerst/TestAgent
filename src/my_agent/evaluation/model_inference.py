"""Model loading and response generation for protocol evaluation."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any


def generate_base_responses(
    *,
    base_model: str,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
    device: str,
    dtype: str,
) -> tuple[list[str], float]:
    print(f"[protocol-eval] Loading base model: {base_model}", flush=True)
    tokenizer, model, torch = _load_base_model(base_model, device=device, dtype=dtype)
    try:
        return _generate_responses(
            tokenizer,
            model,
            torch,
            samples,
            max_new_tokens=max_new_tokens,
            phase="base",
        )
    finally:
        del model
        _empty_cuda_cache(torch)


def generate_sft_responses(
    *,
    base_model: str,
    adapter_dir: str | Path,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
    device: str,
    dtype: str,
) -> tuple[list[str], float]:
    print(
        f"[protocol-eval] Loading SFT model: {base_model} + {adapter_dir}",
        flush=True,
    )
    tokenizer, model, torch = _load_adapter_model(base_model, adapter_dir, device=device, dtype=dtype)
    try:
        return _generate_responses(
            tokenizer,
            model,
            torch,
            samples,
            max_new_tokens=max_new_tokens,
            phase="sft",
        )
    finally:
        del model
        _empty_cuda_cache(torch)


def _load_base_model(base_model: str, *, device: str, dtype: str):
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Model inference requires torch and transformers. Use response files for metric-only runs.") from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = _load_generation_model(
        transformers,
        base_model,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(torch, dtype),
        device_map=_device_map(device),
    )
    model.eval()
    return tokenizer, model, torch


def _load_generation_model(transformers: Any, model_path: str, **model_kwargs: Any):
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


def _load_adapter_model(base_model: str, adapter_dir: str | Path, *, device: str, dtype: str):
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("SFT adapter inference requires peft. Use response files for metric-only runs.") from exc
    tokenizer, model, torch = _load_base_model(base_model, device=device, dtype=dtype)
    adapter_path = _latest_checkpoint(Path(adapter_dir))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"Found missing adapter keys while loading the checkpoint.*",
            category=UserWarning,
        )
        try:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        except UserWarning as exc:
            raise ValueError(
                "SFT adapter is incompatible with the loaded base-model architecture"
            ) from exc
    model.eval()
    return tokenizer, model, torch


def _latest_checkpoint(path: Path) -> Path:
    checkpoints = []
    for child in path.glob("checkpoint-*"):
        suffix = child.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), child))
    if checkpoints:
        return sorted(checkpoints)[-1][1]
    return path


def _device_map(device: str):
    if device == "auto" or "," in device:
        return "auto"
    return device


def _torch_dtype(torch: Any, dtype: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError("dtype must be one of auto, float16, fp16, bfloat16, bf16, float32, fp32.")
    return mapping[dtype]


def _generate_responses(
    tokenizer: Any,
    model: Any,
    torch: Any,
    samples: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    phase: str,
) -> tuple[list[str], float]:
    responses: list[str] = []
    total_tokens = 0
    total_time = 0.0
    device = next(model.parameters()).device
    sample_count = len(samples)
    _print_progress(phase, completed=0, total=sample_count, total_tokens=0, total_time=0.0)

    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            prompt = _build_prompt(tokenizer, sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]
            started = time.perf_counter()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - started
            new_tokens = output_ids[0][input_len:]
            total_tokens += int(new_tokens.shape[0])
            total_time += elapsed
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
            _print_progress(
                phase,
                completed=index,
                total=sample_count,
                total_tokens=total_tokens,
                total_time=total_time,
            )

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    return responses, tokens_per_sec


def _print_progress(
    phase: str,
    *,
    completed: int,
    total: int,
    total_tokens: int,
    total_time: float,
) -> None:
    percentage = (completed / total * 100.0) if total else 100.0
    message = f"\r[protocol-eval] {phase}: {completed}/{total} ({percentage:5.1f}%)"
    if total_time > 0.0:
        message += f" | {total_tokens / total_time:.2f} tok/s"
    print(message, end="\n" if completed >= total else "", flush=True)


def _build_prompt(tokenizer: Any, sample: dict[str, Any]) -> str:
    system = str(sample.get("system", ""))
    user = f"{sample.get('instruction', '')}\n\n{sample.get('input', '')}".strip()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"{system}\n\n{user}\n\n"


def _empty_cuda_cache(torch: Any) -> None:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
