"""Exact full-vocabulary OPD KL with chunked probability materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OPDKLLossOutput:
    loss: Any
    per_token_kl: Any
    token_count: Any


def gather_completion_logits(
    logits: Any,
    prediction_indices: Any,
    completion_mask: Any,
    *,
    torch_module: Any | None = None,
) -> Any:
    """Gather causal next-token logits for the shared student completion prefix."""

    if _is_numpy(logits):
        import numpy as np

        values = np.asarray(logits)
        indexes = np.asarray(prediction_indices)
        mask = np.asarray(completion_mask)
        if values.ndim != 3 or indexes.shape != mask.shape:
            raise ValueError("invalid completion-logit gather shapes")
        if values.shape[0] != indexes.shape[0]:
            raise ValueError("completion indexes do not match logit batch size")
        safe = np.maximum(indexes, 0)
        if np.any((safe >= values.shape[1]) & (mask != 0)):
            raise ValueError("completion prediction index is out of bounds")
        return np.take_along_axis(values, safe[..., None], axis=1)

    torch = torch_module if torch_module is not None else _load_torch()
    if logits.ndim != 3 or prediction_indices.shape != completion_mask.shape:
        raise ValueError("invalid completion-logit gather shapes")
    safe = prediction_indices.clamp_min(0)
    if bool(((safe >= logits.shape[1]) & completion_mask.bool()).any().item()):
        raise ValueError("completion prediction index is out of bounds")
    expanded = safe.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return torch.gather(logits, dim=1, index=expanded)


def chunked_full_vocab_kl(
    teacher_logits: Any,
    student_logits: Any,
    completion_mask: Any,
    *,
    vocab_chunk_size: int = 4_096,
    torch_module: Any | None = None,
) -> OPDKLLossOutput:
    """Compute mean KL(p_teacher || q_student) across every vocabulary token."""

    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")
    if tuple(teacher_logits.shape) != tuple(student_logits.shape):
        raise ValueError("teacher and student logits must have identical shapes")
    if tuple(teacher_logits.shape[:-1]) != tuple(completion_mask.shape):
        raise ValueError("completion mask must align with token logits")
    if _is_numpy(teacher_logits):
        return _numpy_chunked_kl(
            teacher_logits,
            student_logits,
            completion_mask,
            vocab_chunk_size=vocab_chunk_size,
        )
    torch = torch_module if torch_module is not None else _load_torch()
    teacher = teacher_logits.detach().float()
    student = student_logits.float()
    teacher_log_z = _torch_chunked_logsumexp(teacher, vocab_chunk_size, torch)
    student_log_z = _torch_chunked_logsumexp(student, vocab_chunk_size, torch)
    per_token = torch.zeros_like(teacher_log_z, dtype=torch.float32)
    for start in range(0, teacher.shape[-1], vocab_chunk_size):
        stop = min(start + vocab_chunk_size, teacher.shape[-1])
        teacher_log_prob = teacher[..., start:stop] - teacher_log_z.unsqueeze(-1)
        student_log_prob = student[..., start:stop] - student_log_z.unsqueeze(-1)
        probability = teacher_log_prob.exp()
        per_token = per_token + (
            probability * (teacher_log_prob - student_log_prob)
        ).sum(dim=-1)
    active = completion_mask.to(dtype=torch.float32)
    token_count = active.sum()
    if int(token_count.detach().item()) < 1:
        raise ValueError("OPD KL requires at least one active assistant token")
    loss = (per_token * active).sum() / token_count
    return OPDKLLossOutput(loss=loss, per_token_kl=per_token, token_count=token_count)


def chunked_hidden_state_kl(
    teacher_hidden_states: Any,
    student_hidden_states: Any,
    output_weight: Any,
    output_bias: Any | None,
    completion_mask: Any,
    *,
    vocab_chunk_size: int = 4_096,
    torch_module: Any | None = None,
) -> OPDKLLossOutput:
    """Exact KL without ever materializing a full-vocabulary logits tensor."""

    torch = torch_module if torch_module is not None else _load_torch()
    if tuple(teacher_hidden_states.shape) != tuple(student_hidden_states.shape):
        raise ValueError("teacher and student hidden states must have identical shapes")
    if tuple(teacher_hidden_states.shape[:-1]) != tuple(completion_mask.shape):
        raise ValueError("completion mask must align with completion hidden states")
    if output_weight.ndim != 2 or output_weight.shape[1] != teacher_hidden_states.shape[-1]:
        raise ValueError("output projection weight does not match hidden size")
    if output_weight.requires_grad or (
        output_bias is not None and output_bias.requires_grad
    ):
        raise ValueError("chunked OPD requires a frozen output projection")
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")

    class _ChunkedHiddenStateKL(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            teacher_hidden: Any,
            student_hidden: Any,
            weight: Any,
            bias: Any,
            mask: Any,
            chunk_size: int,
        ) -> tuple[Any, Any, Any]:
            teacher = teacher_hidden.detach()
            student = student_hidden.detach()
            frozen_weight = weight.detach()
            frozen_bias = bias.detach() if bias.numel() else None
            active = mask.detach().float()
            token_count = active.sum()
            if int(token_count.item()) < 1:
                raise ValueError("OPD KL requires at least one active assistant token")
            teacher_log_z = _projected_logsumexp(
                teacher,
                frozen_weight,
                frozen_bias,
                chunk_size,
                torch,
            )
            student_log_z = _projected_logsumexp(
                student,
                frozen_weight,
                frozen_bias,
                chunk_size,
                torch,
            )
            per_token = torch.zeros_like(teacher_log_z, dtype=torch.float32)
            for start in range(0, frozen_weight.shape[0], chunk_size):
                stop = min(start + chunk_size, frozen_weight.shape[0])
                teacher_logits = _project_chunk(
                    teacher, frozen_weight, frozen_bias, start, stop, torch
                )
                student_logits = _project_chunk(
                    student, frozen_weight, frozen_bias, start, stop, torch
                )
                teacher_log_prob = teacher_logits - teacher_log_z.unsqueeze(-1)
                student_log_prob = student_logits - student_log_z.unsqueeze(-1)
                probability = teacher_log_prob.exp()
                per_token += (
                    probability * (teacher_log_prob - student_log_prob)
                ).sum(dim=-1)
            loss = (per_token * active).sum() / token_count
            saved_bias = (
                frozen_bias
                if frozen_bias is not None
                else torch.empty(0, device=frozen_weight.device, dtype=frozen_weight.dtype)
            )
            ctx.save_for_backward(
                teacher,
                student,
                frozen_weight,
                saved_bias,
                teacher_log_z,
                student_log_z,
                active,
            )
            ctx.chunk_size = chunk_size
            ctx.has_bias = frozen_bias is not None
            ctx.mark_non_differentiable(per_token, token_count)
            return loss, per_token, token_count

        @staticmethod
        def backward(
            ctx: Any,
            grad_loss: Any,
            _grad_per_token: Any,
            _grad_token_count: Any,
        ) -> tuple[Any, Any, Any, Any, Any, None]:
            (
                teacher,
                student,
                weight,
                saved_bias,
                teacher_log_z,
                student_log_z,
                active,
            ) = ctx.saved_tensors
            bias = saved_bias if ctx.has_bias else None
            token_count = active.sum()
            scale = grad_loss.float() / token_count
            student_gradient = torch.zeros_like(student, dtype=torch.float32)
            for start in range(0, weight.shape[0], ctx.chunk_size):
                stop = min(start + ctx.chunk_size, weight.shape[0])
                teacher_logits = _project_chunk(
                    teacher, weight, bias, start, stop, torch
                )
                student_logits = _project_chunk(
                    student, weight, bias, start, stop, torch
                )
                teacher_probability = (
                    teacher_logits - teacher_log_z.unsqueeze(-1)
                ).exp()
                student_probability = (
                    student_logits - student_log_z.unsqueeze(-1)
                ).exp()
                logit_gradient = (
                    (student_probability - teacher_probability)
                    * active.unsqueeze(-1)
                    * scale
                )
                student_gradient += torch.matmul(
                    logit_gradient,
                    weight[start:stop].float(),
                )
            return (
                None,
                student_gradient.to(dtype=student.dtype),
                None,
                None,
                None,
                None,
            )

    bias_tensor = (
        output_bias
        if output_bias is not None
        else torch.empty(0, device=output_weight.device, dtype=output_weight.dtype)
    )
    loss, per_token, token_count = _ChunkedHiddenStateKL.apply(
        teacher_hidden_states,
        student_hidden_states,
        output_weight,
        bias_tensor,
        completion_mask,
        vocab_chunk_size,
    )
    return OPDKLLossOutput(loss=loss, per_token_kl=per_token, token_count=token_count)


def _projected_logsumexp(
    hidden_states: Any,
    weight: Any,
    bias: Any | None,
    chunk_size: int,
    torch: Any,
) -> Any:
    accumulated = None
    for start in range(0, weight.shape[0], chunk_size):
        stop = min(start + chunk_size, weight.shape[0])
        chunk = torch.logsumexp(
            _project_chunk(hidden_states, weight, bias, start, stop, torch),
            dim=-1,
        )
        accumulated = chunk if accumulated is None else torch.logaddexp(accumulated, chunk)
    if accumulated is None:
        raise ValueError("full-vocabulary KL requires a non-empty vocabulary")
    return accumulated


def _project_chunk(
    hidden_states: Any,
    weight: Any,
    bias: Any | None,
    start: int,
    stop: int,
    torch: Any,
) -> Any:
    chunk_bias = bias[start:stop].float() if bias is not None else None
    return torch.nn.functional.linear(
        hidden_states.float(),
        weight[start:stop].float(),
        chunk_bias,
    )


def _torch_chunked_logsumexp(values: Any, chunk_size: int, torch: Any) -> Any:
    accumulated = None
    for start in range(0, values.shape[-1], chunk_size):
        chunk = torch.logsumexp(values[..., start:start + chunk_size], dim=-1)
        accumulated = chunk if accumulated is None else torch.logaddexp(accumulated, chunk)
    if accumulated is None:
        raise ValueError("full-vocabulary KL requires a non-empty vocabulary")
    return accumulated


def _numpy_chunked_kl(
    teacher_logits: Any,
    student_logits: Any,
    completion_mask: Any,
    *,
    vocab_chunk_size: int,
) -> OPDKLLossOutput:
    import numpy as np

    teacher = np.asarray(teacher_logits, dtype=np.float32)
    student = np.asarray(student_logits, dtype=np.float32)
    mask = np.asarray(completion_mask, dtype=np.float32)
    teacher_log_z = _numpy_chunked_logsumexp(teacher, vocab_chunk_size)
    student_log_z = _numpy_chunked_logsumexp(student, vocab_chunk_size)
    per_token = np.zeros(teacher.shape[:-1], dtype=np.float32)
    for start in range(0, teacher.shape[-1], vocab_chunk_size):
        stop = min(start + vocab_chunk_size, teacher.shape[-1])
        teacher_log_prob = teacher[..., start:stop] - teacher_log_z[..., None]
        student_log_prob = student[..., start:stop] - student_log_z[..., None]
        probability = np.exp(teacher_log_prob)
        per_token += np.sum(
            probability * (teacher_log_prob - student_log_prob),
            axis=-1,
        )
    token_count = float(mask.sum())
    if token_count < 1:
        raise ValueError("OPD KL requires at least one active assistant token")
    loss = float(np.sum(per_token * mask, dtype=np.float64) / token_count)
    return OPDKLLossOutput(loss=loss, per_token_kl=per_token, token_count=token_count)


def _numpy_chunked_logsumexp(values: Any, chunk_size: int) -> Any:
    import numpy as np

    accumulated = None
    for start in range(0, values.shape[-1], chunk_size):
        chunk_values = values[..., start:start + chunk_size]
        maximum = np.max(chunk_values, axis=-1)
        chunk = maximum + np.log(np.exp(chunk_values - maximum[..., None]).sum(axis=-1))
        if accumulated is None:
            accumulated = chunk
        else:
            maximum = np.maximum(accumulated, chunk)
            accumulated = maximum + np.log(
                np.exp(accumulated - maximum) + np.exp(chunk - maximum)
            )
    if accumulated is None:
        raise ValueError("full-vocabulary KL requires a non-empty vocabulary")
    return accumulated


def _is_numpy(value: Any) -> bool:
    return value.__class__.__module__.split(".", 1)[0] == "numpy"


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("OPD KL training requires the 'opd-train' extra") from exc
    return torch


__all__ = [
    "OPDKLLossOutput",
    "chunked_full_vocab_kl",
    "chunked_hidden_state_kl",
    "gather_completion_logits",
]
