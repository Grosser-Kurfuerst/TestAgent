"""Build aligned student/teacher causal-LM batches for OPD KL training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from my_agent.opd_data.schema import LearnerSample
from my_agent.policy.contracts import DecisionRequest, TokenBatch, TrainablePolicy
from my_agent.policy.identity import require_matching_policy_identity


class OPDCollator:
    def __init__(
        self,
        policy: TrainablePolicy,
        *,
        torch_module: Any | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        self.policy = policy
        self._torch = torch_module if torch_module is not None else _load_torch()
        self.pad_token_id = _resolve_pad_token_id(policy, pad_token_id)

    def __call__(self, samples: Sequence[LearnerSample]) -> dict[str, Any]:
        if not samples:
            raise ValueError("OPD collator requires at least one sample")
        student_sequences: list[tuple[int, ...]] = []
        teacher_sequences: list[tuple[int, ...]] = []
        student_masks: list[tuple[int, ...]] = []
        teacher_masks: list[tuple[int, ...]] = []
        completion_masks: list[tuple[int, ...]] = []
        student_prediction_indexes: list[tuple[int, ...]] = []
        teacher_prediction_indexes: list[tuple[int, ...]] = []

        identity = self.policy.identity()
        for sample in samples:
            require_matching_policy_identity(identity, sample.policy_identity)
            student_request = _request(sample, teacher=False)
            teacher_request = _request(sample, teacher=True)
            if self.policy.render_prompt_hash(student_request) != sample.student_prompt_hash:
                raise ValueError("student prompt hash does not match learner sample")
            if self.policy.render_prompt_hash(teacher_request) != sample.teacher_prompt_hash:
                raise ValueError("teacher prompt hash does not match learner sample")
            student_prompt = _first_row_ids(self.policy.tokenize(student_request))
            teacher_prompt = _first_row_ids(self.policy.tokenize(teacher_request))
            if student_prompt != sample.student_prompt_token_ids:
                raise ValueError("student prompt token IDs do not match learner sample")
            if not student_prompt or not teacher_prompt:
                raise ValueError("OPD causal-LM prompts must contain at least one token")
            completion = sample.student_completion_token_ids
            completion_mask = sample.assistant_loss_mask
            student_sequences.append((*student_prompt, *completion))
            teacher_sequences.append((*teacher_prompt, *completion))
            student_masks.append((0,) * len(student_prompt) + completion_mask)
            teacher_masks.append((0,) * len(teacher_prompt) + completion_mask)
            completion_masks.append(completion_mask)
            student_prediction_indexes.append(tuple(
                len(student_prompt) - 1 + index for index in range(len(completion))
            ))
            teacher_prediction_indexes.append(tuple(
                len(teacher_prompt) - 1 + index for index in range(len(completion))
            ))

        student_ids, student_attention = self._pad(student_sequences, self.pad_token_id)
        teacher_ids, teacher_attention = self._pad(teacher_sequences, self.pad_token_id)
        student_full_mask, _ = self._pad(student_masks, 0)
        teacher_full_mask, _ = self._pad(teacher_masks, 0)
        completion_mask, _ = self._pad(completion_masks, 0)
        student_indexes, _ = self._pad(student_prediction_indexes, -1)
        teacher_indexes, _ = self._pad(teacher_prediction_indexes, -1)
        return {
            "student_input_ids": student_ids,
            "student_attention_mask": student_attention,
            "student_assistant_loss_mask": student_full_mask,
            "teacher_input_ids": teacher_ids,
            "teacher_attention_mask": teacher_attention,
            "teacher_assistant_loss_mask": teacher_full_mask,
            "completion_mask": completion_mask,
            "student_prediction_indices": student_indexes,
            "teacher_prediction_indices": teacher_indexes,
            "roles": tuple(sample.role for sample in samples),
            "task_groups": tuple(sample.task_group for sample in samples),
            "sample_ids": tuple(sample.sample_id for sample in samples),
        }

    def _pad(
        self,
        rows: Sequence[tuple[int, ...]],
        value: int,
    ) -> tuple[Any, Any]:
        width = max(len(row) for row in rows)
        padded = [list(row) + [value] * (width - len(row)) for row in rows]
        attention = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return (
            self._torch.tensor(padded, dtype=self._torch.long),
            self._torch.tensor(attention, dtype=self._torch.long),
        )


def _request(sample: LearnerSample, *, teacher: bool) -> DecisionRequest:
    return DecisionRequest(
        role=sample.role,
        purpose="opd_learner",
        messages=(
            sample.canonical_teacher_messages
            if teacher
            else sample.canonical_student_messages
        ),
        tools=sample.canonical_tools,
        max_new_tokens=max(1, len(sample.student_completion_token_ids)),
        temperature=0.0,
        top_p=1.0,
    )


def _first_row_ids(batch: TokenBatch) -> tuple[int, ...]:
    value = batch.input_ids
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("policy.tokenize() must return a non-empty rank-2 batch")
    row = value[0]
    if not isinstance(row, (list, tuple)):
        raise ValueError("policy.tokenize() must return a rank-2 batch")
    return tuple(int(item) for item in row)


def _resolve_pad_token_id(policy: TrainablePolicy, configured: int | None) -> int:
    if configured is not None:
        if configured < 0:
            raise ValueError("pad_token_id must be non-negative")
        return configured
    tokenizer = getattr(policy, "tokenizer", None)
    for field_name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, field_name, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("OPD collation requires the 'opd-train' extra") from exc
    return torch


__all__ = ["OPDCollator"]
