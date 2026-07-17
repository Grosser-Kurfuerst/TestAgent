"""Generation and training contracts shared by all evolver roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from my_agent.policy.identity import PolicyIdentity
from my_agent.training.role_views import CanonicalMessage, CanonicalTool, CanonicalToolCall


EVOLVER_ROLES = frozenset({"selection", "action", "writing", "maintenance"})
DECISION_PURPOSES = frozenset({"fast_loop_evidence", "opd_learner"})


@dataclass(frozen=True)
class DecisionRequest:
    role: str
    purpose: str
    messages: tuple[CanonicalMessage, ...]
    tools: tuple[CanonicalTool, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.role not in EVOLVER_ROLES:
            raise ValueError(f"unsupported evolver role: {self.role!r}")
        if self.purpose not in DECISION_PURPOSES:
            raise ValueError(f"unsupported decision purpose: {self.purpose!r}")
        if not self.messages:
            raise ValueError("decision request messages must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class DecisionResponse:
    raw_completion: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    assistant_loss_mask: tuple[int, ...]
    parsed_tool_calls: tuple[CanonicalToolCall, ...]
    identity: PolicyIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PolicyIdentity):
            raise ValueError("decision response requires PolicyIdentity")
        _validate_token_ids(self.prompt_token_ids, field_name="prompt_token_ids")
        _validate_token_ids(self.completion_token_ids, field_name="completion_token_ids")
        _validate_binary_mask(self.assistant_loss_mask, field_name="assistant_loss_mask")
        if len(self.assistant_loss_mask) != len(self.completion_token_ids):
            raise ValueError(
                "decision response assistant_loss_mask must align with completion_token_ids"
            )


class DecisionOutputError(ValueError):
    """Invalid structured output after exact generation data was captured."""

    def __init__(self, response: DecisionResponse, cause: Exception) -> None:
        super().__init__(str(cause))
        self.response = response
        self.cause = cause


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Any
    attention_mask: Any
    assistant_loss_mask: Any

    def __post_init__(self) -> None:
        shapes = tuple(_shape_of(value) for value in (
            self.input_ids,
            self.attention_mask,
            self.assistant_loss_mask,
        ))
        if shapes[0] is not None and any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError("TokenBatch tensors must have identical shapes")


def completion_mask_to_full_sequence(
    *,
    prompt_token_count: int,
    completion_mask: tuple[int, ...],
) -> tuple[int, ...]:
    if prompt_token_count < 0:
        raise ValueError("prompt_token_count must be >= 0")
    _validate_binary_mask(completion_mask, field_name="completion_mask")
    return (0,) * prompt_token_count + completion_mask


@runtime_checkable
class GenerationPolicy(Protocol):
    def chat(self, *args: Any, **kwargs: Any) -> Any: ...

    def generate_decision(self, request: DecisionRequest) -> DecisionResponse: ...

    def identity(self) -> PolicyIdentity: ...

    def render_prompt_hash(self, request: DecisionRequest) -> str: ...

    def chat_response_from_decision(self, response: DecisionResponse) -> Any: ...


@runtime_checkable
class TrainablePolicy(GenerationPolicy, Protocol):
    def tokenize(self, request: DecisionRequest) -> TokenBatch: ...

    def forward_logits(self, batch: TokenBatch) -> Any: ...


def _validate_token_ids(values: tuple[int, ...], *, field_name: str) -> None:
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError(f"{field_name} must contain non-negative integer token IDs")


def _validate_binary_mask(values: tuple[int, ...], *, field_name: str) -> None:
    if any(value not in (0, 1) for value in values):
        raise ValueError(f"{field_name} must contain only 0 or 1")


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


__all__ = [
    "DECISION_PURPOSES",
    "EVOLVER_ROLES",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionOutputError",
    "GenerationPolicy",
    "TokenBatch",
    "TrainablePolicy",
    "completion_mask_to_full_sequence",
]
