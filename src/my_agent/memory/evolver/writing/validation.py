"""Shared deterministic validation for all Experience writer proposals."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from my_agent.memory.evolver.writing.contracts import ExperienceWriteProposal
from my_agent.memory.experience.models import ExperienceTier, ToolPayload, normalize_experience_tier
from my_agent.memory.experience.serialization import (
    experience_payload_from_dict,
    experience_payload_to_dict,
)

MAX_REASON_CHARS = 500
FORBIDDEN_MARKERS = (
    "hidden_test_output",
    "hidden_ok",
    "official_solution",
    "ground_truth",
    "expected_patch",
    "private_key",
    "api_key",
    "apikey",
    "bearer",
    "github_pat_",
    "ghp_",
    "glpat-",
    "sk-",
)
SECRET_PREFIX_RE = re.compile(
    r"(?i)(?:github_pat_|ghp_|glpat-|xox[baprs]-|AKIA|ASIA|AIza|ya29\.|eyJ[A-Za-z0-9_-]{8,})"
)
DESTRUCTIVE_COMMAND_RE = re.compile(
    r"(?i)(?:\brm\s+-rf\b|\bgit\s+reset\s+--hard\b)"
)


class ExperienceProposalValidator:
    def __init__(
        self,
        *,
        min_confidence: float = 0.70,
        max_records: int = 6,
        max_content_chars: int = 1_200,
    ) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_records = max(0, int(max_records))
        self.max_content_chars = max(0, int(max_content_chars))

    def validate_proposals(
        self,
        proposals: Sequence[ExperienceWriteProposal],
    ) -> tuple[tuple[ExperienceWriteProposal, ...], tuple[dict[str, Any], ...]]:
        if self.max_records <= 0:
            return (), tuple(
                rejection(proposal, "max_records_zero") for proposal in proposals
            )

        accepted: list[ExperienceWriteProposal] = []
        rejected: list[dict[str, Any]] = []
        seen: set[tuple[ExperienceTier, str]] = set()
        for proposal in proposals:
            valid, reason, normalized = self.validate_proposal(proposal)
            if not valid or normalized is None:
                rejected.append(rejection(proposal, reason))
                continue
            identity = (normalized.tier, content_fingerprint(normalized.content))
            if identity in seen:
                rejected.append(rejection(proposal, "duplicate_proposal"))
                continue
            seen.add(identity)
            accepted.append(normalized)
            if len(accepted) >= self.max_records:
                break
        return tuple(accepted), tuple(rejected)

    def validate_proposal(
        self,
        proposal: ExperienceWriteProposal,
    ) -> tuple[bool, str, ExperienceWriteProposal | None]:
        tier = normalize_experience_tier(proposal.tier)
        if tier is None:
            return False, "invalid_tier", None

        content = str(proposal.content or "").strip()
        if not content:
            return False, "empty_content", None

        confidence = safe_float(proposal.confidence)
        if confidence is None:
            return False, "invalid_confidence", None
        if confidence < self.min_confidence:
            return False, "low_confidence", None

        content = truncate_text(content, self.max_content_chars).strip()
        if not content:
            return False, "empty_content", None

        try:
            if isinstance(proposal.payload, Mapping):
                payload = experience_payload_from_dict(tier, proposal.payload)
            else:
                payload = experience_payload_from_dict(
                    tier,
                    experience_payload_to_dict(proposal.payload),
                )
        except (TypeError, ValueError):
            return False, "invalid_payload", None

        raw_payload = experience_payload_to_dict(payload)
        raw_reason = str(proposal.reason or "").strip()
        if (
            contains_forbidden_content(content)
            or contains_forbidden_content(raw_reason)
            or contains_forbidden_content(json_text(raw_payload))
        ):
            return False, "unsafe_content", None
        if isinstance(payload, ToolPayload) and contains_destructive_tool(payload):
            return False, "unsafe_tool_command", None

        return True, "", ExperienceWriteProposal(
            tier=tier,
            content=content,
            payload=payload,
            confidence=confidence,
            reason=truncate_text(raw_reason, MAX_REASON_CHARS).strip(),
        )


def contains_forbidden_content(value: str) -> bool:
    text = str(value or "")
    lower = text.casefold()
    return bool(SECRET_PREFIX_RE.search(text)) or any(
        marker in lower for marker in FORBIDDEN_MARKERS
    )


def contains_destructive_tool(payload: ToolPayload) -> bool:
    return any(
        isinstance(value, str) and DESTRUCTIVE_COMMAND_RE.search(value)
        for value in (payload.code, payload.command)
    )


def json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def truncate_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return "." * max_chars
    return value[: max_chars - 3].rstrip() + "..."


def content_fingerprint(content: str) -> str:
    return " ".join(content.casefold().split())


def rejection(proposal: ExperienceWriteProposal, reason: str) -> dict[str, Any]:
    tier = normalize_experience_tier(proposal.tier)
    return {
        "reason": reason,
        "tier": tier.value if tier is not None else str(proposal.tier),
    }


__all__ = ["ExperienceProposalValidator"]
