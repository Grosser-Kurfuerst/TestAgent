"""Canonical policy identity and hashing rules."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping
import json
import re


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
    "base_model",
    "base_revision",
    "checkpoint_hash",
    "adapter_hash",
    "tokenizer_revision",
    "tokenizer_hash",
    "chat_template_hash",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value using the single OPD canonical representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return f"sha256:{sha256(canonical_json_bytes(value)).hexdigest()}"


def require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must use sha256:<lowercase hex> format")


@dataclass(frozen=True)
class PolicyIdentity:
    base_model: str
    base_revision: str
    checkpoint_hash: str
    adapter_hash: str | None
    tokenizer_revision: str
    tokenizer_hash: str
    chat_template_hash: str

    def __post_init__(self) -> None:
        for field_name in ("base_model", "base_revision", "tokenizer_revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"policy identity {field_name} must not be empty")
        require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        if self.adapter_hash is not None:
            require_sha256(self.adapter_hash, field_name="adapter_hash")
        require_sha256(self.tokenizer_hash, field_name="tokenizer_hash")
        require_sha256(self.chat_template_hash, field_name="chat_template_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "checkpoint_hash": self.checkpoint_hash,
            "adapter_hash": self.adapter_hash,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_hash": self.tokenizer_hash,
            "chat_template_hash": self.chat_template_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyIdentity":
        _require_exact_fields(data, _IDENTITY_FIELDS, schema_name="policy identity")
        adapter_hash = data["adapter_hash"]
        if adapter_hash is not None and not isinstance(adapter_hash, str):
            raise ValueError("policy identity adapter_hash must be a string or null")
        return cls(
            base_model=_required_string(data["base_model"], "base_model"),
            base_revision=_required_string(data["base_revision"], "base_revision"),
            checkpoint_hash=_required_string(data["checkpoint_hash"], "checkpoint_hash"),
            adapter_hash=adapter_hash,
            tokenizer_revision=_required_string(data["tokenizer_revision"], "tokenizer_revision"),
            tokenizer_hash=_required_string(data["tokenizer_hash"], "tokenizer_hash"),
            chat_template_hash=_required_string(data["chat_template_hash"], "chat_template_hash"),
        )

    @property
    def identity_hash(self) -> str:
        return canonical_sha256(self.to_dict())


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_exact_fields(
    data: Mapping[str, Any],
    expected: tuple[str, ...],
    *,
    schema_name: str,
) -> None:
    missing = [name for name in expected if name not in data]
    unknown = sorted(set(data) - set(expected))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(f"invalid {schema_name} fields ({'; '.join(details)})")


__all__ = [
    "PolicyIdentity",
    "canonical_json_bytes",
    "canonical_sha256",
    "require_sha256",
]
