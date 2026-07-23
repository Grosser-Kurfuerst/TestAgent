"""Validation for immutable memory-benchmark source locks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
import json
import re
import subprocess


SOURCE_LOCK_SCHEMA_VERSION = "memory-benchmark-source-lock-v1"
REQUIRED_SOURCES = frozenset({"lifelong_agent_bench", "intercode"})

_REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "url",
        "revision",
        "license",
        "task_data_path",
        "evaluator_entrypoint",
        "container_image",
        "container_digest",
    }
)
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PLACEHOLDER_MARKERS = ("<", ">", "todo", "tbd", "placeholder")
_OPTIONAL_REVISION_FIELDS = ("task_data_revision",)
_OPTIONAL_SHA256_FIELDS = (
    "container_base_digest",
    "container_build_sha256",
    "task_data_sha256",
)


class SourceLockError(ValueError):
    """Raised when a benchmark source lock is incomplete or mutable."""


def load_source_lock(
    path: str | Path,
    *,
    checkout_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Load and validate a memory-benchmark source lock JSON file."""

    lock_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SourceLockError(f"invalid source lock JSON: {lock_path}") from exc
    if not isinstance(payload, dict):
        raise SourceLockError("source lock root must be a JSON object")
    validate_source_lock(payload, checkout_roots=checkout_roots)
    return payload


def validate_source_lock(
    payload: Mapping[str, Any],
    *,
    checkout_roots: Mapping[str, str | Path] | None = None,
) -> None:
    """Validate source identity, data provenance, image digests, and checkout HEADs."""

    if payload.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
        raise SourceLockError(
            f"unsupported source lock schema: {payload.get('schema_version')!r}"
        )
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise SourceLockError("source lock requires a sources object")
    if set(sources) != REQUIRED_SOURCES:
        raise SourceLockError(
            "source lock must contain exactly lifelong_agent_bench and intercode"
        )

    for source_name in sorted(sources):
        source = sources[source_name]
        if not isinstance(source, Mapping):
            raise SourceLockError(f"source {source_name} must be an object")
        missing = sorted(_REQUIRED_SOURCE_FIELDS - set(source))
        if missing:
            raise SourceLockError(f"source {source_name} missing fields: {missing}")
        _validate_source(source_name, source)

    if checkout_roots is not None:
        unknown = sorted(set(checkout_roots) - set(sources))
        if unknown:
            raise SourceLockError(f"unknown checkout roots: {unknown}")
        for source_name, checkout_root in checkout_roots.items():
            _validate_checkout_head(
                source_name,
                Path(checkout_root).expanduser().resolve(),
                str(sources[source_name]["revision"]),
            )


def _validate_source(source_name: str, source: Mapping[str, Any]) -> None:
    url = _required_string(source, "url", source_name)
    _validate_url(url, field_name=f"{source_name}.url")

    revision = _required_string(source, "revision", source_name)
    _validate_revision(revision, field_name=f"{source_name}.revision")

    license_name = _required_string(source, "license", source_name)
    _reject_placeholder(license_name, field_name=f"{source_name}.license")

    task_data_path = _required_string(source, "task_data_path", source_name)
    _validate_relative_path(task_data_path, field_name=f"{source_name}.task_data_path")

    evaluator = _required_string(source, "evaluator_entrypoint", source_name)
    _reject_placeholder(evaluator, field_name=f"{source_name}.evaluator_entrypoint")

    image = _required_string(source, "container_image", source_name)
    _reject_placeholder(image, field_name=f"{source_name}.container_image")
    if image.casefold().endswith(":latest") or "@latest" in image.casefold():
        raise SourceLockError(f"{source_name}.container_image cannot use latest")

    digest = _required_string(source, "container_digest", source_name)
    _validate_sha256(digest, field_name=f"{source_name}.container_digest")

    task_data_url = source.get("task_data_url")
    if task_data_url is not None:
        if not isinstance(task_data_url, str) or not task_data_url.strip():
            raise SourceLockError(f"{source_name}.task_data_url must be non-empty")
        _validate_url(task_data_url, field_name=f"{source_name}.task_data_url")

    for field_name in _OPTIONAL_REVISION_FIELDS:
        value = source.get(field_name)
        if value is not None:
            if not isinstance(value, str):
                raise SourceLockError(f"{source_name}.{field_name} must be a string")
            _validate_revision(value, field_name=f"{source_name}.{field_name}")
    for field_name in _OPTIONAL_SHA256_FIELDS:
        value = source.get(field_name)
        if value is not None:
            if not isinstance(value, str):
                raise SourceLockError(f"{source_name}.{field_name} must be a string")
            _validate_sha256(value, field_name=f"{source_name}.{field_name}")

    build_path = source.get("container_build_path")
    if build_path is not None:
        if not isinstance(build_path, str) or not build_path.strip():
            raise SourceLockError(f"{source_name}.container_build_path must be non-empty")
        _validate_relative_path(
            build_path,
            field_name=f"{source_name}.container_build_path",
        )


def _required_string(source: Mapping[str, Any], field_name: str, source_name: str) -> str:
    value = source.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise SourceLockError(f"{source_name}.{field_name} must be non-empty")
    normalized = value.strip()
    _reject_placeholder(normalized, field_name=f"{source_name}.{field_name}")
    return normalized


def _reject_placeholder(value: str, *, field_name: str) -> None:
    normalized = value.casefold()
    if any(marker in normalized for marker in _PLACEHOLDER_MARKERS):
        raise SourceLockError(f"{field_name} contains a placeholder")


def _validate_url(value: str, *, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SourceLockError(f"{field_name} must be an absolute HTTP(S) URL")


def _validate_revision(value: str, *, field_name: str) -> None:
    if value.casefold() in {"main", "master", "head", "latest", "develop", "dev"}:
        raise SourceLockError(f"{field_name} cannot be a branch or moving revision")
    if _IMMUTABLE_REVISION.fullmatch(value) is None:
        raise SourceLockError(f"{field_name} must be a full immutable revision")


def _validate_sha256(value: str, *, field_name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise SourceLockError(f"{field_name} must be a full sha256 digest")


def _validate_relative_path(value: str, *, field_name: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise SourceLockError(f"{field_name} must be a normalized relative path")


def _validate_checkout_head(source_name: str, checkout_root: Path, revision: str) -> None:
    if not checkout_root.is_dir():
        raise SourceLockError(f"source checkout does not exist for {source_name}: {checkout_root}")
    result = subprocess.run(
        ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SourceLockError(f"source checkout is not a git repository: {checkout_root}")
    head = result.stdout.strip()
    if head != revision:
        raise SourceLockError(
            f"source checkout HEAD mismatch for {source_name}: expected {revision}, got {head}"
        )


__all__ = [
    "SOURCE_LOCK_SCHEMA_VERSION",
    "SourceLockError",
    "load_source_lock",
    "validate_source_lock",
]
