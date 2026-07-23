"""Build bounded, public-only memory episodes from benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

from my_agent.evaluation.manifest_benchmark import ManifestEvalResult
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    ACTION_LOG_SCHEMA_VERSION,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    PreparedBenchmarkTask,
    PublicEpisode,
)


DEFAULT_OBSERVATION_MAX_CHARS = 4_000
DEFAULT_EPISODE_MAX_CHARS = 12_000
_MIN_EPISODE_MAX_CHARS = 1_024
_MIN_CORE_TEXT_CHARS = 32
_MIN_COMMAND_CHARS = 16
_TRUNCATION_MARKER = "...[truncated]"
_PUBLIC_ACTION_FIELDS = (
    "schema_version",
    "sequence",
    "command",
    "returncode",
    "stdout",
    "stderr",
    "timed_out",
    "elapsed_sec",
)


def build_public_episode(
    prepared: PreparedBenchmarkTask,
    result: ManifestEvalResult,
    *,
    observation_max_chars: int = DEFAULT_OBSERVATION_MAX_CHARS,
    episode_max_chars: int = DEFAULT_EPISODE_MAX_CHARS,
) -> PublicEpisode:
    """Build the only episode representation allowed to reach a memory writer."""
    if not isinstance(prepared, PreparedBenchmarkTask):
        raise ValueError("prepared must be a PreparedBenchmarkTask")
    if not isinstance(result, ManifestEvalResult):
        raise ValueError("result must be a ManifestEvalResult")
    _require_positive_int(observation_max_chars, field_name="observation_max_chars")
    _require_positive_int(episode_max_chars, field_name="episode_max_chars")
    if episode_max_chars < _MIN_EPISODE_MAX_CHARS:
        raise ValueError(f"episode_max_chars must be at least {_MIN_EPISODE_MAX_CHARS}")
    if result.task_id != prepared.task.task_id:
        raise ValueError("manifest result does not match the prepared benchmark task")
    if not result.outcome_finalized:
        raise ValueError("public episode requires a finalized official outcome")

    actions = _read_public_actions(
        prepared.action_log_path,
        observation_max_chars=observation_max_chars,
    )
    episode = PublicEpisode(
        task_id=prepared.task.task_id,
        instruction=prepared.public_prompt,
        actions=actions,
        final_response=result.agent_final_answer,
        resolved=result.resolved,
        reward=result.reward,
        failure_type=result.failure_type,
    )
    return _fit_episode(episode, max_chars=episode_max_chars)


def public_episode_char_count(episode: PublicEpisode) -> int:
    """Return the canonical JSON character count used by the episode limit."""
    return len(
        json.dumps(
            episode.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _read_public_actions(
    path: Path,
    *,
    observation_max_chars: int,
) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        raise FileNotFoundError(f"final action log is missing: {path}")
    actions: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid action JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"action log line {line_number} must be a JSON object")
        actions.append(
            _public_action(
                payload,
                line_number=line_number,
                observation_max_chars=observation_max_chars,
            )
        )
    return tuple(actions)


def _public_action(
    payload: Mapping[str, Any],
    *,
    line_number: int,
    observation_max_chars: int,
) -> Mapping[str, Any]:
    missing = [field for field in _PUBLIC_ACTION_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            f"action log line {line_number} is missing public fields: {', '.join(missing)}"
        )
    if payload.get("schema_version") != ACTION_LOG_SCHEMA_VERSION:
        raise ValueError(f"action log line {line_number} has an unsupported schema")
    stdout, stderr = _bounded_observation(
        str(payload.get("stdout", "")),
        str(payload.get("stderr", "")),
        max_chars=observation_max_chars,
    )
    # Copy only the action wrapper's public v1 fields. Extra fields are never
    # propagated because adapters and evaluators may carry private metadata.
    return {
        "schema_version": ACTION_LOG_SCHEMA_VERSION,
        "sequence": payload["sequence"],
        "command": str(payload["command"]),
        "returncode": payload["returncode"],
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": payload["timed_out"],
        "elapsed_sec": payload["elapsed_sec"],
    }


def _bounded_observation(
    stdout: str, stderr: str, *, max_chars: int
) -> tuple[str, str]:
    if len(stdout) + len(stderr) <= max_chars:
        return stdout, stderr
    if stdout and stderr:
        stdout_budget = max_chars // 2
        stderr_budget = max_chars - stdout_budget
    elif stdout:
        stdout_budget, stderr_budget = max_chars, 0
    else:
        stdout_budget, stderr_budget = 0, max_chars
    return (
        _truncate_text(stdout, stdout_budget),
        _truncate_text(stderr, stderr_budget),
    )


def _fit_episode(episode: PublicEpisode, *, max_chars: int) -> PublicEpisode:
    actions = [dict(action) for action in episode.actions]
    candidate = episode
    while len(actions) > 1 and public_episode_char_count(candidate) > max_chars:
        actions.pop(0)
        candidate = _replace_episode(candidate, actions=actions)

    while public_episode_char_count(candidate) > max_chars:
        fields = _shrinkable_fields(candidate)
        if not fields:
            raise ValueError(
                "episode_max_chars is too small for the public episode schema"
            )
        field_name, action_index, value = max(fields, key=lambda item: len(item[2]))
        overflow = public_episode_char_count(candidate) - max_chars
        minimum = _minimum_text_chars(field_name)
        target = max(minimum, len(value) - overflow - len(_TRUNCATION_MARKER))
        shortened = _truncate_text(value, target)
        if shortened == value:
            shortened = value[: max(1, len(value) // 2)]
        candidate = _replace_text_field(
            candidate,
            field_name=field_name,
            action_index=action_index,
            value=shortened,
        )
    return candidate


def _shrinkable_fields(
    episode: PublicEpisode,
) -> list[tuple[str, int | None, str]]:
    fields: list[tuple[str, int | None, str]] = []
    if len(episode.instruction) > _MIN_CORE_TEXT_CHARS:
        fields.append(("instruction", None, episode.instruction))
    if len(episode.final_response) > _MIN_CORE_TEXT_CHARS:
        fields.append(("final_response", None, episode.final_response))
    for index, action in enumerate(episode.actions):
        for field_name in ("command", "stdout", "stderr"):
            value = str(action.get(field_name, ""))
            if len(value) > _minimum_text_chars(field_name):
                fields.append((field_name, index, value))
    return fields


def _replace_text_field(
    episode: PublicEpisode,
    *,
    field_name: str,
    action_index: int | None,
    value: str,
) -> PublicEpisode:
    if action_index is None:
        return PublicEpisode(
            task_id=episode.task_id,
            instruction=value if field_name == "instruction" else episode.instruction,
            actions=episode.actions,
            final_response=(
                value if field_name == "final_response" else episode.final_response
            ),
            resolved=episode.resolved,
            reward=episode.reward,
            failure_type=episode.failure_type,
        )
    actions = [dict(action) for action in episode.actions]
    actions[action_index][field_name] = value
    return _replace_episode(episode, actions=actions)


def _replace_episode(
    episode: PublicEpisode,
    *,
    actions: list[Mapping[str, Any]],
) -> PublicEpisode:
    return PublicEpisode(
        task_id=episode.task_id,
        instruction=episode.instruction,
        actions=tuple(actions),
        final_response=episode.final_response,
        resolved=episode.resolved,
        reward=episode.reward,
        failure_type=episode.failure_type,
    )


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 0:
        return ""
    if max_chars <= len(_TRUNCATION_MARKER):
        return value[:max_chars]
    return value[: max_chars - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _minimum_text_chars(field_name: str) -> int:
    if field_name in {"instruction", "final_response"}:
        return _MIN_CORE_TEXT_CHARS
    if field_name == "command":
        return _MIN_COMMAND_CHARS
    return 0


def _require_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "DEFAULT_EPISODE_MAX_CHARS",
    "DEFAULT_OBSERVATION_MAX_CHARS",
    "build_public_episode",
    "public_episode_char_count",
]
