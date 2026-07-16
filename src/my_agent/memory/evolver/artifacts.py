"""Path derivation and isolation for maintenance transaction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from my_agent.memory.evolver.contracts import MaintenancePlanError


@dataclass(frozen=True)
class _MaintenanceArtifactGraph:
    """Internal set of paths that must remain isolated for one plan."""

    paths: tuple[tuple[str, Path], ...]

    def __post_init__(self) -> None:
        labels = [label for label, _ in self.paths]
        if len(labels) != len(set(labels)):
            raise MaintenancePlanError("maintenance artifact labels must be unique")

    @property
    def backup_path(self) -> Path | None:
        return self._path("backup")

    @property
    def reuse_plan_artifact(self) -> bool:
        return self._path("plan_artifact") is not None

    def _path(self, label: str) -> Path | None:
        return next((path for name, path in self.paths if name == label), None)


def _resolve_maintenance_artifact_graph(
    *,
    store_path: str | Path,
    store_lock_path: str | Path,
    history_path: str | Path,
    backup_dir: str | Path,
    plan_id: str | None,
    memory_dir: str | Path | None = None,
    attribution_path: str | Path | None = None,
    plan_input_path: str | Path | None = None,
    plan_output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    trace_path: str | Path | None = None,
) -> _MaintenanceArtifactGraph:
    """Resolve and validate the complete mutation-time artifact graph."""
    store = Path(store_path)
    history = Path(history_path)
    backup_root = Path(backup_dir)
    store_tmp = _atomic_write_tmp_path(store)
    history_lock = _history_lock_path(history)

    candidates: list[tuple[str, Path]] = [
        ("memory_store", store),
        ("memory_store_tmp", store_tmp),
        ("memory_lock", Path(store_lock_path)),
        ("history", history),
        ("history_lock", history_lock),
    ]
    _append_optional(candidates, "memory_directory", memory_dir)
    _append_optional(candidates, "backup_directory", backup_root)
    _append_optional(candidates, "attribution", attribution_path)
    _append_optional(candidates, "summary", summary_path)
    _append_optional(candidates, "trace", trace_path)

    backup: Path | None = None
    backup_tmp: Path | None = None
    if plan_id is not None:
        backup = _maintenance_backup_path(backup_root, plan_id)
        backup_tmp = _atomic_write_tmp_path(backup)
        candidates.extend((
            ("backup", backup),
            ("backup_tmp", backup_tmp),
        ))

    plan_input = Path(plan_input_path) if plan_input_path is not None else None
    plan_output = Path(plan_output_path) if plan_output_path is not None else None
    reuse_plan_artifact = bool(
        plan_input is not None
        and plan_output is not None
        and _artifact_paths_alias(plan_input, plan_output)
    )
    if reuse_plan_artifact:
        assert plan_output is not None
        candidates.append(("plan_artifact", plan_output))
    else:
        _append_optional(candidates, "plan_input", plan_input)
        _append_optional(candidates, "plan_output", plan_output)

    graph = _MaintenanceArtifactGraph(paths=tuple(candidates))
    _validate_maintenance_artifact_graph(graph)
    return graph


def _validate_maintenance_artifact_graph(graph: _MaintenanceArtifactGraph) -> None:
    """Recheck a previously resolved graph against current filesystem aliases."""
    _validate_pairwise_isolation(graph.paths)


def _maintenance_backup_path(backup_dir: str | Path, plan_id: str) -> Path:
    root = Path(backup_dir).resolve()
    candidate = (root / f"{plan_id}.experience_memory.jsonl").resolve()
    if candidate.parent != root:
        raise MaintenancePlanError("maintenance backup path escapes backup directory")
    return candidate


def _history_lock_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(f".{source.name}.lock")


def _atomic_write_tmp_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".tmp")


def _artifact_paths_alias(left: str | Path, right: str | Path) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    try:
        if left_path.resolve(strict=False) == right_path.resolve(strict=False):
            return True
        if left_path.exists() and right_path.exists() and left_path.samefile(right_path):
            return True
    except (OSError, RuntimeError) as exc:
        raise MaintenancePlanError(
            "maintenance artifact path cannot be resolved safely"
        ) from exc
    return False


def _append_optional(
    candidates: list[tuple[str, Path]],
    label: str,
    value: str | Path | None,
) -> None:
    if value is not None:
        candidates.append((label, Path(value)))


def _validate_pairwise_isolation(candidates: Iterable[tuple[str, Path]]) -> None:
    items = tuple(candidates)
    for index, (left_label, left_path) in enumerate(items):
        for right_label, right_path in items[index + 1:]:
            if _artifact_paths_alias(left_path, right_path):
                raise MaintenancePlanError(
                    "maintenance paths must be distinct: "
                    f"{left_label} conflicts with {right_label}"
                )


__all__: list[str] = []
