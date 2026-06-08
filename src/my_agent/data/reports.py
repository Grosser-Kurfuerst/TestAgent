from __future__ import annotations

"""Build reports for auditable data-pipeline steps."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildReport:
    """Summary after building or converting SFT data."""

    source: str
    total: int = 0
    written: int = 0
    skipped: int = 0
    errors: tuple[str, ...] = ()
    tasks_path: Path | None = None
    sft_path: Path | None = None
    repo_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def count(self) -> int:
        """Compatibility read-only alias for callers that still display counts."""

        return self.written

    def render(self) -> str:
        lines = [
            f"[{self.source}] read {self.total} record(s), wrote {self.written} sample(s)",
        ]
        if self.skipped:
            lines.append(f"  skipped: {self.skipped}")
        if self.tasks_path:
            lines.append(f"  tasks: {self.tasks_path}")
        if self.sft_path:
            lines.append(f"  sft:   {self.sft_path}")
        if self.repo_dir:
            lines.append(f"  repos: {self.repo_dir}")
        for key, value in self.extra.items():
            lines.append(f"  {key}: {value}")
        if self.errors:
            lines.append(f"  errors: {len(self.errors)}")
            for message in self.errors[:5]:
                lines.append(f"    - {message}")
            if len(self.errors) > 5:
                lines.append(f"    - ... {len(self.errors) - 5} more")
        return "\n".join(lines)


__all__ = ["BuildReport"]
