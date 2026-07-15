from __future__ import annotations


class MemoryStoreLockTimeout(RuntimeError):
    """Raised when a directory-scoped memory process lock cannot be acquired."""


class MemoryStoreLoadError(ValueError):
    """Raised when a strict memory repository snapshot is malformed or ambiguous."""


class MemoryStoreRevisionConflict(RuntimeError):
    """Raised when an atomic replacement was planned against a stale revision."""


class MemoryStorePostCommitError(RuntimeError):
    """Raised when replacement occurred but post-commit verification did not finish."""

    def __init__(self, message: str, *, expected_revision: str) -> None:
        super().__init__(message)
        self.expected_revision = expected_revision


__all__ = [
    "MemoryStoreLoadError",
    "MemoryStoreLockTimeout",
    "MemoryStorePostCommitError",
    "MemoryStoreRevisionConflict",
]
