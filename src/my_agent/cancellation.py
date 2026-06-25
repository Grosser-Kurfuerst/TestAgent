from __future__ import annotations

import threading


class CancelledError(RuntimeError):
    """Raised when a run, batch, task, or tool observes cancellation."""


class CancellationToken:
    def __init__(self, parent: "CancellationToken | None" = None) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._parent = parent

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._event.is_set():
            self._reason = reason or "cancelled"
            self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set() or bool(self._parent and self._parent.is_cancelled())

    @property
    def reason(self) -> str:
        if self._event.is_set():
            return self._reason or "cancelled"
        if self._parent and self._parent.is_cancelled():
            return self._parent.reason
        return ""

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise CancelledError(self.reason or "cancelled")

    def child(self) -> "CancellationToken":
        return CancellationToken(parent=self)
