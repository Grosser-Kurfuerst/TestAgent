from __future__ import annotations

from dataclasses import dataclass
import itertools
import threading
from typing import Callable

from my_agent.mcp.transport import JsonMessage, McpTransport


class JsonRpcError(RuntimeError):
    def __init__(self, code: int, message: str, data: object | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass
class _PendingRequest:
    event: threading.Event
    result: object | None = None
    error: JsonRpcError | None = None


class JsonRpcClient:
    def __init__(self, transport: McpTransport) -> None:
        self.transport = transport
        self._ids = itertools.count(1)
        self._pending: dict[str, _PendingRequest] = {}
        self._pending_lock = threading.Lock()
        self._notification_callbacks: list[Callable[[JsonMessage], None]] = []
        self._notification_lock = threading.Lock()
        self.transport.on_receive(self._handle_message)

    def request(self, method: str, params: object | None = None, *, timeout_seconds: int = 60) -> object | None:
        request_id = next(self._ids)
        pending = _PendingRequest(event=threading.Event())
        with self._pending_lock:
            self._pending[str(request_id)] = pending
        message: JsonMessage = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            self.transport.send(message)
            if not pending.event.wait(timeout=max(0.001, timeout_seconds)):
                with self._pending_lock:
                    self._pending.pop(str(request_id), None)
                raise TimeoutError(f"JSON-RPC request timed out: {method}")
            if pending.error is not None:
                raise pending.error
            return pending.result
        except Exception:
            with self._pending_lock:
                self._pending.pop(str(request_id), None)
            raise

    def notify(self, method: str, params: object | None = None) -> None:
        message: JsonMessage = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.transport.send(message)

    def on_notification(self, callback: Callable[[JsonMessage], None]) -> None:
        with self._notification_lock:
            self._notification_callbacks.append(callback)

    def close(self) -> None:
        self.transport.close()

    def _handle_message(self, message: JsonMessage) -> None:
        if "id" not in message or message.get("id") is None:
            with self._notification_lock:
                callbacks = list(self._notification_callbacks)
            for callback in callbacks:
                callback(message)
            return

        request_id = str(message.get("id"))
        with self._pending_lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return

        raw_error = message.get("error")
        if isinstance(raw_error, dict):
            code = raw_error.get("code", -32603)
            pending.error = JsonRpcError(
                int(code) if isinstance(code, int) else -32603,
                str(raw_error.get("message") or "JSON-RPC error"),
                raw_error.get("data"),
            )
        else:
            pending.result = message.get("result")
        pending.event.set()
