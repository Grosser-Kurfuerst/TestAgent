from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess
import threading
from typing import Callable, Mapping, Protocol, Sequence
from urllib import request as urllib_request


JsonMessage = dict[str, object]
ReceiveCallback = Callable[[JsonMessage], None]
MCP_PROTOCOL_VERSION = "2024-11-05"


class McpTransport(Protocol):
    def send(self, message: JsonMessage, *, timeout_seconds: int | None = None) -> None:
        ...

    def on_receive(self, callback: ReceiveCallback) -> None:
        ...

    def close(self) -> None:
        ...

    def stderr_lines(self) -> list[str]:
        ...

    def process_id(self) -> int | None:
        ...

    def transport_name(self) -> str:
        ...


class StdioTransport:
    def __init__(
        self,
        command: str,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        command_line = [command, *(args or ())]
        process_env = None
        if env is not None:
            import os

            process_env = dict(os.environ)
            process_env.update(env)
        self._process = subprocess.Popen(
            command_line,
            cwd=str(cwd) if cwd is not None else None,
            env=process_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._callbacks: list[ReceiveCallback] = []
        self._callbacks_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stderr_ring: deque[str] = deque(maxlen=200)
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="agentcli-mcp-stdio-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="agentcli-mcp-stdio-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def send(self, message: JsonMessage, *, timeout_seconds: int | None = None) -> None:
        with self._send_lock:
            if self._closed:
                raise OSError("MCP stdio transport already closed.")
            if self._process.stdin is None:
                raise OSError("MCP stdio transport stdin is unavailable.")
            self._process.stdin.write(json.dumps(message, ensure_ascii=False))
            self._process.stdin.write("\n")
            self._process.stdin.flush()

    def on_receive(self, callback: ReceiveCallback) -> None:
        with self._callbacks_lock:
            self._callbacks.append(callback)

    def stderr_lines(self) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_ring)

    def process_id(self) -> int | None:
        return self._process.pid

    def transport_name(self) -> str:
        return "stdio"

    def close(self) -> None:
        self._closed = True
        try:
            stdin = self._process.stdin
            if stdin is not None and not stdin.closed:
                try:
                    stdin.close()
                except OSError:
                    pass
            try:
                self._process.wait(timeout=1)
                return
            except subprocess.TimeoutExpired:
                pass

            self._process.terminate()
            try:
                self._process.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                self._process.kill()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        finally:
            self._close_pipe(self._process.stdout)
            self._close_pipe(self._process.stderr)
            self._stdout_thread.join(timeout=0.2)
            self._stderr_thread.join(timeout=0.2)

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._append_stderr(f"[agentcli] invalid MCP stdout JSON: {exc}: {line.strip()}")
                    continue
                if not isinstance(message, dict):
                    self._append_stderr(f"[agentcli] ignored non-object MCP stdout message: {line.strip()}")
                    continue
                self._dispatch(message)
        except OSError as exc:
            self._append_stderr(f"[agentcli] stdout reader stopped: {exc}")

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                self._append_stderr(line.rstrip("\n"))
        except OSError as exc:
            self._append_stderr(f"[agentcli] stderr reader stopped: {exc}")

    def _dispatch(self, message: JsonMessage) -> None:
        with self._callbacks_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback(message)

    def _append_stderr(self, line: str) -> None:
        with self._stderr_lock:
            self._stderr_ring.append(line)

    @staticmethod
    def _close_pipe(stream: object) -> None:
        close = getattr(stream, "close", None)
        closed = getattr(stream, "closed", True)
        if close is not None and not closed:
            try:
                close()
            except OSError:
                pass


class StreamableHttpTransport:
    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        *,
        timeout_seconds: int = 60,
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.timeout_seconds = max(1, int(timeout_seconds or 60))
        self._callbacks: list[ReceiveCallback] = []
        self._callbacks_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stderr_ring: deque[str] = deque(maxlen=200)
        self._stderr_lock = threading.Lock()
        self._session_id = ""
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    def send(self, message: JsonMessage, *, timeout_seconds: int | None = None) -> None:
        data = json.dumps(message, ensure_ascii=False).encode("utf-8")
        with self._send_lock:
            if self._closed:
                raise OSError("MCP HTTP transport already closed.")
            headers = self._request_headers()
        request = urllib_request.Request(
            self.url,
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib_request.urlopen(  # noqa: S310 - user-configured MCP endpoint
            request,
            timeout=_positive_timeout(timeout_seconds, self.timeout_seconds),
        ) as response:
            session_id = response.headers.get("Mcp-Session-Id")
            body = response.read().decode("utf-8", errors="replace")
            content_type = response.headers.get("Content-Type", "")
        self._remember_session(session_id)
        self._handle_response(body, content_type)

    def on_receive(self, callback: ReceiveCallback) -> None:
        with self._callbacks_lock:
            self._callbacks.append(callback)

    def stderr_lines(self) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_ring)

    def process_id(self) -> int | None:
        return None

    def transport_name(self) -> str:
        return "http"

    def close(self) -> None:
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
            if not self._session_id:
                return
            headers = self._request_headers(include_content_type=False)
            request = urllib_request.Request(
                self.url,
                headers=headers,
                method="DELETE",
            )
        try:
            with urllib_request.urlopen(request, timeout=min(self.timeout_seconds, 1)):  # noqa: S310 - user-configured MCP endpoint
                pass
        except Exception as exc:  # noqa: BLE001 - close must not block shutdown
            self._append_stderr(f"[agentcli] MCP HTTP DELETE failed: {type(exc).__name__}: {exc}")

    def _request_headers(self, *, include_content_type: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            **self.headers,
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _remember_session(self, value: str | None) -> None:
        if value and value.strip():
            with self._send_lock:
                if not self._closed:
                    self._session_id = value.strip()

    def _handle_response(self, body: str, content_type: str) -> None:
        if not body.strip():
            return
        if "text/event-stream" in content_type.lower():
            for payload in _sse_data_messages(body):
                self._dispatch_json(payload)
            return
        self._dispatch_json(body)

    def _dispatch_json(self, payload: str) -> None:
        try:
            message = json.loads(payload)
        except json.JSONDecodeError as exc:
            self._append_stderr(f"[agentcli] invalid MCP HTTP JSON: {exc}: {payload.strip()}")
            return
        if isinstance(message, list):
            for item in message:
                if isinstance(item, dict):
                    self._dispatch(item)
                else:
                    self._append_stderr(f"[agentcli] ignored non-object MCP HTTP batch item: {item!r}")
            return
        if not isinstance(message, dict):
            self._append_stderr(f"[agentcli] ignored non-object MCP HTTP message: {payload.strip()}")
            return
        self._dispatch(message)

    def _dispatch(self, message: JsonMessage) -> None:
        with self._callbacks_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback(message)

    def _append_stderr(self, line: str) -> None:
        with self._stderr_lock:
            self._stderr_ring.append(line)


def _sse_data_messages(body: str) -> list[str]:
    messages: list[str] = []
    current: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            if current:
                messages.append("\n".join(current))
                current = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            current.append(line[5:].lstrip())
    if current:
        messages.append("\n".join(current))
    return messages


def _positive_timeout(value: int | None, default: int) -> int:
    if value is None:
        return max(1, int(default or 60))
    return max(1, int(value or default or 60))
