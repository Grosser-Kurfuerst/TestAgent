from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess
import threading
from typing import Callable, Mapping, Protocol, Sequence


JsonMessage = dict[str, object]
ReceiveCallback = Callable[[JsonMessage], None]


class McpTransport(Protocol):
    def send(self, message: JsonMessage) -> None:
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

    def send(self, message: JsonMessage) -> None:
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
