from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from my_agent.cancellation import CancellationToken


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    output_reader_leaked: bool = False
    start_failed: str = ""


class BoundedOutputBuffer:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(1, max_chars)
        self._parts: list[str] = []
        self._chars = 0
        self._truncated = False
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            remaining = self.max_chars - self._chars
            if remaining > 0:
                chunk = text[:remaining]
                self._parts.append(chunk)
                self._chars += len(chunk)
            if len(text) > remaining and not self._truncated:
                self._parts.append("\n...(output truncated)")
                self._truncated = True

    def text(self) -> str:
        with self._lock:
            return "".join(self._parts)


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str],
    stdin_text: str | None = None,
    cancellation_token: CancellationToken | None = None,
    max_output_chars: int = 8_000,
) -> ProcessResult:
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return ProcessResult(returncode=None, stdout="", stderr="", start_failed=f"{type(exc).__name__}: {exc}")

    stdout_buffer = BoundedOutputBuffer(max_output_chars)
    stderr_buffer = BoundedOutputBuffer(max_output_chars)
    stdout_reader = threading.Thread(
        target=_drain_stream,
        args=(process.stdout, stdout_buffer),
        name="agentcli-stdout-reader",
        daemon=True,
    )
    stderr_reader = threading.Thread(
        target=_drain_stream,
        args=(process.stderr, stderr_buffer),
        name="agentcli-stderr-reader",
        daemon=True,
    )
    stdout_reader.start()
    stderr_reader.start()

    if stdin_text is not None and process.stdin is not None:
        try:
            process.stdin.write(stdin_text)
            process.stdin.close()
        except OSError:
            pass

    started = time.monotonic()
    timed_out = False
    cancelled = False
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            _terminate_process(process)
            break
        if time.monotonic() - started > timeout_seconds:
            timed_out = True
            _terminate_process(process)
            break
        time.sleep(0.1)

    if timed_out or cancelled:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()

    leaked = _join_reader(stdout_reader) or _join_reader(stderr_reader)
    return ProcessResult(
        returncode=process.returncode,
        stdout=stdout_buffer.text(),
        stderr=stderr_buffer.text(),
        timed_out=timed_out,
        cancelled=cancelled,
        output_reader_leaked=leaked,
    )


def _drain_stream(stream: object, buffer: BoundedOutputBuffer) -> None:
    if stream is None:
        return
    try:
        for chunk in iter(lambda: stream.read(1024), ""):
            if not chunk:
                break
            buffer.append(chunk)
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        pass


def _join_reader(thread: threading.Thread) -> bool:
    thread.join(timeout=2)
    return thread.is_alive()
