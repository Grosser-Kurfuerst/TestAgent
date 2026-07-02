from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import tempfile
import threading
import textwrap
import time
import unittest
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.mcp.transport import StdioTransport, StreamableHttpTransport


class StdioTransportTests(unittest.TestCase):
    def test_send_receive_and_stderr_ring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "stdio_server.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys

                    for i in range(205):
                        print(f"err-{i}", file=sys.stderr, flush=True)

                    for line in sys.stdin:
                        message = json.loads(line)
                        print(json.dumps({"jsonrpc": "2.0", "id": message.get("id"), "result": {"method": message.get("method")}}), flush=True)
                    """
                ),
                encoding="utf-8",
            )
            transport = StdioTransport(sys.executable, [str(script)], cwd=tmp)
            received: list[dict[str, object]] = []
            event = Event()
            transport.on_receive(lambda message: (received.append(message), event.set()))

            try:
                transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
                self.assertTrue(event.wait(2))
                for _ in range(20):
                    if len(transport.stderr_lines()) >= 200:
                        break
                    time.sleep(0.05)
                stderr_lines = transport.stderr_lines()
            finally:
                transport.close()

        self.assertEqual(received[0]["result"], {"method": "ping"})
        self.assertEqual(len(stderr_lines), 200)
        self.assertEqual(stderr_lines[0], "err-5")
        self.assertEqual(stderr_lines[-1], "err-204")

    def test_close_terminates_process_that_ignores_stdin_eof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "sleep_server.py"
            script.write_text(
                "import time\nwhile True:\n    time.sleep(1)\n",
                encoding="utf-8",
            )
            transport = StdioTransport(sys.executable, [str(script)], cwd=tmp)

            transport.close()

            self.assertIsNotNone(transport._process.poll())


class StreamableHttpTransportTests(unittest.TestCase):
    def test_json_response_keeps_session_id_and_delete_close(self) -> None:
        server, thread, url = _start_http_server("json")
        transport = StreamableHttpTransport(url, headers={"Authorization": "Bearer secret"})
        received: list[dict[str, object]] = []
        transport.on_receive(received.append)
        try:
            transport.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            transport.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            transport.close()

            self.assertEqual(received[0]["id"], 1)
            self.assertEqual(received[1]["id"], 2)
            self.assertEqual(transport.session_id, "session-123")
            requests = server.requests  # type: ignore[attr-defined]
            self.assertEqual(requests[1]["headers"].get("Mcp-Session-Id"), "session-123")
            self.assertEqual(requests[0]["headers"].get("Authorization"), "Bearer secret")
            self.assertTrue(server.delete_seen.wait(1))  # type: ignore[attr-defined]
            delete_headers = _lower_headers(server.delete_headers)  # type: ignore[attr-defined]
            self.assertEqual(delete_headers.get("mcp-session-id"), "session-123")
            self.assertEqual(delete_headers.get("mcp-protocol-version"), "2024-11-05")
        finally:
            _stop_http_server(server, thread)

    def test_sse_response_dispatches_each_data_message(self) -> None:
        server, thread, url = _start_http_server("sse")
        transport = StreamableHttpTransport(url)
        received: list[dict[str, object]] = []
        transport.on_receive(received.append)
        try:
            transport.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

            self.assertEqual([message.get("method") for message in received], [None, "notifications/progress"])
            self.assertEqual(received[0]["result"], {"ok": True})
        finally:
            transport.close()
            _stop_http_server(server, thread)

    def test_close_does_not_wait_for_in_flight_post(self) -> None:
        entered = Event()
        release = Event()
        transport = StreamableHttpTransport("http://127.0.0.1:1/mcp", timeout_seconds=10)
        transport._session_id = "session-123"
        errors: list[BaseException] = []

        class _FakeResponse:
            headers = {}

            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *_args):  # type: ignore[no-untyped-def]
                return False

            def read(self) -> bytes:
                return b""

        def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
            if request.get_method() == "DELETE":
                return _FakeResponse()
            entered.set()
            release.wait(1)
            raise TimeoutError(f"blocked for timeout={timeout}")

        with patch("my_agent.mcp.transport.urllib_request.urlopen", side_effect=fake_urlopen):
            def run_send() -> None:
                try:
                    transport.send({"jsonrpc": "2.0", "id": 1, "method": "slow"}, timeout_seconds=7)
                except BaseException as exc:  # noqa: BLE001 - expected from fake urlopen
                    errors.append(exc)

            thread = threading.Thread(
                target=run_send,
                daemon=True,
            )
            thread.start()
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            transport.close()
            elapsed = time.monotonic() - started
            release.set()
            thread.join(timeout=1)

        self.assertLess(elapsed, 0.2)
        self.assertEqual(len(errors), 1)


class _McpHttpHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - http.server callback name
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = self.rfile.read(length).decode("utf-8")
        message = json.loads(payload)
        self.server.requests.append(  # type: ignore[attr-defined]
            {"method": message.get("method"), "headers": dict(self.headers)}
        )
        if self.server.response_mode == "sse":  # type: ignore[attr-defined]
            self._send_sse(message)
            return
        self._send_json(_response_for(message), session_id="session-123" if message.get("method") == "initialize" else "")

    def do_DELETE(self) -> None:  # noqa: N802 - http.server callback name
        self.server.delete_headers = dict(self.headers)  # type: ignore[attr-defined]
        self.server.delete_seen.set()  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()

    def _send_json(self, payload: dict[str, object], *, session_id: str = "") -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, message: dict[str, Any]) -> None:
        first = json.dumps({"jsonrpc": "2.0", "id": message.get("id"), "result": {"ok": True}})
        second = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"percent": 100}})
        body = f"data: {first}\n\ndata: {second}\n\n".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _response_for(message: dict[str, Any]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"method": message.get("method")}}


def _lower_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _start_http_server(mode: str) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _McpHttpHandler)
    server.response_mode = mode  # type: ignore[attr-defined]
    server.requests = []  # type: ignore[attr-defined]
    server.delete_seen = Event()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, name="mcp-http-test", daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}/mcp"


def _stop_http_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
