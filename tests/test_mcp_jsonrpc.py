from __future__ import annotations

import threading
import unittest

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.mcp.jsonrpc import JsonRpcClient, JsonRpcError


class InMemoryTransport:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.timeouts: list[int | None] = []
        self.listeners = []
        self.closed = False

    def send(self, message: dict[str, object], *, timeout_seconds: int | None = None) -> None:
        self.sent.append(message)
        self.timeouts.append(timeout_seconds)

    def on_receive(self, callback) -> None:
        self.listeners.append(callback)

    def emit(self, message: dict[str, object]) -> None:
        for listener in list(self.listeners):
            listener(message)

    def close(self) -> None:
        self.closed = True

    def stderr_lines(self) -> list[str]:
        return []

    def process_id(self) -> int | None:
        return None

    def transport_name(self) -> str:
        return "memory"


class JsonRpcClientTests(unittest.TestCase):
    def test_request_returns_matching_response_result(self) -> None:
        transport = InMemoryTransport()
        client = JsonRpcClient(transport)

        def respond() -> None:
            while not transport.sent:
                pass
            transport.emit({"jsonrpc": "2.0", "id": transport.sent[0]["id"], "result": {"ok": True}})

        thread = threading.Thread(target=respond)
        thread.start()
        result = client.request("ping", {}, timeout_seconds=1)
        thread.join(timeout=1)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(transport.sent[0]["method"], "ping")
        self.assertEqual(transport.timeouts[0], 1)

    def test_request_raises_jsonrpc_error(self) -> None:
        transport = InMemoryTransport()
        client = JsonRpcClient(transport)

        def respond() -> None:
            while not transport.sent:
                pass
            transport.emit({"jsonrpc": "2.0", "id": transport.sent[0]["id"], "error": {"code": -32601, "message": "missing"}})

        thread = threading.Thread(target=respond)
        thread.start()
        with self.assertRaises(JsonRpcError) as raised:
            client.request("missing", {}, timeout_seconds=1)
        thread.join(timeout=1)

        self.assertEqual(raised.exception.code, -32601)
        self.assertEqual(raised.exception.message, "missing")

    def test_request_times_out(self) -> None:
        client = JsonRpcClient(InMemoryTransport())

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            client.request("never", {}, timeout_seconds=0)

    def test_notification_is_forwarded_to_listeners(self) -> None:
        transport = InMemoryTransport()
        client = JsonRpcClient(transport)
        notifications: list[dict[str, object]] = []
        client.on_notification(notifications.append)

        transport.emit({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})

        self.assertEqual(notifications[0]["method"], "notifications/tools/list_changed")


if __name__ == "__main__":
    unittest.main()
