from __future__ import annotations

import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from threading import Event

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.mcp.transport import StdioTransport


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


if __name__ == "__main__":
    unittest.main()
