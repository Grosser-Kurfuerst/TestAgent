from __future__ import annotations

import threading
from typing import Callable


class EventBuffer:
    def __init__(self) -> None:
        self._events: list[object] = []
        self._lock = threading.Lock()

    def append(self, event: object) -> None:
        with self._lock:
            self._events.append(event)

    def flush_to(self, sink: Callable[[object], None] | None) -> None:
        if sink is None:
            return
        with self._lock:
            events = list(self._events)
            self._events.clear()
        for event in events:
            sink(event)


class BufferedEventSink:
    def __init__(self, sink: Callable[[object], None] | None) -> None:
        self._sink = sink
        self._buffers: dict[str, EventBuffer] = {}

    def buffer_for(self, key: str) -> EventBuffer:
        if key not in self._buffers:
            self._buffers[key] = EventBuffer()
        return self._buffers[key]

    def flush_in_order(self, keys: list[str]) -> None:
        for key in keys:
            self._buffers.get(key, EventBuffer()).flush_to(self._sink)
