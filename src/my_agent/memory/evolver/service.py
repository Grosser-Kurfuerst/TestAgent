"""Compatibility module alias for legacy maintenance service."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.legacy import service as _service

sys.modules[__name__] = _service
