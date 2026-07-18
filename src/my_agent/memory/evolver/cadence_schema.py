"""Compatibility module alias for the cadence schema."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.cadence import schema as _schema

sys.modules[__name__] = _schema
