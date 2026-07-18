"""Compatibility module alias for legacy maintenance planning."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.legacy import planner as _planner

sys.modules[__name__] = _planner
