"""Compatibility module alias for legacy maintenance validation."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.legacy import validation as _validation

sys.modules[__name__] = _validation
