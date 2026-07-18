"""Compatibility module alias for maintenance contracts."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance import contracts as _contracts

sys.modules[__name__] = _contracts
