"""Compatibility module alias for formal maintenance tools."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.formal import tools as _tools

sys.modules[__name__] = _tools
