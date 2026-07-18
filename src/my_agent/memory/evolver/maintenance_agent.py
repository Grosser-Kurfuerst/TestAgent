"""Compatibility module alias for the formal maintenance agent."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.formal import agent as _agent

sys.modules[__name__] = _agent
