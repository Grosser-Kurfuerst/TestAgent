"""Compatibility module alias for formal maintenance prompting."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.formal import prompt as _prompt

sys.modules[__name__] = _prompt
