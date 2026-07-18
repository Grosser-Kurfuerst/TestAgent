"""Compatibility module alias for the maintenance repository reducer."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance import repository_reducer as _repository_reducer

sys.modules[__name__] = _repository_reducer
