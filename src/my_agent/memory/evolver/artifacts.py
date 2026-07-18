"""Compatibility module alias for legacy maintenance artifacts."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.legacy import artifacts as _artifacts

sys.modules[__name__] = _artifacts
