"""Compatibility module alias for legacy maintenance transactions."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.legacy import transaction as _transaction

sys.modules[__name__] = _transaction
