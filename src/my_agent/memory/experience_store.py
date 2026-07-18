"""Compatibility module alias for the typed Experience repository."""

from __future__ import annotations

import sys

from my_agent.memory.experience import repository as _repository

sys.modules[__name__] = _repository
