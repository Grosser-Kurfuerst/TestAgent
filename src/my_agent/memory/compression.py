"""Compatibility module alias for short-term memory compression."""

from __future__ import annotations

import sys

from my_agent.memory.short_term import compression as _compression

sys.modules[__name__] = _compression
