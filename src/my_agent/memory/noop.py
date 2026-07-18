"""Compatibility module alias for disabled memory."""

from __future__ import annotations

import sys

from my_agent.memory import disabled as _disabled

sys.modules[__name__] = _disabled
