"""Compatibility module alias for disabled memory."""

from __future__ import annotations

import sys

import my_agent.memory.disabled as _disabled

sys.modules[__name__] = _disabled
