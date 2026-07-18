"""Compatibility module alias for legacy offline attribution."""

from __future__ import annotations

import sys

from my_agent.opd_data.legacy import attribution as _attribution

sys.modules[__name__] = _attribution
