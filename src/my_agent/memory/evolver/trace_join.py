"""Compatibility module alias for legacy trace joins."""

from __future__ import annotations

import sys

from my_agent.opd_data.legacy import trace_join as _trace_join

sys.modules[__name__] = _trace_join
