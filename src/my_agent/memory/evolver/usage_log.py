"""Compatibility module alias for legacy offline usage logs."""

from __future__ import annotations

import sys

from my_agent.opd_data.legacy import usage_log as _usage_log

sys.modules[__name__] = _usage_log
