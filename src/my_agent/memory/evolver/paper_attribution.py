"""Compatibility module alias for paper-attribution equations."""

from __future__ import annotations

import sys

from my_agent.opd_data.attribution import equations as _equations

sys.modules[__name__] = _equations
