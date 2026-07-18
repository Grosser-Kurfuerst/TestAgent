"""Compatibility module alias for paper-attribution JSONL IO."""

from __future__ import annotations

import sys

from my_agent.opd_data.attribution import io as _io

sys.modules[__name__] = _io
