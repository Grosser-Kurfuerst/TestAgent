"""Compatibility module alias for paper-attribution schemas."""

from __future__ import annotations

import sys

from my_agent.opd_data.attribution import schema as _schema

sys.modules[__name__] = _schema
