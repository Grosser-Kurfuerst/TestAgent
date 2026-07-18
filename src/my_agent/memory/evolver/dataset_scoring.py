"""Compatibility module alias for legacy dataset scoring."""

from __future__ import annotations

import sys

from my_agent.opd_data.legacy import dataset_scoring as _dataset_scoring

sys.modules[__name__] = _dataset_scoring
