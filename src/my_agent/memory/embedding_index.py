"""Compatibility module alias for the Experience embedding index."""

from __future__ import annotations

import sys

from my_agent.memory.experience.retrieval import embedding_index as _embedding_index

sys.modules[__name__] = _embedding_index
