"""Compatibility module alias for the Experience embedding cache."""

from __future__ import annotations

import sys

from my_agent.memory.experience.retrieval import embedding_cache as _embedding_cache

sys.modules[__name__] = _embedding_cache
