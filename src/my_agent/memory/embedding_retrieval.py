"""Compatibility module alias for embedding Experience retrieval."""

from __future__ import annotations

import sys

from my_agent.memory.experience.retrieval import embedding as _embedding

sys.modules[__name__] = _embedding
