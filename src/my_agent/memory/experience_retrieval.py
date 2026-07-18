"""Compatibility module alias for lexical Experience retrieval."""

from __future__ import annotations

import sys

from my_agent.memory.experience.retrieval import lexical as _lexical

sys.modules[__name__] = _lexical
