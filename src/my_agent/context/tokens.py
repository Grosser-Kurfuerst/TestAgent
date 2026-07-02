from __future__ import annotations

import json
from typing import Any


def estimate_tokens(value: Any) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)
