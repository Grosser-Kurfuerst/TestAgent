from __future__ import annotations

from typing import Any


def repair_surrogates(text: str) -> str:
    """Return text that can be safely encoded as UTF-8."""
    if not _has_surrogate(text):
        return text

    # If stdin used surrogateescape under a non-UTF-8 locale, this reconstructs
    # the original UTF-8 bytes. Example: Chinese terminal input decoded as ASCII.
    try:
        return text.encode("utf-8", errors="surrogateescape").decode("utf-8")
    except UnicodeError:
        pass

    # Some inputs may contain UTF-16 surrogate pairs as two Python code points.
    try:
        return text.encode("utf-16", errors="surrogatepass").decode("utf-16")
    except UnicodeError:
        pass

    return text.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return repair_surrogates(value)
    if isinstance(value, dict):
        return {sanitize_json_value(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    return value


def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(char) <= 0xDFFF for char in text)
