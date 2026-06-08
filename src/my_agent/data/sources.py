from __future__ import annotations

"""Dataset loading and offline seed rows for Phase 6 data builders."""

import gzip
import json
import os
from typing import Any

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def load_mbpp_rows(split: str) -> list[dict[str, Any]]:
    _ensure_hf_endpoint()
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        pass
    else:
        try:
            return list(load_dataset("google-research-datasets/mbpp", split=split))
        except Exception:
            pass

    import urllib.request

    url = f"{_hf_endpoint()}/datasets/google-research-datasets/mbpp/resolve/main/sanitized-mbpp.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "my-agent/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: ASYNC100
            rows = json.loads(resp.read().decode("utf-8"))
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected MBPP JSON format")
        return rows
    except Exception:
        return _offline_mbpp_rows()


def load_humaneval_rows(split: str) -> list[dict[str, Any]]:
    _ensure_hf_endpoint()
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        pass
    else:
        try:
            return list(load_dataset("openai/openai_humaneval", split=split))
        except Exception:
            pass

    import urllib.request

    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "my-agent/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: ASYNC100
            content = gzip.decompress(resp.read()).decode("utf-8")
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    except Exception:
        return _offline_humaneval_rows()


def load_dataset_rows(name: str, split: str):
    _ensure_hf_endpoint()
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            f"The 'datasets' package is required to load {name!r}. "
            "Install data dependencies: uv sync --extra data"
        ) from exc
    try:
        return load_dataset(name, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {name!r} (split={split!r}) from HF endpoint {_hf_endpoint()!r}. "
            "Check network, HF_ENDPOINT env var, or cache the dataset manually."
        ) from exc


def _ensure_hf_endpoint() -> None:
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def _offline_mbpp_rows() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "offline_001",
            "text": "Write a function to subtract two numbers.",
            "code": "def subtract(a: int, b: int) -> int:\n    return a - b\n",
            "test_list": ["assert subtract(5, 3) == 2", "assert subtract(-1, -3) == 2"],
        },
        {
            "task_id": "offline_002",
            "text": "Write a function to return the square of a number.",
            "code": "def square(n: float) -> float:\n    return n * n\n",
            "test_list": ["assert square(4) == 16", "assert square(-3) == 9"],
        },
        {
            "task_id": "offline_003",
            "text": "Write a function that checks whether a string is a palindrome.",
            "code": "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n",
            "test_list": [
                "assert is_palindrome('racecar') is True",
                "assert is_palindrome('hello') is False",
            ],
        },
    ]


def _offline_humaneval_rows() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "HumanEval/offline_001",
            "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return a plus b.\"\"\"\n",
            "canonical_solution": "    return a + b\n",
            "test": (
                "def check(candidate):\n"
                "    assert candidate(2, 3) == 5\n"
                "    assert candidate(-1, 1) == 0\n"
            ),
            "entry_point": "add",
        },
        {
            "task_id": "HumanEval/offline_002",
            "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
            "    \"\"\"Check whether any two numbers in the list are closer than threshold.\"\"\"\n",
            "canonical_solution": (
                "    for i in range(len(numbers)):\n"
                "        for j in range(i + 1, len(numbers)):\n"
                "            if abs(numbers[i] - numbers[j]) < threshold:\n"
                "                return True\n"
                "    return False\n"
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate([1.0, 2.0, 3.0], 0.5) is False\n"
                "    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) is True\n"
            ),
            "entry_point": "has_close_elements",
        },
    ]


__all__ = [
    "DEFAULT_HF_ENDPOINT",
    "load_dataset_rows",
    "load_humaneval_rows",
    "load_mbpp_rows",
]
