"""Executable contracts shared by OPD paper-ablation workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from my_agent.policy.identity import canonical_sha256


PAPER_ABLATIONS = (
    "no_attribution",
    "similarity_only",
    "no_writing_distillation",
    "no_maintenance",
    "lexical_retrieval",
    "replay_d0_d1",
)

ABLATION_RECIPES: Mapping[str, Mapping[str, Any]] = {
    "no_attribution": {
        "attribution": "uniform_unscored",
        "selection": "llm",
        "writing_distillation": True,
        "maintenance": True,
        "retrieval": "embedding_cosine",
        "replay_datasets": [],
    },
    "similarity_only": {
        "attribution": "retrieval_score_mean",
        "selection": "similarity_only",
        "writing_distillation": True,
        "maintenance": True,
        "retrieval": "embedding_cosine",
        "replay_datasets": [],
    },
    "no_writing_distillation": {
        "attribution": "paper_eq_11_12",
        "selection": "llm",
        "writing_distillation": False,
        "maintenance": True,
        "retrieval": "embedding_cosine",
        "replay_datasets": [],
    },
    "no_maintenance": {
        "attribution": "paper_eq_11_12",
        "selection": "llm",
        "writing_distillation": True,
        "maintenance": False,
        "retrieval": "embedding_cosine",
        "replay_datasets": [],
    },
    "lexical_retrieval": {
        "attribution": "paper_eq_11_12",
        "selection": "llm",
        "writing_distillation": True,
        "maintenance": True,
        "retrieval": "lexical",
        "replay_datasets": [],
    },
    "replay_d0_d1": {
        "attribution": "paper_eq_11_12",
        "selection": "llm",
        "writing_distillation": True,
        "maintenance": True,
        "retrieval": "embedding_cosine",
        "replay_datasets": ["d0", "d1"],
    },
}

MAIN_ABLATION_RECIPE_HASH = canonical_sha256({})


def ablation_recipe(name: str) -> Mapping[str, Any]:
    normalized = str(name).strip().lower()
    if not normalized:
        return {}
    recipe = ABLATION_RECIPES.get(normalized)
    if recipe is None:
        raise ValueError(f"unsupported paper ablation: {normalized!r}")
    return recipe


def ablation_recipe_hash(name: str) -> str:
    return canonical_sha256(ablation_recipe(name))


def ablation_excluded_roles(name: str) -> frozenset[str]:
    normalized = str(name).strip().lower()
    if normalized == "no_writing_distillation":
        return frozenset({"writing"})
    if normalized == "no_maintenance":
        return frozenset({"maintenance"})
    return frozenset()


def ablation_uses_replay(name: str) -> bool:
    return str(name).strip().lower() == "replay_d0_d1"


__all__ = [
    "ABLATION_RECIPES",
    "MAIN_ABLATION_RECIPE_HASH",
    "PAPER_ABLATIONS",
    "ablation_excluded_roles",
    "ablation_recipe",
    "ablation_recipe_hash",
    "ablation_uses_replay",
]
