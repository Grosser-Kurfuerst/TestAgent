"""Compatibility facade for the legacy weighted selector."""

from my_agent.memory.evolver.selection.contracts import (
    ExperienceCandidate,
    SelectedExperience,
    SelectionResult,
)
from my_agent.memory.evolver.selection.legacy import (
    ExperienceSelector,
    clamp,
    selection_candidate_summary,
    selection_score,
    selection_tier_counts,
)
from my_agent.memory.evolver.selection.rendering import render_selected_experiences

__all__ = [
    "ExperienceCandidate",
    "ExperienceSelector",
    "SelectedExperience",
    "SelectionResult",
    "clamp",
    "render_selected_experiences",
    "selection_candidate_summary",
    "selection_score",
    "selection_tier_counts",
]
