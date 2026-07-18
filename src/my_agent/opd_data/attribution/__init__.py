"""Paper-faithful attribution schemas, equations, IO, and round assembly."""

from my_agent.opd_data.attribution.equations import (
    DEFAULT_PAPER_TIER_PRIORS,
    compute_memory_attribution,
    compute_round_attribution,
    confidence_gamma,
    memory_score,
    positive_selected_memory_ids,
    rho_g,
    teacher_memory_records,
    writing_top_fraction,
)
from my_agent.opd_data.attribution.io import (
    load_attribution_events,
    load_candidate_exposures,
    write_attribution_events,
    write_candidate_exposures,
)
from my_agent.opd_data.attribution.round import (
    ATTRIBUTION_EVENTS_FILENAME,
    CANDIDATE_EXPOSURES_FILENAME,
    RoundAttributionResult,
    build_round_attribution,
)
from my_agent.opd_data.attribution.schema import (
    PAPER_ATTRIBUTION_SCHEMA_VERSION,
    AttributionEvidenceRef,
    CandidateExposure,
    GroupAttribution,
    PaperAttributionRecord,
    WritingScoreDecision,
)

__all__ = [
    "ATTRIBUTION_EVENTS_FILENAME",
    "CANDIDATE_EXPOSURES_FILENAME",
    "DEFAULT_PAPER_TIER_PRIORS",
    "PAPER_ATTRIBUTION_SCHEMA_VERSION",
    "AttributionEvidenceRef",
    "CandidateExposure",
    "GroupAttribution",
    "PaperAttributionRecord",
    "RoundAttributionResult",
    "WritingScoreDecision",
    "build_round_attribution",
    "compute_memory_attribution",
    "compute_round_attribution",
    "confidence_gamma",
    "load_attribution_events",
    "load_candidate_exposures",
    "memory_score",
    "positive_selected_memory_ids",
    "rho_g",
    "teacher_memory_records",
    "write_attribution_events",
    "write_candidate_exposures",
    "writing_top_fraction",
]
