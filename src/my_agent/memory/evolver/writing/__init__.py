"""Writing contracts and legacy/formal Experience writer services."""

from my_agent.memory.evolver.writing.contracts import (
    ExperienceWriteProposal,
    ExperienceWriteRequest,
    ExperienceWriteResult,
    ExperienceWriteStep,
    ProposalGenerator,
)
from my_agent.memory.evolver.writing.dataset import MemoryWriterDatasetLogger
from my_agent.memory.evolver.writing.legacy import (
    ExperienceWriter,
    build_write_steps_from_tool_history,
    proposal_tier_counts,
    runtime_outcome_from_tool_records,
    writer_policy_for_result,
)
from my_agent.memory.evolver.writing.persistence import ExperienceRepositoryWriter
from my_agent.memory.evolver.writing.validation import ExperienceProposalValidator


def __getattr__(name: str):
    if name in {
        "FormalExperienceWriter",
        "build_writing_request",
        "parse_writing_response",
    }:
        from my_agent.memory.evolver.writing import formal

        return getattr(formal, name)
    raise AttributeError(name)

__all__ = [
    "ExperienceProposalValidator",
    "ExperienceRepositoryWriter",
    "ExperienceWriteProposal",
    "ExperienceWriteRequest",
    "ExperienceWriteResult",
    "ExperienceWriteStep",
    "ExperienceWriter",
    "FormalExperienceWriter",
    "MemoryWriterDatasetLogger",
    "ProposalGenerator",
    "build_write_steps_from_tool_history",
    "build_writing_request",
    "parse_writing_response",
    "proposal_tier_counts",
    "runtime_outcome_from_tool_records",
    "writer_policy_for_result",
]
