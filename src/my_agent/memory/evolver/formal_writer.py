"""Formal LLM-only writer invoked after authoritative task evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from math import isfinite
from typing import Any
from uuid import uuid4
import json

from my_agent.memory.evolver.serialization import experience_payload_to_dict
from my_agent.memory.evolver.repository_rules import experience_dedup_key
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.evolver.types import ExperienceCreatedBy, ExperienceMemory
from my_agent.memory.evolver.writer import (
    ExperienceWriteProposal,
    ExperienceWriteResult,
    ExperienceWriter,
)
from my_agent.memory.experience_store import ExperienceStore
from my_agent.memory.store_errors import MemoryStorePostCommitError, MemoryStoreRevisionConflict
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryScope, content_fingerprint
from my_agent.policy.contracts import DecisionRequest, DecisionResponse, GenerationPolicy
from my_agent.policy.identity import canonical_json_bytes
from my_agent.training.contracts import AuthoritativeTaskOutcome
from my_agent.training.decision_log import (
    DecisionAttemptError,
    DecisionEventContext,
    DecisionEventRecorder,
)
from my_agent.training.role_views import (
    CanonicalMessage,
    CanonicalTrajectoryStep,
    TrajectoryEvidence,
    WritingPublic,
)


class FormalExperienceWriter:
    def __init__(
        self,
        *,
        policy: GenerationPolicy,
        recorder: DecisionEventRecorder,
        store: ExperienceStore,
        project_key: str,
        min_confidence: float = 0.70,
        max_records: int = 6,
        max_content_chars: int = 1_200,
        max_new_tokens: int = 1_024,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> None:
        self.policy = policy
        self.recorder = recorder
        self.store = store
        self.project_key = project_key
        self.validator = ExperienceWriter(
            min_confidence=min_confidence,
            max_records=max_records,
            max_content_chars=max_content_chars,
        )
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def __call__(
        self,
        episode: AgentEpisodeArtifact,
        outcome: AuthoritativeTaskOutcome,
    ) -> ExperienceWriteResult:
        if not episode.task.strip():
            return ExperienceWriteResult(
                rejected=({"reason": "missing_task"},),
                llm_used=True,
            )
        public = _writing_public(episode, outcome)
        request = build_writing_request(
            public,
            max_records=self.validator.max_records,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        parsed_proposals: list[tuple[ExperienceWriteProposal, ...]] = []

        def parse_response(response: DecisionResponse) -> Mapping[str, Any]:
            proposals = parse_writing_response(
                _response_content(self.policy, response),
                validator=self.validator,
            )
            parsed_proposals.append(proposals)
            return {
                "fallback_used": False,
                "proposals": [_proposal_to_dict(item) for item in proposals],
            }

        context = DecisionEventContext(
            trajectory_id=episode.session.trajectory_id,
            turn_index=0,
            step_index=0,
            task_id=episode.session.task_id,
            task_group=episode.session.task_group,
            stream_id=episode.session.stream_id,
            memory_project_key=episode.session.memory_project_key,
            run_id=episode.session.trajectory_id,
            repository_revision=episode.session.repository_revision,
            candidate_snapshot_hash=episode.session.candidate_snapshot_hash,
        )
        try:
            self.recorder.generate(
                request,
                context=context,
                parse_response=parse_response,
            )
        except DecisionAttemptError as exc:
            return ExperienceWriteResult(
                rejected=({"reason": "invalid_or_failed_llm_output", "error": str(exc)},),
                llm_used=True,
                fallback_used=False,
            )

        proposals = parsed_proposals[0]
        snapshot = self.store.load_strict_snapshot()
        if snapshot.revision != episode.session.repository_revision:
            return ExperienceWriteResult(
                proposals=proposals,
                rejected=({"reason": "stale_repository_revision"},),
                llm_used=True,
                fallback_used=False,
                error="stale_repository_revision",
            )
        dedup_ids = {experience_dedup_key(memory): memory.id for memory in snapshot.memories}
        saved: list[ExperienceMemory] = []
        duplicate_ids: list[str] = []
        for proposal in proposals:
            entry = _memory_from_proposal(proposal, episode=episode, project_key=self.project_key)
            dedup_key = experience_dedup_key(entry)
            duplicate_id = dedup_ids.get(dedup_key)
            if duplicate_id is not None:
                duplicate_ids.append(duplicate_id)
                continue
            dedup_ids[dedup_key] = entry.id
            saved.append(entry)
        if saved:
            try:
                self.store.replace_all_atomically(
                    (*snapshot.memories, *saved),
                    expected_revision=episode.session.repository_revision,
                )
            except MemoryStoreRevisionConflict:
                return ExperienceWriteResult(
                    proposals=proposals,
                    rejected=({"reason": "stale_repository_revision"},),
                    llm_used=True,
                    fallback_used=False,
                    error="stale_repository_revision",
                )
            except MemoryStorePostCommitError as exc:
                try:
                    recovered = self.store.load_strict_snapshot()
                except Exception:
                    raise exc
                recovered_ids = {memory.id for memory in recovered.memories}
                if (
                    recovered.revision == exc.expected_revision
                    and all(memory.id in recovered_ids for memory in saved)
                ):
                    return ExperienceWriteResult(
                        proposals=proposals,
                        saved=tuple(saved),
                        duplicate_ids=tuple(duplicate_ids),
                        rejected=({"reason": "post_commit_audit_recovered"},),
                        llm_used=True,
                        fallback_used=False,
                    )
                raise
            except Exception as exc:  # noqa: BLE001 - atomic store failure must not become partial writes
                return ExperienceWriteResult(
                    proposals=proposals,
                    rejected=({"reason": "atomic_repository_write_failed"},),
                    llm_used=True,
                    fallback_used=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return ExperienceWriteResult(
            proposals=proposals,
            saved=tuple(saved),
            duplicate_ids=tuple(duplicate_ids),
            llm_used=True,
            fallback_used=False,
        )


def build_writing_request(
    public: WritingPublic,
    *,
    max_records: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> DecisionRequest:
    return DecisionRequest(
        role="writing",
        purpose="fast_loop_evidence",
        messages=(
            CanonicalMessage(
                "system",
                "Return only a JSON array of reusable memory records. Do not include prose or fallback content.",
            ),
            CanonicalMessage(
                "user",
                canonical_json_bytes({
                    "public_view": public.to_dict(),
                    "max_records": max_records,
                    "record_schema": {
                        "tier": "trajectory|tip|skill|tool",
                        "content": "reusable memory text",
                        "payload": {},
                        "confidence": 0.0,
                        "reason": "brief reason",
                    },
                }).decode("utf-8"),
            ),
        ),
        tools=(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def parse_writing_response(
    content: str,
    *,
    validator: ExperienceWriter,
) -> tuple[ExperienceWriteProposal, ...]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("writer output must be valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("writer output must be a JSON array")
    proposals: list[ExperienceWriteProposal] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("writer records must be JSON objects")
        expected = {"tier", "content", "payload", "confidence", "reason"}
        if set(item) != expected:
            raise ValueError("writer record fields do not match the formal schema")
        tier = item["tier"]
        content_value = item["content"]
        payload_value = item["payload"]
        confidence = item["confidence"]
        reason = item["reason"]
        if not isinstance(tier, str):
            raise ValueError("writer tier must be a string")
        if not isinstance(content_value, str):
            raise ValueError("writer content must be a string")
        if not isinstance(payload_value, Mapping):
            raise ValueError("writer payload must be a JSON object")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("writer confidence must be a number")
        if not isfinite(float(confidence)):
            raise ValueError("writer confidence must be finite")
        if not isinstance(reason, str):
            raise ValueError("writer reason must be a string")
        proposals.append(ExperienceWriteProposal(
            tier=tier,  # type: ignore[arg-type]
            content=content_value,
            payload=dict(payload_value),  # type: ignore[arg-type]
            confidence=float(confidence),
            reason=reason,
        ))
    accepted, rejected = validator.validate_proposals(proposals)
    if rejected:
        reasons = sorted({str(item.get("reason") or "invalid") for item in rejected})
        raise ValueError("writer output failed validation: " + ", ".join(reasons))
    return accepted


def _writing_public(
    episode: AgentEpisodeArtifact,
    outcome: AuthoritativeTaskOutcome,
) -> WritingPublic:
    steps = tuple(
        CanonicalTrajectoryStep(
            step_index=index,
            observation="",
            action=step.tool,
            arguments_json=canonical_json_bytes(step.arguments).decode("utf-8"),
            result=step.output,
            reward=None,
        )
        for index, step in enumerate(episode.tool_history)
    )
    trajectory = TrajectoryEvidence(
        trajectory_id=episode.session.trajectory_id,
        task_group=episode.session.task_group,
        outcome="success" if outcome.resolved else "failure",
        reward=outcome.reward,
        steps=steps,
    )
    return WritingPublic(
        task=episode.task,
        trajectory=trajectory,
        reward=outcome.reward,
        selected_memory_ids=episode.session.selected_memory_ids,
    )


def _memory_from_proposal(
    proposal: ExperienceWriteProposal,
    *,
    episode: AgentEpisodeArtifact,
    project_key: str,
) -> ExperienceMemory:
    return ExperienceMemory(
        id=f"exp-{uuid4().hex}",
        content=proposal.content,
        tier=proposal.tier,
        payload=proposal.payload,
        scope=MemoryScope.PROJECT,
        project_key=project_key,
        created_at=datetime.now(timezone.utc),
        token_count=estimate_tokens(proposal.content),
        fingerprint=content_fingerprint(proposal.content),
        source_task=episode.session.task_id,
        run_id=episode.session.trajectory_id,
        stream_id=episode.session.stream_id,
        created_by=ExperienceCreatedBy.WRITER,
        writer_confidence=proposal.confidence,
    )


def _proposal_to_dict(proposal: ExperienceWriteProposal) -> dict[str, Any]:
    return {
        "tier": proposal.tier.value,
        "content": proposal.content,
        "payload": experience_payload_to_dict(proposal.payload),
        "confidence": proposal.confidence,
        "reason": proposal.reason,
    }


def _response_content(policy: GenerationPolicy, response: DecisionResponse) -> str:
    chat_response = policy.chat_response_from_decision(response)
    content = getattr(chat_response, "content", None)
    if not isinstance(content, str):
        raise ValueError("formal writer response conversion did not produce text content")
    return content.strip()


__all__ = [
    "FormalExperienceWriter",
    "build_writing_request",
    "parse_writing_response",
]
