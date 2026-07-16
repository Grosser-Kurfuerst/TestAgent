from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperiencePayload,
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryScope, content_fingerprint


def payload_for_tier(
    tier: ExperienceTier | str,
    content: str,
) -> ExperiencePayload:
    resolved = tier if isinstance(tier, ExperienceTier) else ExperienceTier(tier)
    if resolved == ExperienceTier.TRAJECTORY:
        return TrajectoryPayload(
            task_description=content,
            steps=(ExperienceTrajectoryStep(step_num=1, action="test", result=content, reward=1.0),),
            outcome="success",
            key_learnings=(content,),
            tags=("test",),
        )
    if resolved == ExperienceTier.TIP:
        return TipPayload(category="testing", severity="info", trigger=content)
    if resolved == ExperienceTier.SKILL:
        return SkillPayload(
            category="testing",
            technique=content,
            preconditions=(),
            steps=(content,),
        )
    return ToolPayload(name="test_tool", language="bash", code="echo test", repo_context=content)


def save_typed_experience(
    manager: Any,
    content: str,
    *,
    tier: ExperienceTier | str,
    payload: ExperiencePayload | None = None,
    **kwargs: Any,
):
    resolved = tier if isinstance(tier, ExperienceTier) else ExperienceTier(tier)
    return manager.save_experience(
        tier=resolved,
        content=content,
        payload=payload or payload_for_tier(resolved, content),
        **kwargs,
    )


def typed_experience(
    memory_id: str,
    content: str,
    tier: ExperienceTier = ExperienceTier.TIP,
    *,
    payload: ExperiencePayload | None = None,
    project_key: str = "/repo",
    scope: MemoryScope = MemoryScope.PROJECT,
    created_at: datetime | None = None,
    source_task: str = "",
    run_id: str = "",
    stream_id: str = "",
    created_by: ExperienceCreatedBy = ExperienceCreatedBy.MANUAL,
    writer_confidence: float = 1.0,
    attribution_value: float = 0.0,
    attribution_confidence: float = 0.0,
    candidate_count: int = 0,
    selected_count: int = 0,
    not_selected_count: int = 0,
    last_used: datetime | None = None,
    attribution_updated_at: datetime | None = None,
    protected: bool = False,
    invalidated: bool = False,
    promoted_to: str = "",
    maintenance_operation_id: str = "",
    parent_id: str = "",
    parent_tier: ExperienceTier | None = None,
) -> ExperienceMemory:
    return ExperienceMemory(
        id=memory_id,
        content=content,
        tier=tier,
        payload=payload or payload_for_tier(tier, content),
        scope=scope,
        project_key="" if scope == MemoryScope.GLOBAL else project_key,
        created_at=created_at or datetime.now(timezone.utc),
        token_count=estimate_tokens(content),
        fingerprint=content_fingerprint(content),
        source_task=source_task,
        run_id=run_id,
        stream_id=stream_id,
        created_by=created_by,
        writer_confidence=writer_confidence,
        attribution_value=attribution_value,
        attribution_confidence=attribution_confidence,
        candidate_count=candidate_count,
        selected_count=selected_count,
        not_selected_count=not_selected_count,
        last_used=last_used,
        attribution_updated_at=attribution_updated_at,
        protected=protected,
        invalidated=invalidated,
        promoted_to=promoted_to,
        maintenance_operation_id=maintenance_operation_id,
        parent_id=parent_id,
        parent_tier=parent_tier,
    )
