from __future__ import annotations

from dataclasses import dataclass

from my_agent.opd_data.schema import (
    ActionDecisionEvidence,
    MaintenanceEvidence,
    RepositoryEvidence,
    RepositoryMemoryEvidence,
    TaskEvidence,
    TaskOutcomeEvidence,
)
from my_agent.memory.evolver.attribution_schema import CandidateExposure, PaperAttributionRecord
from my_agent.memory.evolver.paper_attribution import compute_round_attribution
from my_agent.policy.contracts import DecisionRequest, DecisionResponse, TokenBatch
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.training.contracts import DecisionEvent
from my_agent.training.role_views import (
    CandidateSnapshotEntry,
    CanonicalMessage,
    CanonicalTool,
    CanonicalTrajectoryStep,
    RepositorySnapshotRef,
    TaskOutcomeRef,
    TrajectoryEvidence,
)


def identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


def tool(name: str = "shell") -> CanonicalTool:
    parameters = {"properties": {"command": {"type": "string"}}, "type": "object"}
    return CanonicalTool(
        name,
        f"{name} tool",
        canonical_json_bytes(parameters).decode("utf-8"),
        canonical_sha256(parameters),
    )


@dataclass(frozen=True)
class RoundFixture:
    tasks: tuple[TaskEvidence, ...]
    outcomes: tuple[TaskOutcomeEvidence, ...]
    repositories: tuple[RepositoryEvidence, ...]
    maintenance: tuple[MaintenanceEvidence, ...]
    decisions: tuple[DecisionEvent, ...]
    attribution: tuple[PaperAttributionRecord, ...]


def round_fixture() -> RoundFixture:
    policy_identity = identity()
    candidate = CandidateSnapshotEntry(
        "RETRIEVED_SKILL_01",
        "mem-a",
        "skill",
        "Use the focused public test.",
        0.9,
        1,
        5,
    )
    candidate_hash = canonical_sha256([candidate.to_dict()])
    written_candidate = CandidateSnapshotEntry(
        "RETRIEVED_TIP_01",
        "mem-write",
        "tip",
        "Remember the public fix.",
        0.8,
        1,
        4,
    )
    written_candidate_hash = canonical_sha256([written_candidate.to_dict()])
    repository_revision = "rev-round-0"
    repository_snapshot = RepositorySnapshotRef(
        repository_revision,
        "project-a",
        ("mem-a",),
        canonical_sha256({"revision": repository_revision, "ids": ["mem-a"]}),
    )
    repository = RepositoryEvidence(
        collection_round=0,
        event_ordinal=0,
        stream_id="stream-a",
        previous_revision=None,
        snapshot=repository_snapshot,
        memories=(RepositoryMemoryEvidence("mem-a", "skill", candidate.content, 2, 1, "task-1"),),
    )
    maintenance_revision = "rev-round-1"
    maintenance_snapshot = RepositorySnapshotRef(
        maintenance_revision,
        "project-a",
        ("mem-a", "mem-write"),
        canonical_sha256({"revision": maintenance_revision, "ids": ["mem-a", "mem-write"]}),
    )
    maintenance_repository = RepositoryEvidence(
        collection_round=0,
        event_ordinal=1,
        stream_id="stream-a",
        previous_revision=repository_revision,
        snapshot=maintenance_snapshot,
        memories=(
            RepositoryMemoryEvidence("mem-a", "skill", candidate.content, 2, 1, "task-1"),
            RepositoryMemoryEvidence("mem-write", "tip", "Remember the public fix.", 2, 1, "task-1"),
        ),
    )
    trajectory_one = _trajectory("traj-1", "resolved", 1.0, "public success")
    trajectory_two = _trajectory("traj-2", "failed", 0.0, "public failure")
    trajectory_three = _trajectory("traj-3", "resolved", 1.0, "reused writer memory")
    trajectory_four = _trajectory("traj-4", "failed", 0.0, "counterfactual failure")
    task_one = TaskEvidence(
        collection_round=0,
        task_ordinal=1,
        split="train",
        task="Fix public task one",
        task_id="task-1",
        task_group="group-a",
        trajectory_id="traj-1",
        stream_id="stream-a",
        memory_project_key="project-a",
        policy_identity=policy_identity,
        repository_snapshot_hash=repository_snapshot.snapshot_hash,
        candidate_snapshot_hash=candidate_hash,
        candidates=(candidate,),
        selected_memory_ids=("mem-a",),
        trajectory=trajectory_one,
        written_memory_ids=("mem-write",),
        selection_decision_id="dec-selection-1",
        action_decisions=(ActionDecisionEvidence(
            decision_id="dec-action-1",
            turn_index=0,
            step_index=0,
            prefix_messages=(CanonicalMessage("user", "Fix public task one"),),
            tools=(tool(),),
            expected_tool_calls=(),
            observation_messages=(),
        ),),
        writing_decision_id="dec-writing-1",
        selection_token_budget=1_800,
    )
    task_two = TaskEvidence(
        collection_round=0,
        task_ordinal=2,
        split="train",
        task="Fix public task two",
        task_id="task-2",
        task_group="group-a",
        trajectory_id="traj-2",
        stream_id="stream-a",
        memory_project_key="project-a",
        policy_identity=policy_identity,
        repository_snapshot_hash=maintenance_snapshot.snapshot_hash,
        candidate_snapshot_hash=candidate_hash,
        candidates=(candidate,),
        selected_memory_ids=(),
        trajectory=trajectory_two,
        written_memory_ids=(),
        selection_decision_id="dec-selection-2",
        action_decisions=(ActionDecisionEvidence(
            decision_id="dec-action-2",
            turn_index=0,
            step_index=0,
            prefix_messages=(CanonicalMessage("user", "Fix public task two"),),
            tools=(tool(),),
            expected_tool_calls=(),
            observation_messages=(),
        ),),
        writing_decision_id=None,
        selection_token_budget=1_800,
    )
    task_three = TaskEvidence(
        collection_round=0,
        task_ordinal=3,
        split="train",
        task="Fix public task three",
        task_id="task-3",
        task_group="group-a",
        trajectory_id="traj-3",
        stream_id="stream-a",
        memory_project_key="project-a",
        policy_identity=policy_identity,
        repository_snapshot_hash=maintenance_snapshot.snapshot_hash,
        candidate_snapshot_hash=written_candidate_hash,
        candidates=(written_candidate,),
        selected_memory_ids=("mem-write",),
        trajectory=trajectory_three,
        written_memory_ids=(),
        selection_decision_id="dec-selection-3",
        action_decisions=(ActionDecisionEvidence(
            decision_id="dec-action-3",
            turn_index=0,
            step_index=0,
            prefix_messages=(CanonicalMessage("user", "Fix public task three"),),
            tools=(tool(),),
            expected_tool_calls=(),
            observation_messages=(),
        ),),
        writing_decision_id=None,
        selection_token_budget=1_800,
    )
    task_four = TaskEvidence(
        collection_round=0,
        task_ordinal=4,
        split="train",
        task="Fix public task four",
        task_id="task-4",
        task_group="group-a",
        trajectory_id="traj-4",
        stream_id="stream-a",
        memory_project_key="project-a",
        policy_identity=policy_identity,
        repository_snapshot_hash=maintenance_snapshot.snapshot_hash,
        candidate_snapshot_hash=written_candidate_hash,
        candidates=(written_candidate,),
        selected_memory_ids=(),
        trajectory=trajectory_four,
        written_memory_ids=(),
        selection_decision_id="dec-selection-4",
        action_decisions=(ActionDecisionEvidence(
            decision_id="dec-action-4",
            turn_index=0,
            step_index=0,
            prefix_messages=(CanonicalMessage("user", "Fix public task four"),),
            tools=(tool(),),
            expected_tool_calls=(),
            observation_messages=(),
        ),),
        writing_decision_id=None,
        selection_token_budget=1_800,
    )
    evaluator_hash = canonical_sha256({"evaluator": "pytest"})
    outcome_one = TaskOutcomeEvidence(
        0,
        1,
        "traj-1",
        "stream-a",
        "project-a",
        TaskOutcomeRef("task-1", "group-a", 1.0, True, "pytest", "8", evaluator_hash),
        True,
        True,
    )
    outcome_two = TaskOutcomeEvidence(
        0,
        2,
        "traj-2",
        "stream-a",
        "project-a",
        TaskOutcomeRef("task-2", "group-a", 0.0, False, "pytest", "8", evaluator_hash),
        True,
        True,
    )
    outcome_three = TaskOutcomeEvidence(
        0,
        3,
        "traj-3",
        "stream-a",
        "project-a",
        TaskOutcomeRef("task-3", "group-a", 1.0, True, "pytest", "8", evaluator_hash),
        True,
        True,
    )
    outcome_four = TaskOutcomeEvidence(
        0,
        4,
        "traj-4",
        "stream-a",
        "project-a",
        TaskOutcomeRef("task-4", "group-a", 0.0, False, "pytest", "8", evaluator_hash),
        True,
        True,
    )
    cadence_id = canonical_sha256({"cadence": 1})
    maintenance_attempt_id = canonical_sha256({"cadence": 1, "attempt": 1})
    maintenance_tool = tool("finish")
    maintenance = MaintenanceEvidence(
        collection_round=0,
        as_of_task_ordinal=4,
        split="train",
        cadence_id=cadence_id,
        attempt_id=maintenance_attempt_id,
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        policy_identity=policy_identity,
        repository_snapshot_hash=maintenance_snapshot.snapshot_hash,
        outcome_ids=(
            outcome_one.outcome_id,
            outcome_two.outcome_id,
            outcome_three.outcome_id,
            outcome_four.outcome_id,
        ),
        tools=(maintenance_tool,),
        redundancy_diagnostics=(),
        decision_ids=("dec-maintenance-1",),
    )
    decisions = (
        _decision("dec-selection-1", "selection", "traj-1", "task-1", candidate_hash, repository_revision),
        _decision("dec-action-1", "action", "traj-1", "task-1", candidate_hash, repository_revision),
        _decision("dec-writing-1", "writing", "traj-1", "task-1", candidate_hash, repository_revision),
        _decision("dec-selection-2", "selection", "traj-2", "task-2", candidate_hash, maintenance_revision),
        _decision("dec-action-2", "action", "traj-2", "task-2", candidate_hash, maintenance_revision),
        _decision(
            "dec-selection-3", "selection", "traj-3", "task-3",
            written_candidate_hash, maintenance_revision,
        ),
        _decision(
            "dec-action-3", "action", "traj-3", "task-3",
            written_candidate_hash, maintenance_revision,
        ),
        _decision(
            "dec-selection-4", "selection", "traj-4", "task-4",
            written_candidate_hash, maintenance_revision,
        ),
        _decision(
            "dec-action-4", "action", "traj-4", "task-4",
            written_candidate_hash, maintenance_revision,
        ),
        _decision(
            "dec-maintenance-1",
            "maintenance",
            cadence_id,
            cadence_id,
            maintenance_snapshot.snapshot_hash,
            maintenance_revision,
        ),
    )
    exposures = (
        _exposure("mem-a", "skill", 1, True, 1.0, candidate_hash),
        _exposure("mem-a", "skill", 2, False, 0.0, candidate_hash),
        _exposure("mem-write", "tip", 3, True, 1.0, written_candidate_hash),
        _exposure("mem-write", "tip", 4, False, 0.0, written_candidate_hash),
    )
    attribution = compute_round_attribution(
        exposures,
        collection_round=0,
        valid_task_ordinals=(1, 2, 3, 4),
    )
    return RoundFixture(
        tasks=(task_one, task_two, task_three, task_four),
        outcomes=(outcome_one, outcome_two, outcome_three, outcome_four),
        repositories=(repository, maintenance_repository),
        maintenance=(maintenance,),
        decisions=decisions,
        attribution=attribution,
    )


class FakeTrainablePolicy:
    def __init__(self) -> None:
        self._identity = identity()
        self.requests: list[DecisionRequest] = []

    def identity(self) -> PolicyIdentity:
        return self._identity

    def render_prompt_hash(self, request: DecisionRequest) -> str:
        return canonical_sha256({
            "messages": [item.to_dict() for item in request.messages],
            "tools": [item.to_dict() for item in request.tools],
        })

    def tokenize(self, request: DecisionRequest) -> TokenBatch:
        return TokenBatch(
            input_ids=((10, _role_token(request.role)),),
            attention_mask=((1, 1),),
            assistant_loss_mask=((0, 0),),
        )

    def generate_decision(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        completion = (100 + len(self.requests),)
        return DecisionResponse(
            raw_completion=f"completion-{len(self.requests)}",
            prompt_token_ids=(10, _role_token(request.role)),
            completion_token_ids=completion,
            assistant_loss_mask=(1,),
            parsed_tool_calls=(),
            identity=self._identity,
        )

    def forward_logits(self, batch: TokenBatch):
        return batch.input_ids

    def verify_completion_round_trip(self, response: DecisionResponse) -> bool:
        return bool(response.raw_completion) == bool(response.completion_token_ids)

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")

    def chat_response_from_decision(self, response):
        raise AssertionError("not used")


def _role_token(role: str) -> int:
    return {"selection": 1, "action": 2, "writing": 3, "maintenance": 4}[role]


def _trajectory(trajectory_id: str, outcome: str, reward: float, result: str) -> TrajectoryEvidence:
    return TrajectoryEvidence(
        trajectory_id,
        "group-a",
        outcome,
        reward,
        (CanonicalTrajectoryStep(0, "public observation", "shell", "{}", result, reward),),
    )


def _decision(
    decision_id: str,
    role: str,
    trajectory_id: str,
    task_id: str,
    candidate_snapshot_hash: str,
    repository_revision: str,
) -> DecisionEvent:
    canonical_tools = (
        (tool(),)
        if role == "action"
        else ((tool("finish"),) if role == "maintenance" else ())
    )
    task_messages = {
        "task-1": (CanonicalMessage("user", "Fix public task one"),),
        "task-2": (CanonicalMessage("user", "Fix public task two"),),
        "task-3": (CanonicalMessage("user", "Fix public task three"),),
        "task-4": (CanonicalMessage("user", "Fix public task four"),),
    }
    return DecisionEvent(
        role=role,
        purpose="fast_loop_evidence",
        decision_id=decision_id,
        trajectory_id=trajectory_id,
        turn_index=0,
        step_index=0,
        task_id=task_id,
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        run_id=(
            canonical_sha256({"cadence": 1, "attempt": 1})
            if role == "maintenance"
            else "round-0"
        ),
        policy_identity=identity(),
        repository_revision=repository_revision,
        candidate_snapshot_hash=candidate_snapshot_hash,
        canonical_messages=(
            task_messages[task_id]
            if role == "action"
            else (CanonicalMessage("user", "public"),)
        ),
        canonical_tools=canonical_tools,
        rendered_prompt_hash=canonical_sha256({"decision": decision_id}),
        prompt_token_ids=(1,),
        raw_completion="done",
        completion_token_ids=(2,),
        assistant_loss_mask=(1,),
        parsed_output=(
            {
                "tool_call": {
                    "call_id": "maintenance-finish",
                    "name": "finish",
                    "arguments": {"summary": "done"},
                }
            }
            if role == "maintenance"
            else {}
        ),
        retry_of=None,
        status="success",
    )


def _exposure(
    memory_id: str,
    tier: str,
    task_ordinal: int,
    selected: bool,
    reward: float,
    candidate_snapshot_hash: str,
) -> CandidateExposure:
    return CandidateExposure(
        task_id=f"task-{task_ordinal}",
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        memory_id=memory_id,
        tier=tier,
        selected=selected,
        reward=reward,
        collection_round=0,
        task_ordinal=task_ordinal,
        candidate_snapshot_hash=candidate_snapshot_hash,
        policy_identity=identity(),
        repository_revision="rev-round-0" if task_ordinal == 1 else "rev-round-1",
        evaluator_name="pytest",
        evaluator_version="8",
        evaluator_hash=canonical_sha256({"evaluator": "pytest"}),
    )


__all__ = ["FakeTrainablePolicy", "RoundFixture", "identity", "round_fixture", "tool"]
