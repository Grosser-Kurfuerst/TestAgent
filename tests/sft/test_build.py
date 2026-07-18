from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pytest

from my_agent.opd_data.schema import (
    ActionDecisionEvidence,
    ActionExecutionEvidence,
    MaintenanceAttemptEvidence,
    MaintenanceEvidence,
    TaskEvidence,
    TaskOutcomeEvidence,
)
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.sft.build import (
    ExpertCorrection,
    SyntheticSFTRecord,
    build_canonical_sft_dataset,
    validate_semantic_sample,
)
from my_agent.sft.contracts import deterministic_tool_call_id
from my_agent.sft.manifest import SFTDatasetManifest
from my_agent.sft.semantic import SemanticSFTSample
from my_agent.training.contracts import DecisionEvent
from my_agent.training.role_views import (
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
    TaskOutcomeRef,
    TrajectoryEvidence,
)


def test_evidence_join_filters_failed_action_and_accepts_recovery(tmp_path: Path) -> None:
    fixture = _task_fixture(recovery=True)
    source = tmp_path / "evidence"
    _write_streams(source, **fixture)

    result = build_canonical_sft_dataset(
        source_dir=source,
        output_dir=tmp_path / "canonical",
    )

    samples = result.samples_by_split["train"]
    assert [sample.role for sample in samples].count("selection") == 1
    actions = [sample for sample in samples if sample.role == "action"]
    assert len(actions) == 1
    assert actions[0].metadata["source_id"] == "dec-action-2"
    assert actions[0].target.tool_calls[0].call_id == deterministic_tool_call_id(
        call_index=0,
        name="read_file",
        arguments={"path": "src/fix.py"},
    )
    assert result.manifest.filter_reason_counts["excluded_failed_action_execution"] == 1
    assert result.manifest.filter_reason_counts["accepted_error_recovery"] == 1
    assert result.manifest.quality_counts == {"accepted": 2, "excluded": 1}
    assert SFTDatasetManifest.from_dict(result.manifest.to_dict()) == result.manifest
    assert (tmp_path / "canonical" / "train.jsonl").exists()
    assert (tmp_path / "canonical" / "validation.jsonl").read_text() == ""


def test_expert_correction_accepts_failed_evidence_and_round_trips(tmp_path: Path) -> None:
    fixture = _task_fixture(recovery=False, execution_ok=False)
    source = tmp_path / "evidence"
    _write_streams(source, **fixture)
    corrected_call = _call("corrected-call", "read_file", {"path": "src/fix.py"})
    correction = ExpertCorrection.create(
        original_decision_id="dec-action-1",
        expected_output_kind="tool_call",
        expected_tool_call_count=1,
        target=CanonicalMessage(
            "assistant",
            "",
            tool_calls=(replace(
                corrected_call,
                call_id=deterministic_tool_call_id(
                    call_index=0,
                    name=corrected_call.name,
                    arguments=corrected_call.arguments_json,
                ),
            ),),
        ),
        annotator="reviewed-labeler-v1",
        license="internal-training",
    )
    assert ExpertCorrection.from_dict(correction.to_dict()) == correction

    result = build_canonical_sft_dataset(
        source_dir=source,
        output_dir=tmp_path / "canonical",
        corrections=(correction,),
    )

    corrected = [
        sample
        for sample in result.samples_by_split["train"]
        if sample.metadata["quality_status"] == "corrected"
    ]
    assert len(corrected) == 1
    assert result.manifest.quality_counts["corrected"] == 1


def test_error_recovery_finish_still_requires_resolved_outcome(tmp_path: Path) -> None:
    fixture = _task_fixture(recovery=True)
    finish_tool = _maintenance_tool()
    finish_call = _call("runtime-finish", "finish", {"summary": "Stopping early."})
    decisions = list(fixture["decisions"])
    decisions[-1] = replace(
        decisions[-1],
        canonical_tools=(finish_tool,),
        parsed_output={"tool_calls": [finish_call.to_dict()]},
    )
    tasks = list(fixture["tasks"])
    actions = list(tasks[0].action_decisions)
    actions[-1] = replace(
        actions[-1],
        tools=(finish_tool,),
        expected_tool_calls=(finish_call,),
    )
    tasks[0] = replace(tasks[0], action_decisions=tuple(actions))
    executions = list(fixture["executions"])
    executions[-1] = replace(
        executions[-1],
        call_id=finish_call.call_id,
        tool_name=finish_call.name,
        arguments_hash=canonical_sha256(json.loads(finish_call.arguments_json)),
    )
    outcomes = list(fixture["outcomes"])
    outcomes[0] = replace(
        outcomes[0],
        outcome=replace(outcomes[0].outcome, resolved=False, reward=0.0),
    )
    source = tmp_path / "evidence"
    _write_streams(
        source,
        decisions=tuple(decisions),
        tasks=tuple(tasks),
        outcomes=tuple(outcomes),
        executions=tuple(executions),
        exclusions=fixture["exclusions"],
        maintenance=fixture["maintenance"],
        attempts=fixture["attempts"],
    )

    result = build_canonical_sft_dataset(
        source_dir=source,
        output_dir=tmp_path / "canonical",
    )

    assert result.manifest.filter_reason_counts["excluded_finish_outcome"] == 1
    assert not any(
        sample.metadata.get("source_id") == "dec-action-2"
        for sample in result.samples_by_split["train"]
    )


def test_maintenance_requires_committed_or_noop_attempt(tmp_path: Path) -> None:
    fixture = _task_fixture()
    maintenance = _maintenance_fixture(status="abandoned")
    fixture["decisions"] = (*fixture["decisions"], maintenance["decision"])
    fixture["maintenance"] = (maintenance["evidence"],)
    fixture["attempts"] = (maintenance["attempt"],)
    source = tmp_path / "evidence"
    _write_streams(source, **fixture)

    rejected = build_canonical_sft_dataset(
        source_dir=source,
        output_dir=tmp_path / "rejected",
    )
    assert rejected.manifest.filter_reason_counts["excluded_nonterminal_maintenance"] == 1
    assert not any(
        sample.role == "maintenance"
        for sample in rejected.samples_by_split["train"]
    )

    committed = _maintenance_fixture(status="committed")
    fixture["decisions"] = (*fixture["decisions"][:-1], committed["decision"])
    fixture["maintenance"] = (committed["evidence"],)
    fixture["attempts"] = (committed["attempt"],)
    _write_streams(source, **fixture)
    accepted = build_canonical_sft_dataset(
        source_dir=source,
        output_dir=tmp_path / "accepted",
    )
    assert any(
        sample.role == "maintenance"
        for sample in accepted.samples_by_split["train"]
    )


def test_synthetic_contract_enforces_group_split_and_leakage(tmp_path: Path) -> None:
    fixture = _task_fixture()
    source = tmp_path / "evidence"
    _write_streams(source, **fixture)
    synthetic = SyntheticSFTRecord.create(
        sample=_synthetic_sample("project-b", "group-b"),
        split="validation",
        generator="schema-scenarios-v1",
        license="internal-training",
    )
    assert SyntheticSFTRecord.from_dict(synthetic.to_dict()) == synthetic
    result = build_canonical_sft_dataset(
        source_dir=source,
        output_dir=tmp_path / "canonical",
        synthetic_records=(synthetic,),
    )
    assert len(result.samples_by_split["validation"]) == 1

    crossing = SyntheticSFTRecord.create(
        sample=synthetic.sample,
        split="test",
        generator=synthetic.generator,
        license=synthetic.license,
    )
    with pytest.raises(ValueError, match="crosses dataset splits"):
        build_canonical_sft_dataset(
            source_dir=source,
            output_dir=tmp_path / "crossing",
            synthetic_records=(synthetic, crossing),
        )

    leaked = SemanticSFTSample.create(
        role="action",
        expected_output_kind="assistant_text",
        expected_tool_call_count=None,
        messages=(CanonicalMessage("user", "Inspect /home/alice/private/repo"),),
        tools=(),
        target=CanonicalMessage("assistant", "Done."),
        metadata=_metadata("leak", "project-c", "group-c"),
    )
    with pytest.raises(ValueError, match="privacy path"):
        validate_semantic_sample(leaked)


def _task_fixture(
    *,
    recovery: bool = False,
    execution_ok: bool = True,
) -> dict[str, tuple[object, ...]]:
    identity = _identity()
    selection_payload = {
        "selected_skills": [],
        "selected_tips": [],
        "selected_tools": [],
        "selected_trajectories": [],
        "reasoning": "No candidate is needed.",
    }
    selection = _event(
        decision_id="dec-selection",
        role="selection",
        identity=identity,
        messages=(CanonicalMessage("user", "Select public candidates."),),
        tools=(),
        parsed_output=selection_payload,
        raw_completion=canonical_json_bytes(selection_payload).decode(),
    )
    tool = _tool("read_file")
    first_call = _call("runtime-call-1", "read_file", {"path": "missing.py"})
    first_messages = (CanonicalMessage("user", "Fix the task."),)
    first = _event(
        decision_id="dec-action-1",
        role="action",
        identity=identity,
        messages=first_messages,
        tools=(tool,),
        parsed_output={"tool_calls": [first_call.to_dict()]},
    )
    actions = [ActionDecisionEvidence(
        decision_id=first.decision_id,
        turn_index=0,
        step_index=0,
        prefix_messages=first_messages,
        tools=(tool,),
        expected_tool_calls=(first_call,),
        observation_messages=(),
    )]
    decisions = [selection, first]
    executions = [_execution(
        first,
        first_call,
        call_index=0,
        ok=execution_ok and not recovery,
        output="missing file" if recovery or not execution_ok else "content",
    )]
    if recovery:
        observation = CanonicalMessage("tool", "missing file", tool_call_id=first_call.call_id)
        second_call = _call("runtime-call-2", "read_file", {"path": "src/fix.py"})
        second_messages = (
            *first_messages,
            CanonicalMessage("assistant", "", tool_calls=(first_call,)),
            observation,
        )
        second = _event(
            decision_id="dec-action-2",
            role="action",
            identity=identity,
            messages=second_messages,
            tools=(tool,),
            parsed_output={"tool_calls": [second_call.to_dict()]},
            turn_index=1,
            step_index=1,
        )
        actions[0] = replace(actions[0], observation_messages=(observation,))
        actions.append(ActionDecisionEvidence(
            decision_id=second.decision_id,
            turn_index=1,
            step_index=1,
            prefix_messages=second_messages,
            tools=(tool,),
            expected_tool_calls=(second_call,),
            observation_messages=(),
        ))
        decisions.append(second)
        executions.append(_execution(
            second,
            second_call,
            call_index=0,
            ok=True,
            output="fixed content",
        ))
    task = TaskEvidence(
        collection_round=0,
        task_ordinal=1,
        split="train",
        task="Fix the task.",
        task_id="task-1",
        task_group="group-a",
        trajectory_id="trajectory-1",
        stream_id="stream-a",
        memory_project_key="project-a",
        policy_identity=identity,
        repository_snapshot_hash=canonical_sha256({"repository": 1}),
        candidate_snapshot_hash=canonical_sha256([]),
        candidates=(),
        selected_memory_ids=(),
        trajectory=TrajectoryEvidence("trajectory-1", "group-a", "success", 1.0, ()),
        written_memory_ids=(),
        selection_decision_id=selection.decision_id,
        action_decisions=tuple(actions),
        writing_decision_id=None,
        selection_token_budget=1024,
    )
    outcome = TaskOutcomeEvidence(
        collection_round=0,
        task_ordinal=1,
        trajectory_id=task.trajectory_id,
        stream_id=task.stream_id,
        memory_project_key=task.memory_project_key,
        outcome=TaskOutcomeRef(
            task.task_id,
            task.task_group,
            1.0,
            True,
            "pytest",
            "8",
            canonical_sha256({"evaluator": "pytest-8"}),
        ),
        task_valid=True,
        outcome_finalized=True,
    )
    return {
        "decisions": tuple(decisions),
        "tasks": (task,),
        "outcomes": (outcome,),
        "executions": tuple(executions),
        "exclusions": (),
        "maintenance": (),
        "attempts": (),
    }


def _maintenance_fixture(*, status: str) -> dict[str, object]:
    identity = _identity()
    cadence_id = canonical_sha256({"cadence": 1})
    repository_hash = canonical_sha256({"repository": "maintenance"})
    attempt = MaintenanceAttemptEvidence(
        collection_round=0,
        split="train",
        cadence_id=cadence_id,
        attempt_index=1,
        status=status,
        task_group="maintenance-group",
        stream_id="stream-a",
        memory_project_key="maintenance-project",
        repository_snapshot_hash=repository_hash,
        as_of_task_ordinal=1,
        outcome_ids=(),
        redundancy_diagnostics=(),
        decision_ids=("dec-maintenance",),
        reason="done",
    )
    tool = _maintenance_tool()
    runtime_call = _call("runtime-maintenance", "finish", {"summary": "No changes."})
    decision = DecisionEvent(
        role="maintenance",
        purpose="fast_loop_evidence",
        decision_id="dec-maintenance",
        trajectory_id=cadence_id,
        turn_index=0,
        step_index=0,
        task_id=cadence_id,
        task_group="maintenance-group",
        stream_id="stream-a",
        memory_project_key="maintenance-project",
        run_id=attempt.attempt_id,
        policy_identity=identity,
        repository_revision="revision-1",
        candidate_snapshot_hash=repository_hash,
        canonical_messages=(CanonicalMessage("user", "Finish maintenance."),),
        canonical_tools=(tool,),
        rendered_prompt_hash=canonical_sha256({"prompt": "maintenance"}),
        prompt_token_ids=(1,),
        raw_completion="finish",
        completion_token_ids=(2,),
        assistant_loss_mask=(1,),
        parsed_output={
            "tool_call": {
                "call_id": runtime_call.call_id,
                "name": runtime_call.name,
                "arguments": json.loads(runtime_call.arguments_json),
            }
        },
        retry_of=None,
        status="success",
    )
    evidence = MaintenanceEvidence(
        collection_round=0,
        as_of_task_ordinal=1,
        split="train",
        cadence_id=cadence_id,
        attempt_id=attempt.attempt_id,
        task_group="maintenance-group",
        stream_id="stream-a",
        memory_project_key="maintenance-project",
        policy_identity=identity,
        repository_snapshot_hash=repository_hash,
        outcome_ids=(),
        tools=(tool,),
        redundancy_diagnostics=(),
        decision_ids=(decision.decision_id,),
    )
    return {"decision": decision, "evidence": evidence, "attempt": attempt}


def _event(
    *,
    decision_id: str,
    role: str,
    identity: PolicyIdentity,
    messages: tuple[CanonicalMessage, ...],
    tools: tuple[CanonicalTool, ...],
    parsed_output: dict[str, object],
    raw_completion: str = "tool call",
    turn_index: int = 0,
    step_index: int = 0,
) -> DecisionEvent:
    return DecisionEvent(
        role=role,
        purpose="fast_loop_evidence",
        decision_id=decision_id,
        trajectory_id="trajectory-1",
        turn_index=turn_index,
        step_index=step_index,
        task_id="task-1",
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        run_id="trajectory-1",
        policy_identity=identity,
        repository_revision="revision-1",
        candidate_snapshot_hash=canonical_sha256([]),
        canonical_messages=messages,
        canonical_tools=tools,
        rendered_prompt_hash=canonical_sha256({"prompt": decision_id}),
        prompt_token_ids=(1,),
        raw_completion=raw_completion,
        completion_token_ids=(2,),
        assistant_loss_mask=(1,),
        parsed_output=parsed_output,
        retry_of=None,
        status="success",
    )


def _execution(
    event: DecisionEvent,
    call: CanonicalToolCall,
    *,
    call_index: int,
    ok: bool,
    output: str,
) -> ActionExecutionEvidence:
    return ActionExecutionEvidence(
        collection_round=0,
        task_ordinal=1,
        split="train",
        task_id=event.task_id,
        task_group=event.task_group,
        decision_id=event.decision_id,
        trajectory_id=event.trajectory_id,
        stream_id=event.stream_id,
        memory_project_key=event.memory_project_key,
        run_id=event.run_id,
        policy_identity=event.policy_identity,
        turn_index=event.turn_index,
        step_index=event.step_index,
        call_index=call_index,
        call_id=call.call_id,
        tool_name=call.name,
        arguments_hash=canonical_sha256(json.loads(call.arguments_json)),
        ok=ok,
        blocked=False,
        error_code="" if ok else "not_found",
        output_hash=canonical_sha256(output),
    )


def _synthetic_sample(repository_key: str, task_group: str) -> SemanticSFTSample:
    tool = _tool("read_file")
    return SemanticSFTSample.create(
        role="action",
        expected_output_kind="tool_call",
        expected_tool_call_count=1,
        messages=(CanonicalMessage("user", "Inspect src/a.py"),),
        tools=(tool,),
        target=CanonicalMessage(
            "assistant",
            "",
            tool_calls=(replace(
                _call("placeholder", "read_file", {"path": "src/a.py"}),
                call_id=deterministic_tool_call_id(
                    call_index=0,
                    name="read_file",
                    arguments={"path": "src/a.py"},
                ),
            ),),
        ),
        metadata=_metadata("synthetic-1", repository_key, task_group),
    )


def _metadata(source_id: str, repository_key: str, task_group: str) -> dict[str, str]:
    return {
        "source": "schema_grounded_synthetic",
        "source_id": source_id,
        "task_group": task_group,
        "repository_key": repository_key,
        "quality_status": "accepted",
    }


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model",
        "revision",
        "sha256:" + "1" * 64,
        None,
        "tokenizer",
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
    )


def _tool(name: str) -> CanonicalTool:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    return CanonicalTool(
        name,
        f"Use {name}.",
        canonical_json_bytes(schema).decode(),
        canonical_sha256(schema),
    )


def _maintenance_tool() -> CanonicalTool:
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    return CanonicalTool(
        "finish",
        "Finish maintenance.",
        canonical_json_bytes(schema).decode(),
        canonical_sha256(schema),
    )


def _call(call_id: str, name: str, arguments: dict[str, object]) -> CanonicalToolCall:
    return CanonicalToolCall(
        call_id,
        name,
        canonical_json_bytes(arguments).decode(),
    )


def _write_streams(
    root: Path,
    *,
    decisions: tuple[object, ...],
    tasks: tuple[object, ...],
    outcomes: tuple[object, ...],
    executions: tuple[object, ...],
    exclusions: tuple[object, ...],
    maintenance: tuple[object, ...],
    attempts: tuple[object, ...],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, records in {
        "decision_events.jsonl": decisions,
        "task_evidence.jsonl": tasks,
        "task_outcomes.jsonl": outcomes,
        "tool_execution_evidence.jsonl": executions,
        "runtime_exclusions.jsonl": exclusions,
        "maintenance_evidence.jsonl": maintenance,
        "maintenance_attempts.jsonl": attempts,
    }.items():
        with (root / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json_bytes(record.to_dict()).decode() + "\n")
