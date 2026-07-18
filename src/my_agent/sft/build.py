"""Evidence-joined construction of canonical SFT warm-start datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

from my_agent.memory.evolver.maintenance.formal.tools import parse_maintenance_tool_call
from my_agent.memory.evolver.selection.formal import parse_selection_response
from my_agent.opd_data.export import (
    load_action_execution_evidence,
    load_maintenance_attempts,
    load_maintenance_evidence,
    load_runtime_exclusions,
    load_task_evidence,
    load_task_outcomes,
)
from my_agent.opd_data.schema import (
    ActionDecisionEvidence,
    ActionExecutionEvidence,
    MaintenanceAttemptEvidence,
    MaintenanceEvidence,
    RuntimeExclusionEvidence,
    TaskEvidence,
    TaskOutcomeEvidence,
)
from my_agent.policy.identity import (
    canonical_json_bytes,
    canonical_sha256,
    require_matching_policy_identity,
)
from my_agent.sft.contracts import deterministic_tool_call_id
from my_agent.sft.manifest import SFTDatasetManifest
from my_agent.sft.semantic import SemanticSFTSample
from my_agent.tools.validation import validate_arguments_schema
from my_agent.training.contracts import DecisionEvent
from my_agent.training.decision_log import load_decision_events
from my_agent.training.role_views import (
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
    without_selected_memory_context,
)


EXPERT_CORRECTION_SCHEMA_VERSION = "agentcli-sft-expert-correction-v1"
SYNTHETIC_SAMPLE_SCHEMA_VERSION = "agentcli-sft-synthetic-v1"
_SPLITS = frozenset({"train", "validation", "test"})
_PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)"
)
_SECRET_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|"
    r"\b(?:sk|ghp|github_pat|hf)_[A-Za-z0-9_-]{16,}\b)",
    re.IGNORECASE,
)
_SECRET_FIELD_RE = re.compile(
    r'"(?:api[_-]?key|authorization|password|secret|credential|token)"\s*:\s*"(?!<redacted>)[^"\s]{8,}"',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExpertCorrection:
    original_decision_id: str
    expected_output_kind: str
    expected_tool_call_count: int | None
    target: CanonicalMessage
    annotator: str
    license: str
    correction_id: str
    schema_version: str = EXPERT_CORRECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPERT_CORRECTION_SCHEMA_VERSION:
            raise ValueError("unsupported expert correction schema")
        for value, field_name in (
            (self.original_decision_id, "original_decision_id"),
            (self.annotator, "annotator"),
            (self.license, "license"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"expert correction {field_name} must be non-empty")
        if self.target.role != "assistant":
            raise ValueError("expert correction target must be an assistant message")
        if self.correction_id != canonical_sha256(self.payload_without_id()):
            raise ValueError("expert correction ID mismatch")

    @classmethod
    def create(
        cls,
        *,
        original_decision_id: str,
        expected_output_kind: str,
        expected_tool_call_count: int | None,
        target: CanonicalMessage,
        annotator: str,
        license: str,
    ) -> "ExpertCorrection":
        values = {
            "original_decision_id": original_decision_id,
            "expected_output_kind": expected_output_kind,
            "expected_tool_call_count": expected_tool_call_count,
            "target": target,
            "annotator": annotator,
            "license": license,
        }
        return cls(**values, correction_id=canonical_sha256(_correction_payload(**values)))

    def payload_without_id(self) -> dict[str, Any]:
        return _correction_payload(
            original_decision_id=self.original_decision_id,
            expected_output_kind=self.expected_output_kind,
            expected_tool_call_count=self.expected_tool_call_count,
            target=self.target,
            annotator=self.annotator,
            license=self.license,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"correction_id": self.correction_id, **self.payload_without_id()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpertCorrection":
        expected = {
            "schema_version", "correction_id", "original_decision_id",
            "expected_output_kind", "expected_tool_call_count", "target",
            "annotator", "license",
        }
        if set(data) != expected:
            raise ValueError("expert correction fields do not match schema")
        target = data["target"]
        if not isinstance(target, Mapping):
            raise ValueError("expert correction target must be an object")
        count = data["expected_tool_call_count"]
        if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
            raise ValueError("expert correction tool-call count must be integer or null")
        return cls(
            schema_version=_required_string(data["schema_version"], "schema_version"),
            correction_id=_required_string(data["correction_id"], "correction_id"),
            original_decision_id=_required_string(
                data["original_decision_id"], "original_decision_id"
            ),
            expected_output_kind=_required_string(
                data["expected_output_kind"], "expected_output_kind"
            ),
            expected_tool_call_count=count,
            target=CanonicalMessage.from_dict(target),
            annotator=_required_string(data["annotator"], "annotator"),
            license=_required_string(data["license"], "license"),
        )


@dataclass(frozen=True)
class SyntheticSFTRecord:
    sample: SemanticSFTSample
    split: str
    generator: str
    license: str
    synthetic_id: str
    schema_version: str = SYNTHETIC_SAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SYNTHETIC_SAMPLE_SCHEMA_VERSION:
            raise ValueError("unsupported synthetic SFT schema")
        if self.split not in _SPLITS:
            raise ValueError("synthetic SFT split is invalid")
        for value, field_name in ((self.generator, "generator"), (self.license, "license")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"synthetic SFT {field_name} must be non-empty")
        if self.sample.metadata.get("source") != "schema_grounded_synthetic":
            raise ValueError("synthetic sample source must be schema_grounded_synthetic")
        if self.sample.metadata.get("quality_status") != "accepted":
            raise ValueError("synthetic sample quality_status must be accepted")
        if self.synthetic_id != canonical_sha256(self.payload_without_id()):
            raise ValueError("synthetic SFT ID mismatch")

    @classmethod
    def create(
        cls,
        *,
        sample: SemanticSFTSample,
        split: str,
        generator: str,
        license: str,
    ) -> "SyntheticSFTRecord":
        values = {
            "sample": sample,
            "split": split,
            "generator": generator,
            "license": license,
        }
        return cls(**values, synthetic_id=canonical_sha256(_synthetic_payload(**values)))

    def payload_without_id(self) -> dict[str, Any]:
        return _synthetic_payload(
            sample=self.sample,
            split=self.split,
            generator=self.generator,
            license=self.license,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"synthetic_id": self.synthetic_id, **self.payload_without_id()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SyntheticSFTRecord":
        expected = {
            "schema_version", "synthetic_id", "sample", "split", "generator", "license"
        }
        if set(data) != expected:
            raise ValueError("synthetic SFT fields do not match schema")
        sample = data["sample"]
        if not isinstance(sample, Mapping):
            raise ValueError("synthetic SFT sample must be an object")
        return cls(
            schema_version=_required_string(data["schema_version"], "schema_version"),
            synthetic_id=_required_string(data["synthetic_id"], "synthetic_id"),
            sample=SemanticSFTSample.from_dict(sample),
            split=_required_string(data["split"], "split"),
            generator=_required_string(data["generator"], "generator"),
            license=_required_string(data["license"], "license"),
        )


@dataclass(frozen=True)
class SFTBuildResult:
    samples_by_split: Mapping[str, tuple[SemanticSFTSample, ...]]
    manifest: SFTDatasetManifest


def build_canonical_sft_dataset(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    corrections: Sequence[ExpertCorrection] = (),
    synthetic_records: Sequence[SyntheticSFTRecord] = (),
    held_out_hashes: Iterable[str] = (),
) -> SFTBuildResult:
    """Join formal runtime streams and write canonical train/validation/test JSONL."""

    source = Path(source_dir)
    decisions = load_decision_events(source / "decision_events.jsonl")
    tasks = load_task_evidence(source / "task_evidence.jsonl")
    outcomes = load_task_outcomes(source / "task_outcomes.jsonl")
    executions = load_action_execution_evidence(source / "tool_execution_evidence.jsonl")
    exclusions = load_runtime_exclusions(source / "runtime_exclusions.jsonl")
    maintenance = load_maintenance_evidence(source / "maintenance_evidence.jsonl")
    attempts = load_maintenance_attempts(source / "maintenance_attempts.jsonl")

    builder = _EvidenceJoinedBuilder(
        decisions=decisions,
        tasks=tasks,
        outcomes=outcomes,
        executions=executions,
        exclusions=exclusions,
        maintenance=maintenance,
        maintenance_attempts=attempts,
        held_out_hashes=frozenset(held_out_hashes),
    )
    samples_by_split, quality_counts, reason_counts, group_splits = builder.build(
        corrections=corrections,
        synthetic_records=synthetic_records,
    )
    source_hashes = {
        "decision_events.jsonl": canonical_sha256([item.to_dict() for item in decisions]),
        "task_evidence.jsonl": canonical_sha256([item.to_dict() for item in tasks]),
        "task_outcomes.jsonl": canonical_sha256([item.to_dict() for item in outcomes]),
        "tool_execution_evidence.jsonl": canonical_sha256(
            [item.to_dict() for item in executions]
        ),
        "runtime_exclusions.jsonl": canonical_sha256(
            [item.to_dict() for item in exclusions]
        ),
        "maintenance_evidence.jsonl": canonical_sha256(
            [item.to_dict() for item in maintenance]
        ),
        "maintenance_attempts.jsonl": canonical_sha256(
            [item.to_dict() for item in attempts]
        ),
        "expert_corrections": canonical_sha256([item.to_dict() for item in corrections]),
        "synthetic_samples": canonical_sha256(
            [item.to_dict() for item in synthetic_records]
        ),
    }
    manifest = SFTDatasetManifest.create(
        samples_by_split=samples_by_split,
        quality_counts=quality_counts,
        filter_reason_counts=reason_counts,
        source_evidence_hashes=source_hashes,
        group_splits=group_splits,
    )
    _write_dataset(Path(output_dir), samples_by_split, manifest)
    return SFTBuildResult(samples_by_split=samples_by_split, manifest=manifest)


class _EvidenceJoinedBuilder:
    def __init__(
        self,
        *,
        decisions: Sequence[DecisionEvent],
        tasks: Sequence[TaskEvidence],
        outcomes: Sequence[TaskOutcomeEvidence],
        executions: Sequence[ActionExecutionEvidence],
        exclusions: Sequence[RuntimeExclusionEvidence],
        maintenance: Sequence[MaintenanceEvidence],
        maintenance_attempts: Sequence[MaintenanceAttemptEvidence],
        held_out_hashes: frozenset[str],
    ) -> None:
        self.decisions = _unique_index(decisions, lambda item: item.decision_id, "decision")
        self.tasks = tuple(tasks)
        self.outcomes = _unique_index(outcomes, _outcome_key, "task outcome")
        self.executions_by_decision: dict[str, list[ActionExecutionEvidence]] = defaultdict(list)
        for execution in executions:
            self.executions_by_decision[execution.decision_id].append(execution)
        self.exclusions = tuple(exclusions)
        self.maintenance = tuple(maintenance)
        self.attempts_by_id: dict[str, list[MaintenanceAttemptEvidence]] = defaultdict(list)
        for attempt in maintenance_attempts:
            self.attempts_by_id[attempt.attempt_id].append(attempt)
        self.held_out_hashes = held_out_hashes
        self.samples: dict[str, list[SemanticSFTSample]] = {
            split: [] for split in sorted(_SPLITS)
        }
        self.sample_ids: set[str] = set()
        self.group_splits: dict[str, str] = {}
        self.quality_counts: Counter[str] = Counter()
        self.reason_counts: Counter[str] = Counter()

    def build(
        self,
        *,
        corrections: Sequence[ExpertCorrection],
        synthetic_records: Sequence[SyntheticSFTRecord],
    ) -> tuple[
        dict[str, tuple[SemanticSFTSample, ...]],
        dict[str, int],
        dict[str, int],
        dict[str, str],
    ]:
        for task in self.tasks:
            self._add_task_samples(task)
        for evidence in self.maintenance:
            self._add_maintenance_samples(evidence)
        for correction in corrections:
            self._add_correction(correction)
        for record in synthetic_records:
            validate_semantic_sample(record.sample)
            self._accept(
                record.sample,
                split=record.split,
                group_key=_sample_group_key(record.sample),
                reason="accepted_schema_grounded_synthetic",
            )
        ordered = {
            split: tuple(sorted(samples, key=lambda item: item.sample_id))
            for split, samples in self.samples.items()
        }
        return (
            ordered,
            dict(sorted(self.quality_counts.items())),
            dict(sorted(self.reason_counts.items())),
            dict(sorted(self.group_splits.items())),
        )

    def _add_task_samples(self, task: TaskEvidence) -> None:
        outcome = self.outcomes.get(_task_outcome_key(task))
        if outcome is None:
            raise ValueError(f"task {task.task_id} is missing authoritative outcome evidence")
        _validate_task_outcome_join(task, outcome)
        selection = self._decision(task.selection_decision_id, task=task, role="selection")
        if self._excluded(selection, task):
            self._reject("excluded_runtime_selection")
        elif (
            selection.status == "success"
            and outcome.task_valid
            and outcome.outcome_finalized
            and outcome.outcome.resolved
        ):
            content = canonical_json_bytes(selection.parsed_output).decode("utf-8")
            parse_selection_response(content, candidates=task.candidates)
            self._accept_event_sample(
                event=selection,
                task=task,
                role="selection",
                expected_output_kind="selection_json",
                expected_tool_call_count=None,
                messages=selection.canonical_messages,
                tools=selection.canonical_tools,
                target=CanonicalMessage("assistant", content),
                reason="accepted_resolved_selection",
            )
        else:
            self._reject("excluded_selection_outcome")

        previous_failed: tuple[ActionExecutionEvidence, ...] = ()
        for action in task.action_decisions:
            event = self._decision(action.decision_id, task=task, role="action")
            _validate_action_decision_join(task, action, event)
            executions = tuple(sorted(
                self.executions_by_decision.get(event.decision_id, ()),
                key=lambda item: item.call_index,
            ))
            _validate_action_executions(task, event, action, executions)
            calls = action.expected_tool_calls
            if self._excluded(event, task):
                self._reject("excluded_runtime_action")
            elif not calls:
                if (
                    event.status == "success"
                    and event.raw_completion.strip()
                    and outcome.task_valid
                    and outcome.outcome_finalized
                    and outcome.outcome.resolved
                ):
                    self._accept_event_sample(
                        event=event,
                        task=task,
                        role="action",
                        expected_output_kind="assistant_text",
                        expected_tool_call_count=None,
                        messages=action.prefix_messages,
                        tools=action.tools,
                        target=CanonicalMessage("assistant", event.raw_completion.strip()),
                        reason="accepted_resolved_assistant_finish",
                    )
                else:
                    self._reject("excluded_unresolved_assistant_text")
            else:
                recovery = bool(previous_failed) and _failed_observations_are_present(
                    previous_failed,
                    action.prefix_messages,
                )
                has_finish = any(call.name == "finish" for call in calls)
                successful = len(executions) == len(calls) and all(item.ok for item in executions)
                outcome_allows_finish = (
                    outcome.task_valid
                    and outcome.outcome_finalized
                    and outcome.outcome.resolved
                )
                if has_finish and not outcome_allows_finish:
                    self._reject("excluded_finish_outcome")
                elif recovery or successful:
                    target = CanonicalMessage(
                        "assistant",
                        "",
                        tool_calls=_deterministic_calls(calls),
                    )
                    self._accept_event_sample(
                        event=event,
                        task=task,
                        role="action",
                        expected_output_kind="tool_call",
                        expected_tool_call_count=len(calls),
                        messages=action.prefix_messages,
                        tools=action.tools,
                        target=target,
                        reason=(
                            "accepted_error_recovery"
                            if recovery
                            else "accepted_successful_action"
                        ),
                    )
                elif executions and any(not item.ok for item in executions):
                    self._reject("excluded_failed_action_execution")
                else:
                    self._reject("excluded_incomplete_action_execution")
            previous_failed = tuple(item for item in executions if not item.ok)

        if task.writing_decision_id is not None:
            writing = self._decision(task.writing_decision_id, task=task, role="writing")
            proposals = writing.parsed_output.get("proposals")
            if self._excluded(writing, task):
                self._reject("excluded_runtime_writing")
            elif (
                writing.status == "success"
                and outcome.outcome_finalized
                and isinstance(proposals, list)
                and len(proposals) == len(task.written_memory_ids)
            ):
                content = canonical_json_bytes(proposals).decode("utf-8")
                _validate_writing_content(content)
                self._accept_event_sample(
                    event=writing,
                    task=task,
                    role="writing",
                    expected_output_kind="writing_json",
                    expected_tool_call_count=None,
                    messages=writing.canonical_messages,
                    tools=writing.canonical_tools,
                    target=CanonicalMessage("assistant", content),
                    reason="accepted_persisted_writing",
                )
            else:
                self._reject("excluded_inconsistent_writing_result")

    def _add_maintenance_samples(self, evidence: MaintenanceEvidence) -> None:
        terminal = tuple(
            item
            for item in self.attempts_by_id.get(evidence.attempt_id, ())
            if item.status in {"committed", "noop"}
        )
        if len(terminal) != 1:
            self._reject("excluded_nonterminal_maintenance", len(evidence.decision_ids))
            return
        attempt = terminal[0]
        _validate_maintenance_attempt_join(evidence, attempt)
        for decision_id in evidence.decision_ids:
            event = self.decisions.get(decision_id)
            if event is None:
                raise ValueError(f"maintenance decision is missing: {decision_id}")
            _validate_maintenance_event_join(evidence, event)
            if event.status != "success" or self._excluded_maintenance(event, evidence):
                self._reject("excluded_runtime_maintenance")
                continue
            call = _maintenance_call(event)
            target = CanonicalMessage(
                "assistant",
                "",
                tool_calls=_deterministic_calls((call,)),
            )
            parse_maintenance_tool_call(target.tool_calls)
            sample = SemanticSFTSample.create(
                role="maintenance",
                expected_output_kind="maintenance_tool_call",
                expected_tool_call_count=1,
                messages=event.canonical_messages,
                tools=event.canonical_tools,
                target=target,
                metadata={
                    "source": "formal_maintenance_evidence",
                    "source_id": event.decision_id,
                    "task_group": evidence.task_group,
                    "repository_key": evidence.memory_project_key,
                    "quality_status": "accepted",
                    "trajectory_id": event.trajectory_id,
                    "task_id": event.task_id,
                    "collection_round": evidence.collection_round,
                    "license": "runtime_generated",
                },
            )
            validate_semantic_sample(sample)
            self._accept(
                sample,
                split=evidence.split,
                group_key=_group_key(evidence.memory_project_key, evidence.task_group),
                reason="accepted_terminal_maintenance",
            )

    def _add_correction(self, correction: ExpertCorrection) -> None:
        event = self.decisions.get(correction.original_decision_id)
        if event is None:
            raise ValueError("expert correction references an unknown decision")
        task = next(
            (
                item
                for item in self.tasks
                if item.trajectory_id == event.trajectory_id and item.task_id == event.task_id
            ),
            None,
        )
        if task is None:
            raise ValueError("expert correction requires joined task evidence")
        executions = self.executions_by_decision.get(event.decision_id, ())
        explicitly_excluded = self._excluded(event, task)
        if event.status == "success" and not explicitly_excluded and not any(
            not item.ok for item in executions
        ):
            raise ValueError("expert correction must reference failed or excluded evidence")
        sample = SemanticSFTSample.create(
            role=event.role,
            expected_output_kind=correction.expected_output_kind,
            expected_tool_call_count=correction.expected_tool_call_count,
            messages=(
                without_selected_memory_context(event.canonical_messages)
                if event.role == "action"
                else event.canonical_messages
            ),
            tools=event.canonical_tools,
            target=correction.target,
            metadata={
                "source": "expert_correction",
                "source_id": correction.correction_id,
                "original_decision_id": event.decision_id,
                "task_group": event.task_group,
                "repository_key": event.memory_project_key,
                "quality_status": "corrected",
                "trajectory_id": event.trajectory_id,
                "task_id": event.task_id,
                "annotator": correction.annotator,
                "license": correction.license,
            },
        )
        validate_semantic_sample(sample)
        self._accept(
            sample,
            split=task.split,
            group_key=_group_key(task.memory_project_key, task.task_group),
            reason="accepted_expert_correction",
        )

    def _accept_event_sample(
        self,
        *,
        event: DecisionEvent,
        task: TaskEvidence,
        role: str,
        expected_output_kind: str,
        expected_tool_call_count: int | None,
        messages: tuple[CanonicalMessage, ...],
        tools: tuple[CanonicalTool, ...],
        target: CanonicalMessage,
        reason: str,
    ) -> None:
        sample = SemanticSFTSample.create(
            role=role,
            expected_output_kind=expected_output_kind,
            expected_tool_call_count=expected_tool_call_count,
            messages=messages,
            tools=tools,
            target=target,
            metadata={
                "source": "formal_decision_event",
                "source_id": event.decision_id,
                "task_group": task.task_group,
                "repository_key": task.memory_project_key,
                "quality_status": "accepted",
                "trajectory_id": task.trajectory_id,
                "task_id": task.task_id,
                "collection_round": task.collection_round,
                "task_evidence_id": task.evidence_id,
                "license": "runtime_generated",
            },
        )
        validate_semantic_sample(sample)
        self._accept(
            sample,
            split=task.split,
            group_key=_group_key(task.memory_project_key, task.task_group),
            reason=reason,
        )

    def _accept(
        self,
        sample: SemanticSFTSample,
        *,
        split: str,
        group_key: str,
        reason: str,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"unsupported SFT split: {split}")
        group_hash = canonical_sha256({"group_key": group_key})
        if sample.sample_id in self.held_out_hashes or group_hash in self.held_out_hashes:
            self._reject("excluded_held_out_hash")
            return
        previous_split = self.group_splits.get(group_key)
        if previous_split is not None and previous_split != split:
            raise ValueError("SFT group crosses dataset splits")
        self.group_splits[group_key] = split
        if sample.sample_id in self.sample_ids:
            self._reject("excluded_duplicate_sample")
            return
        self.sample_ids.add(sample.sample_id)
        self.samples[split].append(sample)
        quality = str(sample.metadata["quality_status"])
        self.quality_counts[quality] += 1
        self.reason_counts[reason] += 1

    def _reject(self, reason: str, count: int = 1) -> None:
        self.quality_counts["excluded"] += count
        self.reason_counts[reason] += count

    def _decision(self, decision_id: str, *, task: TaskEvidence, role: str) -> DecisionEvent:
        event = self.decisions.get(decision_id)
        if event is None:
            raise ValueError(f"task decision is missing: {decision_id}")
        _validate_task_event_join(task, event, role=role)
        return event

    def _excluded(self, event: DecisionEvent, task: TaskEvidence) -> bool:
        return any(
            item.collection_round == task.collection_round
            and item.split == task.split
            and item.task_id == task.task_id
            and item.trajectory_id == task.trajectory_id
            and item.stream_id == task.stream_id
            and item.memory_project_key == task.memory_project_key
            and item.role == event.role
            and (not item.decision_ids or event.decision_id in item.decision_ids)
            for item in self.exclusions
        )

    def _excluded_maintenance(
        self,
        event: DecisionEvent,
        evidence: MaintenanceEvidence,
    ) -> bool:
        return any(
            item.collection_round == evidence.collection_round
            and item.split == evidence.split
            and item.stream_id == evidence.stream_id
            and item.memory_project_key == evidence.memory_project_key
            and item.role == "maintenance"
            and (not item.decision_ids or event.decision_id in item.decision_ids)
            for item in self.exclusions
        )


def validate_semantic_sample(sample: SemanticSFTSample) -> None:
    """Apply parser, tool-schema, and leakage checks shared by every source."""

    tools = {tool.name: tool for tool in sample.tools}
    calls = [
        call
        for message in (*sample.messages, sample.target)
        for call in message.tool_calls
    ]
    for call in calls:
        tool = tools.get(call.name)
        if tool is None:
            raise ValueError(f"SFT sample references unknown tool: {call.name}")
        arguments = json.loads(call.arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("SFT tool arguments must be a JSON object")
        schema = json.loads(tool.parameters_json)
        if not isinstance(schema, dict):
            raise ValueError("SFT tool schema must be a JSON object")
        errors = validate_arguments_schema(schema, arguments)
        if errors:
            raise ValueError("SFT tool arguments failed schema validation: " + "; ".join(errors))
    if sample.expected_output_kind == "selection_json":
        _validate_selection_content(sample.target.content)
    elif sample.expected_output_kind == "writing_json":
        _validate_writing_content(sample.target.content)
    elif sample.expected_output_kind == "maintenance_tool_call":
        parse_maintenance_tool_call(sample.target.tool_calls)
    serialized = canonical_json_bytes(sample.to_dict()).decode("utf-8")
    if _PRIVATE_PATH_RE.search(serialized):
        raise ValueError("SFT sample contains an absolute user privacy path")
    if _SECRET_RE.search(serialized) or _SECRET_FIELD_RE.search(serialized):
        raise ValueError("SFT sample contains secret-like material")
    if "privileged_hindsight" in serialized or "teacher_only" in serialized:
        raise ValueError("SFT sample contains privileged hindsight fields")


def load_expert_corrections(path: str | Path) -> tuple[ExpertCorrection, ...]:
    return _load_jsonl(path, ExpertCorrection.from_dict)


def load_synthetic_records(path: str | Path) -> tuple[SyntheticSFTRecord, ...]:
    return _load_jsonl(path, SyntheticSFTRecord.from_dict)


def load_held_out_hashes(path: str | Path) -> frozenset[str]:
    hashes: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = line.strip()
            if not value:
                continue
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise ValueError(f"invalid held-out hash at line {line_number}")
            hashes.add(value)
    return frozenset(hashes)


def _validate_task_outcome_join(task: TaskEvidence, outcome: TaskOutcomeEvidence) -> None:
    if (
        task.collection_round != outcome.collection_round
        or task.task_ordinal != outcome.task_ordinal
        or task.trajectory_id != outcome.trajectory_id
        or task.stream_id != outcome.stream_id
        or task.memory_project_key != outcome.memory_project_key
        or task.task_id != outcome.outcome.task_id
        or task.task_group != outcome.outcome.task_group
    ):
        raise ValueError("task outcome evidence does not join task evidence")


def _validate_task_event_join(task: TaskEvidence, event: DecisionEvent, *, role: str) -> None:
    if event.role != role:
        raise ValueError("task decision role mismatch")
    if (
        event.trajectory_id != task.trajectory_id
        or event.task_id != task.task_id
        or event.task_group != task.task_group
        or event.stream_id != task.stream_id
        or event.memory_project_key != task.memory_project_key
    ):
        raise ValueError("decision event does not join task evidence")
    require_matching_policy_identity(task.policy_identity, event.policy_identity)


def _validate_action_decision_join(
    task: TaskEvidence,
    action: ActionDecisionEvidence,
    event: DecisionEvent,
) -> None:
    if event.status != "success":
        raise ValueError("task action evidence requires a successful decision event")
    if event.turn_index != action.turn_index or event.step_index != action.step_index:
        raise ValueError("action decision indexes do not match")
    if without_selected_memory_context(event.canonical_messages) != action.prefix_messages:
        raise ValueError("action decision public messages do not match task evidence")
    if event.canonical_tools != action.tools:
        raise ValueError("action decision tools do not match task evidence")
    if _action_calls(event) != action.expected_tool_calls:
        raise ValueError("action decision calls do not match task evidence")


def _validate_action_executions(
    task: TaskEvidence,
    event: DecisionEvent,
    action: ActionDecisionEvidence,
    executions: tuple[ActionExecutionEvidence, ...],
) -> None:
    by_index = {item.call_index: item for item in executions}
    if len(by_index) != len(executions):
        raise ValueError("action execution call indexes are duplicated")
    for index, call in enumerate(action.expected_tool_calls):
        execution = by_index.get(index)
        if execution is None:
            continue
        if (
            execution.collection_round != task.collection_round
            or execution.task_ordinal != task.task_ordinal
            or execution.split != task.split
            or execution.task_id != task.task_id
            or execution.task_group != task.task_group
            or execution.decision_id != event.decision_id
            or execution.trajectory_id != task.trajectory_id
            or execution.stream_id != task.stream_id
            or execution.memory_project_key != task.memory_project_key
            or execution.run_id != event.run_id
            or execution.turn_index != event.turn_index
            or execution.step_index != event.step_index
            or execution.call_id != call.call_id
            or execution.tool_name != call.name
            or execution.arguments_hash != canonical_sha256(json.loads(call.arguments_json))
        ):
            raise ValueError("action execution evidence does not join its decision")
        require_matching_policy_identity(event.policy_identity, execution.policy_identity)
    if any(index >= len(action.expected_tool_calls) for index in by_index):
        raise ValueError("action execution contains an unexpected call index")


def _validate_maintenance_attempt_join(
    evidence: MaintenanceEvidence,
    attempt: MaintenanceAttemptEvidence,
) -> None:
    if (
        evidence.collection_round != attempt.collection_round
        or evidence.split != attempt.split
        or evidence.cadence_id != attempt.cadence_id
        or evidence.attempt_id != attempt.attempt_id
        or evidence.task_group != attempt.task_group
        or evidence.stream_id != attempt.stream_id
        or evidence.memory_project_key != attempt.memory_project_key
        or evidence.repository_snapshot_hash != attempt.repository_snapshot_hash
        or evidence.as_of_task_ordinal != attempt.as_of_task_ordinal
        or evidence.outcome_ids != attempt.outcome_ids
        or evidence.decision_ids != attempt.decision_ids
    ):
        raise ValueError("maintenance attempt does not join maintenance evidence")


def _validate_maintenance_event_join(
    evidence: MaintenanceEvidence,
    event: DecisionEvent,
) -> None:
    if (
        event.role != "maintenance"
        or event.trajectory_id != evidence.cadence_id
        or event.task_id != evidence.cadence_id
        or event.task_group != evidence.task_group
        or event.stream_id != evidence.stream_id
        or event.memory_project_key != evidence.memory_project_key
        or event.run_id != evidence.attempt_id
    ):
        raise ValueError("maintenance decision does not join maintenance evidence")
    require_matching_policy_identity(evidence.policy_identity, event.policy_identity)


def _failed_observations_are_present(
    failures: tuple[ActionExecutionEvidence, ...],
    messages: tuple[CanonicalMessage, ...],
) -> bool:
    observations = {
        message.tool_call_id: canonical_sha256(message.content)
        for message in messages
        if message.role == "tool"
    }
    return all(observations.get(item.call_id) == item.output_hash for item in failures)


def _action_calls(event: DecisionEvent) -> tuple[CanonicalToolCall, ...]:
    value = event.parsed_output.get("tool_calls")
    if not isinstance(value, list):
        raise ValueError("action decision parsed_output.tool_calls must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError("action decision tool calls must be objects")
    return tuple(CanonicalToolCall.from_dict(item) for item in value)


def _maintenance_call(event: DecisionEvent) -> CanonicalToolCall:
    value = event.parsed_output.get("tool_call")
    if not isinstance(value, Mapping):
        raise ValueError("maintenance decision parsed_output.tool_call must be an object")
    arguments = value.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("maintenance decision arguments must be an object")
    return CanonicalToolCall(
        _required_string(value.get("call_id"), "call_id"),
        _required_string(value.get("name"), "name"),
        canonical_json_bytes(dict(arguments)).decode("utf-8"),
    )


def _deterministic_calls(calls: Sequence[CanonicalToolCall]) -> tuple[CanonicalToolCall, ...]:
    return tuple(
        CanonicalToolCall(
            deterministic_tool_call_id(
                call_index=index,
                name=call.name,
                arguments=call.arguments_json,
            ),
            call.name,
            call.arguments_json,
        )
        for index, call in enumerate(calls)
    )


def _validate_selection_content(content: str) -> None:
    payload = _json_mapping(content, "selection target")
    expected = {
        "selected_skills", "selected_tips", "selected_tools",
        "selected_trajectories", "reasoning",
    }
    if set(payload) != expected or not isinstance(payload["reasoning"], str):
        raise ValueError("selection target fields do not match the formal schema")
    for field_name in expected - {"reasoning"}:
        value = payload[field_name]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("selection target lists must contain strings")


def _validate_writing_content(content: str) -> None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("writing target must be valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("writing target must be a JSON array")
    expected = {"tier", "content", "payload", "confidence", "reason"}
    for item in payload:
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError("writing target fields do not match the formal schema")
        if (
            not isinstance(item["tier"], str)
            or not isinstance(item["content"], str)
            or not isinstance(item["payload"], Mapping)
            or isinstance(item["confidence"], bool)
            or not isinstance(item["confidence"], (int, float))
            or not isinstance(item["reason"], str)
        ):
            raise ValueError("writing target field types do not match the formal schema")


def _sample_group_key(sample: SemanticSFTSample) -> str:
    return _group_key(
        _required_string(sample.metadata.get("repository_key"), "repository_key"),
        _required_string(sample.metadata.get("task_group"), "task_group"),
    )


def _group_key(repository_key: str, task_group: str) -> str:
    return canonical_json_bytes({
        "repository_key": repository_key,
        "task_group": task_group,
    }).decode("utf-8")


def _outcome_key(outcome: TaskOutcomeEvidence) -> tuple[int, str, str, str]:
    return (
        outcome.collection_round,
        outcome.stream_id,
        outcome.memory_project_key,
        outcome.outcome.task_id,
    )


def _task_outcome_key(task: TaskEvidence) -> tuple[int, str, str, str]:
    return (
        task.collection_round,
        task.stream_id,
        task.memory_project_key,
        task.task_id,
    )


def _unique_index(items: Sequence[Any], key, label: str) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for item in items:
        item_key = key(item)
        if item_key in result:
            raise ValueError(f"duplicate {label} key: {item_key}")
        result[item_key] = item
    return result


def _correction_payload(**values: Any) -> dict[str, Any]:
    return {
        "schema_version": EXPERT_CORRECTION_SCHEMA_VERSION,
        "original_decision_id": values["original_decision_id"],
        "expected_output_kind": values["expected_output_kind"],
        "expected_tool_call_count": values["expected_tool_call_count"],
        "target": values["target"].to_dict(),
        "annotator": values["annotator"],
        "license": values["license"],
    }


def _synthetic_payload(**values: Any) -> dict[str, Any]:
    return {
        "schema_version": SYNTHETIC_SAMPLE_SCHEMA_VERSION,
        "sample": values["sample"].to_dict(),
        "split": values["split"],
        "generator": values["generator"],
        "license": values["license"],
    }


def _json_mapping(content: str, field_name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return payload


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _load_jsonl(path: str | Path, loader) -> tuple[Any, ...]:
    records: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(loader(payload))
    return tuple(records)


def _write_dataset(
    output_dir: Path,
    samples_by_split: Mapping[str, Sequence[SemanticSFTSample]],
    manifest: SFTDatasetManifest,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(_SPLITS):
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples_by_split.get(split, ()):
                handle.write(canonical_json_bytes(sample.to_dict()).decode("utf-8") + "\n")
    (output_dir / "dataset_manifest.json").write_bytes(
        canonical_json_bytes(manifest.to_dict()) + b"\n"
    )


__all__ = [
    "EXPERT_CORRECTION_SCHEMA_VERSION",
    "SYNTHETIC_SAMPLE_SCHEMA_VERSION",
    "ExpertCorrection",
    "SFTBuildResult",
    "SyntheticSFTRecord",
    "build_canonical_sft_dataset",
    "load_expert_corrections",
    "load_held_out_hashes",
    "load_synthetic_records",
    "validate_semantic_sample",
]
