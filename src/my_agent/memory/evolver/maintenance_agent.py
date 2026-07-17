"""Multi-turn pure-LLM formal maintenance agent with staged mutations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from my_agent.memory.evolver.maintenance_prompt import (
    build_maintenance_request,
    maintenance_initial_messages,
    maintenance_public_view,
)
from my_agent.memory.evolver.maintenance_tools import (
    MaintenanceToolCommand,
    build_delete_operation,
    build_merge_operation,
    formal_maintenance_tools,
    parse_maintenance_tool_call,
)
from my_agent.memory.evolver.planner import lookup_experiences
from my_agent.memory.evolver.repository_reducer import validate_formal_operations
from my_agent.memory.evolver.transaction import apply_formal_maintenance_operations
from my_agent.memory.evolver.contracts import MaintenanceOperation, MaintenancePlanError
from my_agent.memory.experience_store import ExperienceStore
from my_agent.policy.contracts import DecisionResponse, GenerationPolicy
from my_agent.policy.identity import canonical_json_bytes
from my_agent.training.decision_log import (
    DecisionAttemptError,
    DecisionEventContext,
    DecisionEventRecorder,
)
from my_agent.training.role_views import CanonicalMessage, TaskOutcomeRef


@dataclass(frozen=True)
class FormalMaintenanceResult:
    status: str
    maintenance_id: str
    turns: int
    operation_ids: tuple[str, ...]
    before_revision: str
    after_revision: str
    summary: str = ""
    error: str = ""


class FormalMaintenanceAgent:
    def __init__(
        self,
        *,
        policy: GenerationPolicy,
        recorder: DecisionEventRecorder,
        store: ExperienceStore,
        project_key: str,
        max_turns: int = 8,
        max_new_tokens: int = 1_024,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> None:
        self.policy = policy
        self.recorder = recorder
        self.store = store
        self.project_key = project_key
        self.max_turns = max(1, int(max_turns))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def run(
        self,
        *,
        maintenance_id: str,
        stream_id: str,
        task_group: str,
        history_window: tuple[TaskOutcomeRef, ...] = (),
    ) -> FormalMaintenanceResult:
        if not maintenance_id or not stream_id or not task_group:
            raise ValueError("formal maintenance requires maintenance_id, stream_id, and task_group")
        snapshot = self.store.load_strict_snapshot()
        entries = snapshot.memories
        tools = formal_maintenance_tools()
        public = maintenance_public_view(
            entries,
            repository_revision=snapshot.revision,
            project_key=self.project_key,
            history_window=history_window,
            tools=tools,
        )
        messages = list(maintenance_initial_messages(public))
        staged: list[MaintenanceOperation] = []
        for turn_index in range(self.max_turns):
            parsed_commands: list[MaintenanceToolCommand] = []
            request = build_maintenance_request(
                messages=tuple(messages),
                tools=tools,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            def parse_response(response: DecisionResponse) -> Mapping[str, Any]:
                command = parse_maintenance_tool_call(response.parsed_tool_calls)
                parsed_commands.append(command)
                return {
                    "tool_call": {
                        "call_id": command.call_id,
                        "name": command.name,
                        "arguments": dict(command.arguments),
                    }
                }

            context = DecisionEventContext(
                trajectory_id=maintenance_id,
                turn_index=turn_index,
                step_index=turn_index,
                task_id=maintenance_id,
                task_group=task_group,
                stream_id=stream_id,
                memory_project_key=self.project_key,
                run_id=maintenance_id,
                repository_revision=snapshot.revision,
                candidate_snapshot_hash=public.repository_snapshot.snapshot_hash,
            )
            try:
                logged = self.recorder.generate(
                    request,
                    context=context,
                    parse_response=parse_response,
                )
                command = parsed_commands[0]
                messages.append(CanonicalMessage(
                    "assistant",
                    "",
                    tool_calls=logged.response.parsed_tool_calls,
                ))
                if command.name == "lookup":
                    observation = self._lookup(command, entries=entries)
                elif command.name == "delete":
                    staged.append(build_delete_operation(command, repository_entries=entries))
                    validate_formal_operations(entries, staged, project_key=self.project_key)
                    observation = {"status": "staged", "operation_id": staged[-1].operation_id}
                elif command.name == "merge":
                    staged.append(build_merge_operation(command, repository_entries=entries))
                    validate_formal_operations(entries, staged, project_key=self.project_key)
                    observation = {"status": "staged", "operation_id": staged[-1].operation_id}
                else:
                    summary = str(command.arguments["summary"])
                    applied = apply_formal_maintenance_operations(
                        store=self.store,
                        expected_revision=snapshot.revision,
                        project_key=self.project_key,
                        operations=tuple(staged),
                    )
                    return FormalMaintenanceResult(
                        status=applied.status,
                        maintenance_id=maintenance_id,
                        turns=turn_index + 1,
                        operation_ids=applied.operation_ids,
                        before_revision=applied.before_revision,
                        after_revision=applied.after_revision,
                        summary=summary,
                        error=applied.error,
                    )
                messages.append(CanonicalMessage(
                    "tool",
                    canonical_json_bytes(observation).decode("utf-8"),
                    tool_call_id=command.call_id,
                ))
            except (DecisionAttemptError, MaintenancePlanError, ValueError) as exc:
                return FormalMaintenanceResult(
                    status="aborted",
                    maintenance_id=maintenance_id,
                    turns=turn_index + 1,
                    operation_ids=tuple(operation.operation_id for operation in staged),
                    before_revision=snapshot.revision,
                    after_revision=self.store.revision(),
                    error=f"{type(exc).__name__}: {exc}",
                )
        return FormalMaintenanceResult(
            status="aborted",
            maintenance_id=maintenance_id,
            turns=self.max_turns,
            operation_ids=tuple(operation.operation_id for operation in staged),
            before_revision=snapshot.revision,
            after_revision=self.store.revision(),
            error="maintenance_max_turns_reached_without_finish",
        )

    def _lookup(
        self,
        command: MaintenanceToolCommand,
        *,
        entries: tuple[Any, ...],
    ) -> dict[str, Any]:
        tiers = command.arguments.get("tiers")
        limit = command.arguments.get("limit", 20)
        hits = lookup_experiences(
            entries,
            str(command.arguments["query"]),
            project_key=self.project_key,
            tiers=tiers if isinstance(tiers, list) else None,
            limit=int(limit),
        )
        return {
            "hits": [
                {
                    "memory_id": hit.memory.id,
                    "tier": hit.tier,
                    "content": hit.memory.content,
                    "score": hit.score,
                }
                for hit in hits
            ]
        }


__all__ = ["FormalMaintenanceAgent", "FormalMaintenanceResult"]
