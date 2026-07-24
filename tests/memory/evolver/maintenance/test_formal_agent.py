from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

from my_agent.llm.types import ChatResponse
from my_agent.memory.evolver.maintenance.cadence.ledger import stable_cadence_id
from my_agent.memory.evolver.maintenance.formal.agent import FormalMaintenanceAgent
from my_agent.memory.evolver.maintenance.formal.tools import formal_maintenance_tools
from my_agent.memory.experience.models import ExperienceCreatedBy, ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.policy.contracts import DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.training.decision_log import DecisionEventRecorder
from my_agent.training.role_views import CanonicalToolCall
from tests.memory.experience.fixtures import typed_experience


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


def _call(name: str, arguments: dict) -> CanonicalToolCall:
    return CanonicalToolCall(
        f"call-{name}",
        name,
        canonical_json_bytes(arguments).decode("utf-8"),
    )


def _cadence_id() -> str:
    return stable_cadence_id(
        stream_id="stream-a",
        memory_project_key="project-a",
        interval_tasks=30,
        cadence_index=1,
    )


class _Policy:
    def __init__(self, calls: list[CanonicalToolCall | None]) -> None:
        self.calls = list(calls)
        self.requests = []

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256([item.to_dict() for item in request.messages])

    def generate_decision(self, request):
        self.requests.append(request)
        call = self.calls.pop(0)
        return DecisionResponse(
            raw_completion=(
                "plain text without a tool call"
                if call is None
                else f"<tool_call>{call.arguments_json}</tool_call>"
            ),
            prompt_token_ids=(1,),
            completion_token_ids=(2,),
            assistant_loss_mask=(1,),
            parsed_tool_calls=() if call is None else (call,),
            identity=self.identity(),
        )

    def chat_response_from_decision(self, response):
        return ChatResponse(content="", tool_calls=[])

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")


class FormalMaintenanceAgentTests(unittest.TestCase):
    def test_empty_repository_prompt_requires_finish_without_invented_ids(self) -> None:
        policy = _Policy([_call("finish", {"summary": "repository is empty"})])
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            agent = FormalMaintenanceAgent(
                policy=policy,
                recorder=DecisionEventRecorder(policy=policy),
                store=store,
                project_key="project-a",
            )

            result = agent.run(
                maintenance_id=_cadence_id(),
                attempt_id="attempt-a",
                stream_id="stream-a",
                task_group="group-a",
            )

        request = policy.requests[0]
        public = json.loads(request.messages[1].content)["public_view"]
        self.assertEqual(result.status, "noop")
        self.assertEqual(public["repository_snapshot"]["memory_ids"], [])
        self.assertIn("repository is empty", request.messages[0].content)
        self.assertIn("exactly one tool call", request.messages[0].content)
        self.assertIn("never invent memory IDs", request.messages[0].content)

    def test_invalid_output_is_corrected_once_then_finish_is_noop(self) -> None:
        policy = _Policy([None, _call("finish", {"summary": "corrected"})])
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            recorder = DecisionEventRecorder(
                policy=policy,
                trace_sink=lambda event, payload: events.append((event, payload)),
            )
            agent = FormalMaintenanceAgent(
                policy=policy,
                recorder=recorder,
                store=store,
                project_key="project-a",
            )

            result = agent.run(
                maintenance_id=_cadence_id(),
                attempt_id="attempt-a",
                stream_id="stream-a",
                task_group="group-a",
            )

        decisions = [payload for event, payload in events if event == "opd.decision"]
        self.assertEqual(result.status, "noop")
        self.assertEqual(result.turns, 2)
        self.assertEqual([item["status"] for item in decisions], ["invalid_output", "success"])
        self.assertEqual(decisions[1]["retry_of"], decisions[0]["decision_id"])
        self.assertEqual(policy.requests[1].messages[-1].role, "user")
        self.assertIn(
            "exactly one valid maintenance tool call",
            policy.requests[1].messages[-1].content,
        )

    def test_invalid_output_aborts_after_two_correction_retries(self) -> None:
        policy = _Policy([None, None, None])
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            agent = FormalMaintenanceAgent(
                policy=policy,
                recorder=DecisionEventRecorder(policy=policy),
                store=store,
                project_key="project-a",
            )

            result = agent.run(
                maintenance_id=_cadence_id(),
                attempt_id="attempt-a",
                stream_id="stream-a",
                task_group="group-a",
            )

        self.assertEqual(result.status, "aborted")
        self.assertEqual(result.turns, 3)
        self.assertEqual(len(policy.requests), 3)
        self.assertIn("ValueError", result.error)
        self.assertIn("exactly one tool call", result.error)

    def test_tools_exclude_promote(self) -> None:
        self.assertEqual(
            [tool.name for tool in formal_maintenance_tools()],
            ["lookup", "merge", "delete", "finish"],
        )

    def test_multi_turn_lookup_merge_finish_logs_each_decision_and_commits_once(self) -> None:
        calls = [
            _call("lookup", {"query": "focused test", "limit": 10}),
            _call("merge", {
                "source_ids": ["tip-a", "tip-b"],
                "replacement": {
                    "content": "Run the focused test before the full suite.",
                    "payload": {
                        "category": "testing",
                        "severity": "info",
                        "trigger": "verification",
                    },
                },
                "reason": "combine overlapping verification tips",
            }),
            _call("finish", {"summary": "merged duplicate tips"}),
        ]
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            created_at = datetime(2026, 7, 18, 9, 30, tzinfo=timezone.utc)
            store.add(typed_experience(
                "tip-a", "Run focused tests first.", ExperienceTier.TIP,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
                created_at=created_at,
            ))
            store.add(typed_experience(
                "tip-b", "Run the focused test.", ExperienceTier.TIP,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
                created_at=created_at,
            ))
            policy = _Policy(calls)
            agent = FormalMaintenanceAgent(
                policy=policy,
                recorder=DecisionEventRecorder(
                    policy=policy,
                    trace_sink=lambda event, payload: events.append((event, payload)),
                ),
                store=store,
                project_key="project-a",
            )

            result = agent.run(
                maintenance_id=_cadence_id(),
                attempt_id="attempt-a",
                stream_id="stream-a",
                task_group="group-a",
            )
            memories = store.all(project_key="project-a")

        self.assertEqual(result.status, "committed")
        self.assertEqual(result.turns, 3)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, "tip-a")
        self.assertEqual(memories[0].content, "Run the focused test before the full suite.")
        lookup_observation = json.loads(policy.requests[1].messages[-1].content)
        self.assertEqual(
            set(lookup_observation["hits"][0]["payload"]),
            {"category", "severity", "trigger"},
        )
        decisions = [payload for event, payload in events if event == "opd.decision"]
        self.assertEqual([item["turn_index"] for item in decisions], [0, 1, 2])
        self.assertTrue(all(item["role"] == "maintenance" for item in decisions))
        normalized = [
            {**item, "decision_id": f"<decision-{index}>"}
            for index, item in enumerate(decisions)
        ]
        self.assertEqual(
            canonical_sha256(normalized),
            "sha256:e9d940e8106d33cafe2e142ba0fb95987f06e211b9ec56737ed0b99cedc1f35a",
        )

    def test_lookup_accepts_exact_snapshot_memory_id(self) -> None:
        calls = [
            _call("lookup", {"query": "tip-a", "limit": 10}),
            _call("finish", {"summary": "inspected tip-a"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience(
                "tip-a",
                "Run focused tests first.",
                ExperienceTier.TIP,
                project_key="project-a",
                created_by=ExperienceCreatedBy.WRITER,
            ))
            policy = _Policy(calls)
            agent = FormalMaintenanceAgent(
                policy=policy,
                recorder=DecisionEventRecorder(policy=policy),
                store=store,
                project_key="project-a",
            )

            result = agent.run(
                maintenance_id=_cadence_id(),
                attempt_id="attempt-a",
                stream_id="stream-a",
                task_group="group-a",
            )

        observation = json.loads(policy.requests[1].messages[-1].content)
        self.assertEqual(result.status, "noop")
        self.assertEqual(observation["hits"][0]["memory_id"], "tip-a")

    def test_protected_delete_aborts_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            protected = typed_experience(
                "tip-protected", "protected tip", ExperienceTier.TIP,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
                protected=True,
            )
            store.add(protected)
            policy = _Policy([_call("delete", {
                "source_ids": ["tip-protected"],
                "reason": "remove it",
            })])
            agent = FormalMaintenanceAgent(
                policy=policy,
                recorder=DecisionEventRecorder(policy=policy),
                store=store,
                project_key="project-a",
            )

            result = agent.run(
                maintenance_id=_cadence_id(),
                attempt_id="attempt-a",
                stream_id="stream-a",
                task_group="group-a",
            )
            remaining = store.get("tip-protected")

        self.assertEqual(result.status, "aborted")
        self.assertEqual(remaining, protected)

    def test_staged_mutation_without_finish_never_changes_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience(
                "tip-a", "tip a", ExperienceTier.TIP,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
            ))
            store.add(typed_experience(
                "tip-b", "tip b", ExperienceTier.TIP,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
            ))
            policy = _Policy([_call("delete", {
                "source_ids": ["tip-b"],
                "reason": "staged only",
            })])
            agent = FormalMaintenanceAgent(
                policy=policy,
                recorder=DecisionEventRecorder(policy=policy),
                store=store,
                project_key="project-a",
                max_turns=1,
            )

            result = agent.run(
                maintenance_id=_cadence_id(),
                attempt_id="attempt-a",
                stream_id="stream-a",
                task_group="group-a",
            )
            ids = [item.id for item in store.all(project_key="project-a")]

        self.assertEqual(result.status, "aborted")
        self.assertEqual(ids, ["tip-a", "tip-b"])


if __name__ == "__main__":
    unittest.main()
