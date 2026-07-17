from __future__ import annotations

import tempfile
import unittest

from my_agent.llm.types import ChatResponse
from my_agent.memory.evolver.cadence_ledger import stable_cadence_id
from my_agent.memory.evolver.maintenance_agent import FormalMaintenanceAgent
from my_agent.memory.evolver.maintenance_tools import formal_maintenance_tools
from my_agent.memory.evolver.types import ExperienceCreatedBy, ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.policy.contracts import DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.training.decision_log import DecisionEventRecorder
from my_agent.training.role_views import CanonicalToolCall
from tests.memory.experience_fixtures import typed_experience


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
    def __init__(self, calls: list[CanonicalToolCall]) -> None:
        self.calls = list(calls)

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256([item.to_dict() for item in request.messages])

    def generate_decision(self, request):
        call = self.calls.pop(0)
        return DecisionResponse(
            raw_completion=f"<tool_call>{call.arguments_json}</tool_call>",
            prompt_token_ids=(1,),
            completion_token_ids=(2,),
            assistant_loss_mask=(1,),
            parsed_tool_calls=(call,),
            identity=self.identity(),
        )

    def chat_response_from_decision(self, response):
        return ChatResponse(content="", tool_calls=[])

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")


class FormalMaintenanceAgentTests(unittest.TestCase):
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
            store.add(typed_experience(
                "tip-a", "Run focused tests first.", ExperienceTier.TIP,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
            ))
            store.add(typed_experience(
                "tip-b", "Run the focused test.", ExperienceTier.TIP,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
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
        decisions = [payload for event, payload in events if event == "opd.decision"]
        self.assertEqual([item["turn_index"] for item in decisions], [0, 1, 2])
        self.assertTrue(all(item["role"] == "maintenance" for item in decisions))

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
