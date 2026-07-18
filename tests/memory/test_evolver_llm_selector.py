from __future__ import annotations

import json
import tempfile
import unittest

from my_agent.llm.types import ChatResponse
from my_agent.memory.embedding_retrieval import EmbeddingRetriever
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.selector_prompt import LLMTaskSelectionPolicy
from my_agent.memory.evolver.types import ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.policy.contracts import DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.training.decision_log import DecisionEventContext, DecisionEventRecorder
from my_agent.training.role_views import CandidateSnapshotEntry
from tests.memory.experience_fixtures import typed_experience


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


class _Policy:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = 0

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256([item.to_dict() for item in request.messages])

    def generate_decision(self, request):
        self.calls += 1
        return DecisionResponse(
            raw_completion=self.output,
            prompt_token_ids=(1, 2),
            completion_token_ids=(3, 4),
            assistant_loss_mask=(1, 1),
            parsed_tool_calls=(),
            identity=self.identity(),
        )

    def chat_response_from_decision(self, response):
        return ChatResponse(content=self.output)

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")


class _Encoder:
    model_revision = "embed-revision"
    tokenizer_revision = "embed-tokenizer"

    def encode_queries(self, texts):
        return ((1.0, 0.0),) * len(texts)

    def encode_documents(self, texts):
        return ((1.0, 0.0),) * len(texts)


def _candidate(label: str, memory_id: str, tier: str) -> CandidateSnapshotEntry:
    return CandidateSnapshotEntry(label, memory_id, tier, f"{tier} content", 0.9, 1, 4)


def _context() -> DecisionEventContext:
    return DecisionEventContext(
        "traj-1", 0, 0, "task-1", "group-a", "stream-a", "project-a", "run-m0",
        "rev-1", canonical_sha256([]),
    )


class LLMTaskSelectionPolicyTests(unittest.TestCase):
    def test_coordinator_wires_shared_policy_and_selects_once_per_task(self) -> None:
        output = json.dumps({
            "selected_skills": ["RETRIEVED_SKILL_01"],
            "selected_tips": [],
            "selected_tools": [],
            "selected_trajectories": [],
            "reasoning": "directly useful",
        })
        policy = _Policy(output)
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience(
                "skill-a",
                "useful skill",
                ExperienceTier.SKILL,
                project_key="project-a",
            ))
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                retriever=EmbeddingRetriever(_Encoder()),
                policy=policy,
            )

            session = coordinator.begin_task(
                task="task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            coordinator.context_for_session(session)

        self.assertEqual(policy.calls, 1)
        self.assertEqual(session.selected_memory_ids, ("skill-a",))

    def test_selects_candidate_labels_once_and_records_exact_decision(self) -> None:
        output = json.dumps({
            "selected_skills": ["RETRIEVED_SKILL_01"],
            "selected_tips": [],
            "selected_tools": [],
            "selected_trajectories": [],
            "reasoning": "directly useful",
        })
        policy = _Policy(output)
        events = []
        selector = LLMTaskSelectionPolicy(
            policy=policy,
            recorder=DecisionEventRecorder(
                policy=policy,
                trace_sink=lambda event, payload: events.append((event, payload)),
            ),
        )

        selected = selector.select(
            task="task",
            candidates=(_candidate("RETRIEVED_SKILL_01", "mem-1", "skill"),),
            token_budget=100,
            max_items=20,
            context=_context(),
        )

        self.assertEqual(selected, ("mem-1",))
        self.assertEqual(policy.calls, 1)
        self.assertEqual(events[0][1]["role"], "selection")
        self.assertEqual(events[0][1]["status"], "success")
        normalized = {**events[0][1], "decision_id": "<decision-id>"}
        self.assertEqual(
            canonical_sha256(normalized),
            "sha256:908aa7b427643ee68a0d431c34a1700933e91e4006d99fabb7ce8e025aeaf66d",
        )

    def test_invalid_reference_fails_closed_to_empty_selection(self) -> None:
        output = json.dumps({
            "selected_skills": ["RETRIEVED_SKILL_99"],
            "selected_tips": [],
            "selected_tools": [],
            "selected_trajectories": [],
            "reasoning": "bad reference",
        })
        policy = _Policy(output)
        events = []
        selector = LLMTaskSelectionPolicy(
            policy=policy,
            recorder=DecisionEventRecorder(
                policy=policy,
                trace_sink=lambda event, payload: events.append((event, payload)),
            ),
        )

        selected = selector.select(
            task="task",
            candidates=(_candidate("RETRIEVED_SKILL_01", "mem-1", "skill"),),
            token_budget=100,
            max_items=20,
            context=_context(),
        )

        self.assertEqual(selected, ())
        self.assertEqual(events[0][1]["status"], "invalid_output")
        self.assertEqual(events[0][1]["completion_token_ids"], [3, 4])

    def test_formal_selection_is_clipped_by_shared_item_and_token_limits(self) -> None:
        output = json.dumps({
            "selected_skills": [
                "RETRIEVED_SKILL_01",
                "RETRIEVED_SKILL_02",
            ],
            "selected_tips": [],
            "selected_tools": [],
            "selected_trajectories": [],
            "reasoning": "both are useful",
        })
        policy = _Policy(output)
        candidates = (
            _candidate("RETRIEVED_SKILL_01", "mem-1", "skill"),
            _candidate("RETRIEVED_SKILL_02", "mem-2", "skill"),
        )
        selector = LLMTaskSelectionPolicy(
            policy=policy,
            recorder=DecisionEventRecorder(policy=policy),
        )

        selected = selector.select(
            task="task",
            candidates=candidates,
            token_budget=8,
            max_items=1,
            context=_context(),
        )

        self.assertEqual(selected, ("mem-1",))


if __name__ == "__main__":
    unittest.main()
