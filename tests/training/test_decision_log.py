from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from my_agent.policy.contracts import DecisionOutputError, DecisionRequest, DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.training.decision_log import (
    DecisionAttemptError,
    DecisionEventContext,
    DecisionEventRecorder,
    load_decision_events,
)
from my_agent.training.role_views import CanonicalMessage, CanonicalToolCall


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model",
        "model-revision",
        "sha256:" + "1" * 64,
        None,
        "tokenizer-revision",
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
    )


class _Policy:
    def __init__(self, *, fail: bool = False, invalid: bool = False) -> None:
        self.fail = fail
        self.invalid = invalid

    def identity(self):
        return _identity()

    def generate_decision(self, request):
        if self.fail:
            raise RuntimeError("maximum context length exceeded")
        response = DecisionResponse(
            raw_completion='<tool_call>{"name":"finish","arguments":{}}</tool_call>',
            prompt_token_ids=(10, 11),
            completion_token_ids=(20, 21),
            assistant_loss_mask=(1, 1),
            parsed_tool_calls=(CanonicalToolCall("call-1", "finish", "{}"),),
            identity=self.identity(),
        )
        if self.invalid:
            raise DecisionOutputError(response, ValueError("invalid tool JSON"))
        return response

    def render_prompt_hash(self, request):
        return canonical_sha256({
            "messages": [item.to_dict() for item in request.messages],
            "tools": [item.to_dict() for item in request.tools],
        })

    def chat_response_from_decision(self, response):
        return response

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")


def _request() -> DecisionRequest:
    return DecisionRequest(
        role="action",
        purpose="fast_loop_evidence",
        messages=(CanonicalMessage("user", "task"),),
        tools=(),
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )


def _context() -> DecisionEventContext:
    return DecisionEventContext(
        trajectory_id="traj-1",
        turn_index=0,
        step_index=0,
        task_id="task-1",
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        run_id="run-m0",
        repository_revision="rev-1",
        candidate_snapshot_hash=canonical_sha256([]),
    )


class DecisionEventRecorderTests(unittest.TestCase):
    def test_records_exact_success_event_to_dataset_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision_events.jsonl"
            traced = []
            recorder = DecisionEventRecorder(
                policy=_Policy(),
                dataset_path=path,
                trace_sink=lambda event, payload: traced.append((event, payload)),
            )

            logged = recorder.generate(_request(), context=_context())
            events = load_decision_events(path)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.decision_id, logged.decision_id)
        self.assertEqual(event.prompt_token_ids, (10, 11))
        self.assertEqual(event.completion_token_ids, (20, 21))
        self.assertEqual(event.assistant_loss_mask, (1, 1))
        self.assertEqual(event.policy_identity_hash, _identity().identity_hash)
        self.assertEqual(event.parsed_output["tool_calls"][0]["name"], "finish")
        self.assertEqual(traced[0][0], "opd.decision")

    def test_generation_failure_is_audited_and_exposes_retry_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision_events.jsonl"
            recorder = DecisionEventRecorder(policy=_Policy(fail=True), dataset_path=path)

            with self.assertRaises(DecisionAttemptError) as captured:
                recorder.generate(_request(), context=_context(), retry_of="dec-previous")
            events = load_decision_events(path)

        self.assertEqual(events[0].decision_id, captured.exception.decision_id)
        self.assertEqual(events[0].retry_of, "dec-previous")
        self.assertEqual(events[0].status, "llm_error")
        self.assertEqual(events[0].completion_token_ids, ())

    def test_invalid_output_preserves_generated_tokens_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision_events.jsonl"
            recorder = DecisionEventRecorder(policy=_Policy(invalid=True), dataset_path=path)

            with self.assertRaises(DecisionAttemptError):
                recorder.generate(_request(), context=_context())
            event = load_decision_events(path)[0]

        self.assertEqual(event.status, "invalid_output")
        self.assertEqual(event.prompt_token_ids, (10, 11))
        self.assertEqual(event.completion_token_ids, (20, 21))
        self.assertEqual(event.assistant_loss_mask, (1, 1))
        self.assertIn("tool JSON", event.parsed_output["error"])


if __name__ == "__main__":
    unittest.main()
