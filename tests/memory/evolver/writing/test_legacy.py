from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.llm.types import ChatResponse, MessageLike
from my_agent.memory.evolver.writing.contracts import (
    ExperienceWriteProposal,
    ExperienceWriteRequest,
    ExperienceWriteStep,
)
from my_agent.memory.evolver.writing.dataset import MemoryWriterDatasetLogger
from my_agent.memory.evolver.writing.legacy import (
    ExperienceWriter,
    build_write_steps_from_tool_history,
    runtime_outcome_from_tool_records,
    writer_policy_for_result,
)
from my_agent.memory.experience.models import (
    ExperienceTier,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)


def _skill_proposal(content: str, confidence: float, *, reason: str = "") -> ExperienceWriteProposal:
    return ExperienceWriteProposal(
        tier=ExperienceTier.SKILL,
        content=content,
        payload=SkillPayload(
            category="debugging",
            technique="Focused verification",
            preconditions=(),
            steps=("rerun the focused test",),
        ),
        confidence=confidence,
        reason=reason,
    )


def _tip_proposal(content: str, confidence: float) -> ExperienceWriteProposal:
    return ExperienceWriteProposal(
        tier=ExperienceTier.TIP,
        content=content,
        payload=TipPayload(category="debugging", severity="warning", trigger="test failure"),
        confidence=confidence,
    )


class FakeWriterLLM:
    supports_tools = True

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        self.prompts.append("\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict)))
        return ChatResponse(content=self.response)


class EvolverWriterTests(unittest.TestCase):
    def test_build_write_steps_from_tool_history_truncates_outputs(self) -> None:
        steps = build_write_steps_from_tool_history(
            [
                {
                    "call": {
                        "tool": "run_tests",
                        "arguments": {"command": "pytest tests/test_example.py -q"},
                    },
                    "result": {
                        "ok": True,
                        "output": "x" * 30,
                        "blocked": False,
                        "reason": "",
                    },
                }
            ],
            max_output_chars=10,
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].step_num, 1)
        self.assertEqual(steps[0].tool, "run_tests")
        self.assertEqual(steps[0].arguments["command"], "pytest tests/test_example.py -q")
        self.assertTrue(steps[0].ok)
        self.assertEqual(steps[0].output, "xxxxxxx...")

    def test_runtime_outcome_from_tool_records_is_unknown_for_finish_without_tests(self) -> None:
        self.assertEqual(
            runtime_outcome_from_tool_records(
                "finish_called",
                [{"call": {"tool": "read_file", "arguments": {}}, "result": {"ok": True, "output": "done"}}],
            ),
            "unknown",
        )

    def test_runtime_outcome_from_tool_records_uses_latest_run_tests_result(self) -> None:
        tool_history = [
            {"call": {"tool": "run_tests"}, "result": {"ok": False, "output": "failed"}},
            {"call": {"tool": "run_tests"}, "result": {"ok": True, "output": "passed"}},
        ]

        self.assertEqual(runtime_outcome_from_tool_records("finish_called", tool_history), "success")

        tool_history[-1] = {"call": {"tool": "run_tests"}, "result": {"ok": False, "output": "failed"}}

        self.assertEqual(runtime_outcome_from_tool_records("finish_called", tool_history), "failure")

    def test_runtime_outcome_from_tool_records_uses_failure_stop_reason_without_tests(self) -> None:
        self.assertEqual(runtime_outcome_from_tool_records("max_steps_reached", []), "failure")

    def test_validate_proposals_filters_low_confidence_and_duplicates(self) -> None:
        writer = ExperienceWriter(min_confidence=0.7, max_records=3, max_content_chars=20)

        accepted, rejected = writer.validate_proposals(
            [
                _skill_proposal("Reusable loop for focused tests", 0.8),
                _skill_proposal("Reusable loop for focused tests", 0.9),
                _tip_proposal("too weak", 0.1),
            ]
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].tier, ExperienceTier.SKILL)
        self.assertEqual(accepted[0].content, "Reusable loop for...")
        self.assertEqual([item["reason"] for item in rejected], ["duplicate_proposal", "low_confidence"])

    def test_validate_proposals_respects_zero_max_records(self) -> None:
        writer = ExperienceWriter(max_records=0)

        accepted, rejected = writer.validate_proposals(
            [_skill_proposal("Reusable loop", 0.8)]
        )

        self.assertEqual(accepted, ())
        self.assertEqual(rejected[0]["reason"], "max_records_zero")

    def test_validate_proposal_rejects_content_truncated_to_empty(self) -> None:
        writer = ExperienceWriter(max_content_chars=0)

        accepted, rejected = writer.validate_proposals(
            [_skill_proposal("Reusable loop", 0.8)]
        )

        self.assertEqual(accepted, ())
        self.assertEqual(rejected[0]["reason"], "empty_content")

    def test_fallback_success_proposes_skill_and_tool_from_passing_tests(self) -> None:
        request = _request(
            outcome="success",
            steps=(
                ExperienceWriteStep(
                    step_num=1,
                    tool="run_tests",
                    arguments={"command": "pytest tests/test_example.py -q"},
                    ok=True,
                    output="passed",
                ),
            ),
        )

        result = ExperienceWriter().propose(request)

        self.assertTrue(result.fallback_used)
        self.assertEqual([proposal.tier for proposal in result.proposals], [ExperienceTier.SKILL, ExperienceTier.TOOL])
        self.assertIsInstance(result.proposals[1].payload, ToolPayload)
        self.assertEqual(result.proposals[1].payload.command, "pytest tests/test_example.py -q")

    def test_fallback_failure_proposes_tip_and_trajectory(self) -> None:
        request = _request(
            outcome="failure",
            stop_reason="max_steps_reached",
            steps=(
                ExperienceWriteStep(step_num=1, tool="read_file", ok=True, output="code"),
                ExperienceWriteStep(step_num=2, tool="run_tests", ok=False, output="failed", error_code="failed"),
            ),
        )

        result = ExperienceWriter().propose(request)

        self.assertEqual([proposal.tier for proposal in result.proposals], [ExperienceTier.TIP, ExperienceTier.TRAJECTORY])
        self.assertIsInstance(result.proposals[1].payload, TrajectoryPayload)
        self.assertEqual(result.proposals[1].payload.outcome, "failure")

    def test_fallback_unknown_trajectory_is_low_confidence_by_default(self) -> None:
        request = _request(
            outcome="unknown",
            steps=(ExperienceWriteStep(step_num=1, tool="read_file", ok=True, output="code"),),
        )

        result = ExperienceWriter().propose(request)

        self.assertEqual(result.proposals, ())
        self.assertEqual(result.rejected, ({"reason": "low_confidence", "tier": "trajectory"},))

    def test_fallback_unknown_proposes_typed_trajectory_when_threshold_allows_it(self) -> None:
        request = _request(
            outcome="unknown",
            steps=(ExperienceWriteStep(step_num=1, tool="read_file", ok=True, output="code"),),
        )

        result = ExperienceWriter(min_confidence=0.60).propose(request)

        self.assertEqual([proposal.tier for proposal in result.proposals], [ExperienceTier.TRAJECTORY])
        self.assertEqual(result.proposals[0].confidence, 0.60)
        self.assertIsInstance(result.proposals[0].payload, TrajectoryPayload)
        self.assertEqual(result.proposals[0].payload.outcome, "unknown")
        self.assertEqual(result.rejected, ())

    def test_llm_writer_accepts_json_array_proposals(self) -> None:
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "skill",
                        "content": "For focused pytest failures, rerun the narrow test after the smallest patch.",
                        "confidence": 0.82,
                        "payload": {
                            "category": "debugging",
                            "technique": "Focused pytest loop",
                            "preconditions": [],
                            "steps": ["rerun the narrow test after the smallest patch"],
                        },
                        "reason": "reusable loop",
                    }
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        self.assertTrue(result.llm_used)
        self.assertFalse(result.fallback_used)
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.proposals[0].tier, ExperienceTier.SKILL)
        self.assertIsInstance(result.proposals[0].payload, SkillPayload)
        self.assertEqual(result.proposals[0].payload.category, "debugging")

    def test_llm_writer_accepts_fenced_json(self) -> None:
        llm = FakeWriterLLM(
            "```json\n"
            "[{\"tier\":\"tip\",\"content\":\"Inspect the latest failing assertion before broad reruns.\","
            "\"confidence\":0.77,\"payload\":{\"category\":\"debugging\","
            "\"severity\":\"warning\",\"trigger\":\"failing assertion\"}}]\n"
            "```"
        )

        result = ExperienceWriter(llm=llm).propose(_request(outcome="failure"), mode="llm")

        self.assertTrue(result.llm_used)
        self.assertEqual(result.proposals[0].tier, ExperienceTier.TIP)

    def test_llm_writer_falls_back_on_invalid_json(self) -> None:
        result = ExperienceWriter(llm=FakeWriterLLM("not json")).propose(_successful_request(), mode="llm")

        self.assertTrue(result.llm_used)
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.proposals)
        self.assertEqual(result.rejected[0]["reason"], "llm_parse_failed")

    def test_llm_writer_rejects_low_confidence_secret_hidden_and_destructive_proposals(self) -> None:
        secret_like = "".join(("gh", "p_", "abcdefghijklmnopqrstuvwxyz123456"))
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "skill",
                        "content": "Useful but low confidence",
                        "payload": {
                            "category": "debugging", "technique": "focused", "preconditions": [],
                            "steps": ["rerun tests"]
                        },
                        "confidence": 0.2,
                    },
                    {
                        "tier": "tip",
                        "content": f"Never save {secret_like}",
                        "payload": {"category": "safety", "severity": "warning", "trigger": "secret"},
                        "confidence": 0.9,
                    },
                    {
                        "tier": "trajectory",
                        "content": "Do not save hidden_test_output details.",
                        "payload": {
                            "task_description": "hidden run",
                            "steps": [{"step_num": 1, "action": "test"}],
                            "outcome": "failure"
                        },
                        "confidence": 0.9,
                    },
                    {
                        "tier": "tool",
                        "content": "Reset the repo.",
                        "confidence": 0.9,
                        "payload": {
                            "name": "reset", "language": "bash", "code": "git reset --hard",
                            "command": "git reset --hard"
                        },
                    },
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        self.assertTrue(result.fallback_used)
        self.assertIn("low_confidence", [item["reason"] for item in result.rejected])
        self.assertGreaterEqual([item["reason"] for item in result.rejected].count("unsafe_content"), 2)
        self.assertIn("unsafe_tool_command", [item["reason"] for item in result.rejected])

    def test_llm_writer_rejects_unsafe_reason(self) -> None:
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "skill",
                        "content": "Use focused pytest verification after a small patch.",
                        "payload": {
                            "category": "debugging", "technique": "focused", "preconditions": [],
                            "steps": ["rerun tests"]
                        },
                        "confidence": 0.9,
                        "reason": "contains hidden_test_output",
                    }
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        self.assertTrue(result.fallback_used)
        self.assertIn("unsafe_content", [item["reason"] for item in result.rejected])

    def test_llm_writer_rejects_incomplete_or_unknown_payload_fields(self) -> None:
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "skill",
                        "content": "Use focused pytest verification after a small patch.",
                        "confidence": 0.9,
                        "payload": {"category": "debugging", "technique": "focused", "steps": []},
                    },
                    {
                        "tier": "tip",
                        "content": "Inspect a failing assertion.",
                        "confidence": 0.9,
                        "payload": {
                            "category": "debugging", "severity": "warning", "trigger": "failure",
                            "source_task": "must-not-enter-payload"
                        },
                    }
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        self.assertTrue(result.fallback_used)
        self.assertEqual([item["reason"] for item in result.rejected].count("invalid_payload"), 2)

    def test_llm_prompt_truncates_tool_output(self) -> None:
        llm = FakeWriterLLM("[]")
        long_output = "x" * 2_000
        request = _request(
            outcome="success",
            steps=(ExperienceWriteStep(step_num=1, tool="run_tests", ok=True, output=long_output),),
        )

        ExperienceWriter(llm=llm, max_input_chars=12_000).propose(request, mode="llm")

        self.assertNotIn(long_output, llm.prompts[0])
        self.assertIn("x" * 997 + "...", llm.prompts[0])

    def test_llm_writer_parses_canonical_trajectory_payload(self) -> None:
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "trajectory",
                        "content": "Task: fix tests\nOutcome: success\nKey steps: run_tests passed",
                        "confidence": 0.9,
                        "payload": {
                            "task_description": "Fix tests",
                            "steps": [
                                {
                                    "step_num": 1,
                                    "action": "run_tests",
                                    "result": "passed",
                                }
                            ],
                            "outcome": "success",
                        },
                    }
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        payload = result.proposals[0].payload
        self.assertIsInstance(payload, TrajectoryPayload)
        self.assertEqual(payload.steps[0].action, "run_tests")
        self.assertEqual(payload.steps[0].result, "passed")

    def test_writer_dataset_logger_creates_parent_and_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "writer.jsonl"
            logger = MemoryWriterDatasetLogger(path)

            logger.append({"run_id": "run-1", "schema_version": 1})
            logger.append({"run_id": "run-2", "schema_version": 1})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["run_id"] for row in rows], ["run-1", "run-2"])
            self.assertEqual(rows[0]["schema_version"], 1)

    def test_writer_policy_for_result_distinguishes_llm_and_fallback_provenance(self) -> None:
        # Pure deterministic fallback (default mode): provenance is fallback.
        self.assertEqual(writer_policy_for_result(llm_used=False, fallback_used=True), "fallback_runtime_v1")
        # LLM JSON accepted, no fallback needed: provenance is LLM-only.
        self.assertEqual(writer_policy_for_result(llm_used=True, fallback_used=False), "llm_json_v1")
        # LLM attempted but fell back to deterministic outputs: combined provenance.
        self.assertEqual(
            writer_policy_for_result(llm_used=True, fallback_used=True),
            "llm_then_fallback_runtime_v1",
        )

    def test_failure_fallback_tip_attributes_to_failing_test_signal(self) -> None:
        request = _request(
            outcome="failure",
            stop_reason="max_steps_reached",
            steps=(
                ExperienceWriteStep(step_num=1, tool="read_file", ok=True, output="code"),
                ExperienceWriteStep(step_num=2, tool="run_tests", ok=False, output="failed", error_code="assertion_failed"),
            ),
        )

        result = ExperienceWriter().propose(request)

        tip = result.proposals[0]
        self.assertEqual(tip.tier, ExperienceTier.TIP)
        self.assertIn("run_tests signal failed", tip.content)
        self.assertIsInstance(tip.payload, TipPayload)
        self.assertEqual(tip.payload.trigger, "assertion_failed")

    def test_failure_fallback_tip_attributes_to_failing_or_blocked_tool_signal(self) -> None:
        request = _request(
            outcome="failure",
            stop_reason="max_steps_reached",
            steps=(
                ExperienceWriteStep(step_num=1, tool="read_file", ok=True, output="code"),
                ExperienceWriteStep(step_num=2, tool="apply_patch", ok=False, blocked=True, output="", error_code="BLOCKED"),
            ),
        )

        result = ExperienceWriter().propose(request)

        tip = result.proposals[0]
        self.assertIn("the latest tool was blocked", tip.content)
        # Must NOT misattribute to a run_tests signal that did not fail.
        self.assertNotIn("run_tests signal failed", tip.content)
        # The blocked tool's error_code is the most specific signal, so it leads the trigger.
        self.assertIsInstance(tip.payload, TipPayload)
        self.assertEqual(tip.payload.trigger, "BLOCKED")

    def test_failure_fallback_tip_uses_stop_reason_only_when_no_signal(self) -> None:
        # Failure inferred purely from stop_reason (budget / max steps / timeout) with
        # no failing test or blocked tool to attribute it to — the tip must NOT claim a
        # tool/test signal failed.
        request = _request(
            outcome="failure",
            stop_reason="context_over_budget",
            steps=(
                ExperienceWriteStep(step_num=1, tool="read_file", ok=True, output="code"),
                ExperienceWriteStep(step_num=2, tool="run_tests", ok=True, output="passed"),
            ),
        )

        result = ExperienceWriter().propose(request)

        tip = result.proposals[0]
        self.assertIn("before reaching a verified success", tip.content)
        self.assertNotIn("signal failed", tip.content)
        self.assertIsInstance(tip.payload, TipPayload)
        self.assertEqual(tip.payload.trigger, "context_over_budget")


def _request(
    *,
    outcome: str,
    stop_reason: str = "finish_called",
    steps: tuple[ExperienceWriteStep, ...] = (),
) -> ExperienceWriteRequest:
    return ExperienceWriteRequest(
        task="Fix a failing pytest regression",
        run_id="run-1",
        trace_path=None,
        stop_reason=stop_reason,
        outcome=outcome,
        outcome_source="runtime",
        steps=steps,
        source_task="task-1",
        stream_id="stream-a",
        task_type="manifest",
        project_key="stream:a",
    )


def _successful_request() -> ExperienceWriteRequest:
    return _request(
        outcome="success",
        steps=(
            ExperienceWriteStep(
                step_num=1,
                tool="run_tests",
                arguments={"command": "pytest tests/test_example.py -q"},
                ok=True,
                output="passed",
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
