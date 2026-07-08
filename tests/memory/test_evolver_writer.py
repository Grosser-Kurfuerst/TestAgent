from __future__ import annotations

import json
import unittest

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.llm.types import ChatResponse, MessageLike
from my_agent.memory.evolver import (
    ExperienceTier,
    ExperienceWriteProposal,
    ExperienceWriteRequest,
    ExperienceWriteStep,
    ExperienceWriter,
    build_write_steps_from_tool_history,
    runtime_outcome_from_tool_records,
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
                ExperienceWriteProposal(ExperienceTier.SKILL, "Reusable loop for focused tests", 0.8),
                ExperienceWriteProposal(ExperienceTier.SKILL, "Reusable loop for focused tests", 0.9),
                ExperienceWriteProposal(ExperienceTier.TIP, "too weak", 0.1),
            ]
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].tier, ExperienceTier.SKILL)
        self.assertEqual(accepted[0].content, "Reusable loop for...")
        self.assertEqual([item["reason"] for item in rejected], ["duplicate_proposal", "low_confidence"])

    def test_validate_proposals_respects_zero_max_records(self) -> None:
        writer = ExperienceWriter(max_records=0)

        accepted, rejected = writer.validate_proposals(
            [ExperienceWriteProposal(ExperienceTier.SKILL, "Reusable loop", 0.8)]
        )

        self.assertEqual(accepted, ())
        self.assertEqual(rejected[0]["reason"], "max_records_zero")

    def test_validate_proposal_rejects_content_truncated_to_empty(self) -> None:
        writer = ExperienceWriter(max_content_chars=0)

        accepted, rejected = writer.validate_proposals(
            [ExperienceWriteProposal(ExperienceTier.SKILL, "Reusable loop", 0.8)]
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
        self.assertEqual(result.proposals[1].metadata["command"], "pytest tests/test_example.py -q")

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
        self.assertEqual(result.proposals[1].metadata["outcome"], "failure")

    def test_fallback_unknown_trajectory_is_low_confidence_by_default(self) -> None:
        request = _request(
            outcome="unknown",
            steps=(ExperienceWriteStep(step_num=1, tool="read_file", ok=True, output="code"),),
        )

        result = ExperienceWriter().propose(request)

        self.assertEqual(result.proposals, ())
        self.assertEqual(result.rejected[0]["reason"], "low_confidence")

    def test_llm_writer_accepts_json_array_proposals(self) -> None:
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "skill",
                        "content": "For focused pytest failures, rerun the narrow test after the smallest patch.",
                        "confidence": 0.82,
                        "metadata": {"category": "debugging"},
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
        self.assertEqual(result.proposals[0].metadata["category"], "debugging")

    def test_llm_writer_accepts_fenced_json(self) -> None:
        llm = FakeWriterLLM(
            "```json\n"
            "[{\"tier\":\"tip\",\"content\":\"Inspect the latest failing assertion before broad reruns.\","
            "\"confidence\":0.77,\"metadata\":{\"category\":\"debugging\"}}]\n"
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
                        "confidence": 0.2,
                    },
                    {
                        "tier": "tip",
                        "content": f"Never save {secret_like}",
                        "confidence": 0.9,
                    },
                    {
                        "tier": "trajectory",
                        "content": "Do not save hidden_test_output details.",
                        "confidence": 0.9,
                    },
                    {
                        "tier": "tool",
                        "content": "Reset the repo.",
                        "confidence": 0.9,
                        "metadata": {"command": "git reset --hard"},
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
                        "confidence": 0.9,
                        "reason": "contains hidden_test_output",
                    }
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        self.assertTrue(result.fallback_used)
        self.assertIn("unsafe_content", [item["reason"] for item in result.rejected])

    def test_llm_writer_truncates_nested_metadata_strings_and_removes_reserved_keys(self) -> None:
        long_log = "x" * 2_000
        long_key = "k" * 2_000
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "skill",
                        "content": "Use focused pytest verification after a small patch.",
                        "confidence": 0.9,
                        "metadata": {
                            "debug_log": long_log,
                            "nested": {"log": long_log},
                            long_key: "long key value",
                            "source_task": "llm-overrides-task",
                            "memory_project_key": "llm-overrides-project",
                        },
                        "reason": long_log,
                    }
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        proposal = result.proposals[0]
        self.assertNotIn("source_task", proposal.metadata)
        self.assertNotIn("memory_project_key", proposal.metadata)
        self.assertNotIn(long_key, proposal.metadata)
        self.assertTrue(any(key.startswith("k" * 100) for key in proposal.metadata))
        self.assertTrue(all(len(key) <= 1_000 for key in proposal.metadata))
        self.assertLessEqual(len(proposal.metadata["debug_log"]), 1_000)
        self.assertLessEqual(len(proposal.metadata["nested"]["log"]), 1_000)
        self.assertLessEqual(len(proposal.reason), 500)

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

    def test_llm_writer_truncates_trajectory_metadata_step_results(self) -> None:
        long_result = "x" * 2_000
        llm = FakeWriterLLM(
            json.dumps(
                [
                    {
                        "tier": "trajectory",
                        "content": "Task: fix tests\nOutcome: success\nKey steps: run_tests passed",
                        "confidence": 0.9,
                        "metadata": {
                            "steps": [
                                {
                                    "step_num": 1,
                                    "action": "run_tests",
                                    "result": long_result,
                                    "output": long_result,
                                }
                            ],
                            "outcome": "success",
                        },
                    }
                ]
            )
        )

        result = ExperienceWriter(llm=llm).propose(_successful_request(), mode="llm")

        step = result.proposals[0].metadata["steps"][0]
        self.assertNotEqual(step["result"], long_result)
        self.assertNotEqual(step["output"], long_result)
        self.assertLessEqual(len(step["result"]), 240)
        self.assertLessEqual(len(step["output"]), 240)


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
