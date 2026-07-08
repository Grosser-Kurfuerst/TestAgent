from __future__ import annotations

import unittest

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    ExperienceTier,
    ExperienceWriteProposal,
    ExperienceWriter,
    build_write_steps_from_tool_history,
    runtime_outcome_from_tool_records,
)


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
        self.assertEqual(steps[0].output, "xxxxxxxxx...")

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
        self.assertEqual(accepted[0].content, "Reusable loop for f...")
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


if __name__ == "__main__":
    unittest.main()
