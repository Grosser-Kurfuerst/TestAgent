from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from my_agent.evaluation.formal_role_protocol import (
    evaluate_formal_role_events,
    run_formal_role_protocol_evaluation,
)
from my_agent.memory.evolver.maintenance.formal.tools import formal_maintenance_tools
from my_agent.memory.evolver.selection.prompt import build_selection_request
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.training.contracts import DecisionEvent
from my_agent.training.role_views import (
    CandidateSnapshotEntry,
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
)


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


def _read_file_tool() -> CanonicalTool:
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    return CanonicalTool(
        "read_file",
        "Read a file.",
        canonical_json_bytes(parameters).decode("utf-8"),
        canonical_sha256(parameters),
    )


def _event(
    *,
    decision_id: str,
    role: str,
    raw_completion: str,
    parsed_output: dict,
    status: str = "success",
    messages: tuple[CanonicalMessage, ...] | None = None,
    tools: tuple[CanonicalTool, ...] = (),
) -> DecisionEvent:
    return DecisionEvent(
        role=role,
        purpose="fast_loop_evidence",
        decision_id=decision_id,
        trajectory_id=f"traj-{decision_id}",
        turn_index=0,
        step_index=0,
        task_id=f"task-{decision_id}",
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        run_id=f"run-{decision_id}",
        policy_identity=_identity(),
        repository_revision="revision-a",
        candidate_snapshot_hash=canonical_sha256([]),
        canonical_messages=messages or (CanonicalMessage("user", "{}"),),
        canonical_tools=tools,
        rendered_prompt_hash=canonical_sha256(decision_id),
        prompt_token_ids=(1,),
        raw_completion=raw_completion,
        completion_token_ids=(2,),
        assistant_loss_mask=(1,),
        parsed_output=parsed_output,
        retry_of=None,
        status=status,
    )


def _selection_events() -> list[DecisionEvent]:
    candidate = CandidateSnapshotEntry(
        "RETRIEVED_TIP_01",
        "tip-a",
        "tip",
        "Run focused tests first.",
        0.9,
        1,
        5,
    )
    request = build_selection_request(
        task="fix task",
        candidates=(candidate,),
        token_budget=100,
        max_items=2,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
    )
    valid = {
        "selected_skills": [],
        "selected_tips": ["RETRIEVED_TIP_01"],
        "selected_tools": [],
        "selected_trajectories": [],
        "reasoning": "useful",
    }
    invalid = {**valid, "selected_tips": ["MADE_UP_TIP"]}
    return [
        _event(
            decision_id="selection-valid",
            role="selection",
            raw_completion=json.dumps(valid),
            parsed_output=valid,
            messages=request.messages,
        ),
        _event(
            decision_id="selection-invalid",
            role="selection",
            raw_completion=json.dumps(invalid),
            parsed_output={"error": "selector referenced unknown candidate label: MADE_UP_TIP"},
            status="invalid_output",
            messages=request.messages,
        ),
    ]


class FormalRoleProtocolTests(unittest.TestCase):
    def test_scores_all_four_roles_and_unknown_references(self) -> None:
        read_call = CanonicalToolCall(
            "call-read",
            "read_file",
            canonical_json_bytes({"path": "README.md"}).decode("utf-8"),
        )
        writing_prompt = CanonicalMessage(
            "user",
            json.dumps({"min_confidence": 0.7, "max_records": 6}),
        )
        valid_tip = [{
            "tier": "tip",
            "content": "Run focused tests before the full suite.",
            "payload": {
                "category": "testing",
                "severity": "info",
                "trigger": "after a focused code change",
            },
            "confidence": 0.9,
            "reason": "reusable verification guidance",
        }]
        maintenance_messages = (
            CanonicalMessage("system", "maintain"),
            CanonicalMessage("user", json.dumps({
                "public_view": {
                    "repository_snapshot": {"memory_ids": ["tip-a"]},
                }
            })),
        )
        delete_call = CanonicalToolCall(
            "call-delete",
            "delete",
            canonical_json_bytes({
                "source_ids": ["invented-tip"],
                "reason": "duplicate",
            }).decode("utf-8"),
        )
        events = [
            *_selection_events(),
            _event(
                decision_id="action-tool",
                role="action",
                raw_completion=json.dumps({
                    "tool": "read_file",
                    "arguments": {"path": "README.md"},
                }),
                parsed_output={"tool_calls": [read_call.to_dict()]},
                tools=(_read_file_tool(),),
            ),
            _event(
                decision_id="action-final",
                role="action",
                raw_completion="Done.",
                parsed_output={"tool_calls": []},
                tools=(_read_file_tool(),),
            ),
            _event(
                decision_id="writing-valid",
                role="writing",
                raw_completion=json.dumps(valid_tip),
                parsed_output={"proposals": valid_tip},
                messages=(writing_prompt,),
            ),
            _event(
                decision_id="writing-invalid",
                role="writing",
                raw_completion="not-json",
                parsed_output={"error": "writer output must be valid JSON"},
                status="invalid_output",
                messages=(writing_prompt,),
            ),
            _event(
                decision_id="maintenance-unknown",
                role="maintenance",
                raw_completion="<tool_call>" + json.dumps({
                    "name": "delete",
                    "arguments": {
                        "source_ids": ["invented-tip"],
                        "reason": "duplicate",
                    },
                }) + "</tool_call>",
                parsed_output={
                    "tool_call": {
                        "call_id": delete_call.call_id,
                        "name": delete_call.name,
                        "arguments": {
                            "source_ids": ["invented-tip"],
                            "reason": "duplicate",
                        },
                    }
                },
                messages=maintenance_messages,
                tools=formal_maintenance_tools(),
            ),
        ]

        summary, details = evaluate_formal_role_events(events)

        self.assertEqual(summary["role_coverage"], {
            "selection": 2,
            "action": 2,
            "writing": 2,
            "maintenance": 1,
        })
        self.assertTrue(summary["coverage_complete"])
        self.assertEqual(summary["roles"]["selection"]["schema_valid_rate"], 0.5)
        self.assertEqual(summary["roles"]["selection"]["unknown_label_rate"], 0.5)
        self.assertEqual(summary["roles"]["action"]["n_tool_call_attempts"], 1)
        self.assertEqual(summary["roles"]["action"]["runtime_tool_call_parse_rate"], 1.0)
        self.assertEqual(summary["roles"]["writing"]["json_array_rate"], 0.5)
        self.assertEqual(summary["roles"]["writing"]["validator_accept_rate"], 0.5)
        self.assertEqual(summary["roles"]["maintenance"]["runtime_tool_call_parse_rate"], 1.0)
        self.assertEqual(summary["roles"]["maintenance"]["unknown_memory_id_rate"], 1.0)
        self.assertFalse(summary["all_available_checks_pass"])
        self.assertFalse(summary["acceptance_pass"])
        self.assertEqual(len(details), 7)

    def test_missing_roles_are_unavailable_not_passing(self) -> None:
        summary, _details = evaluate_formal_role_events(_selection_events()[:1])

        self.assertFalse(summary["coverage_complete"])
        self.assertFalse(summary["acceptance_pass"])
        self.assertIsNone(summary["checks"]["writing.validator_accept_rate"])
        self.assertIsNone(summary["checks"]["maintenance.runtime_tool_call_parse_rate"])

    def test_runner_writes_summary_details_and_report(self) -> None:
        events = _selection_events()[:1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decision_events.jsonl"
            source.write_text(
                "\n".join(
                    canonical_json_bytes(event.to_dict()).decode("utf-8")
                    for event in events
                ) + "\n",
                encoding="utf-8",
            )

            summary = run_formal_role_protocol_evaluation(
                decision_events_path=source,
                output_dir=root / "report",
            )

            stored = json.loads(
                (root / "report" / "metrics_summary.json").read_text(encoding="utf-8")
            )
            report = (root / "report" / "experiment_report.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(summary["n_events"], 1)
        self.assertEqual(stored["roles"]["selection"]["schema_valid_rate"], 1.0)
        self.assertIn("Formal Role Protocol Evaluation", report)
        self.assertIn("UNAVAILABLE", report)


if __name__ == "__main__":
    unittest.main()
