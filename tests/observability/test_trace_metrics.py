from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.observability.trace_metrics import (
    collect_trace_metrics,
    format_trace_metrics,
)


class TraceMetricsEvolverTests(unittest.TestCase):
    def test_collects_formal_selection_writer_status_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_session_started",
                            "payload": {
                                "candidate_count": 2,
                                "candidates": [
                                    {"memory_id": "tip-1", "tier": "tip"},
                                    {"memory_id": "skill-1", "tier": "skill"},
                                ],
                                "selected_count": 1,
                                "selected_memory_ids": ["skill-1"],
                            },
                        },
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_task_finalized",
                            "payload": {
                                "writer_status": "committed",
                                "written_memory_ids": ["new-tip"],
                            },
                        },
                        {
                            "run_id": "run-2",
                            "event": "memory.evolver_task_finalized",
                            "payload": {
                                "writer_status": "no_write",
                                "written_memory_ids": [],
                            },
                        },
                        {
                            "run_id": "run-3",
                            "event": "memory.evolver_task_finalized",
                            "payload": {
                                "writer_status": "failed_no_write",
                                "written_memory_ids": [],
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_candidate_events, 1)
        self.assertEqual(metrics.evolver_selected_events, 1)
        self.assertEqual(metrics.evolver_candidates_total, 2)
        self.assertEqual(metrics.evolver_selected_total, 1)
        self.assertEqual(metrics.evolver_selected_ids, ("skill-1",))
        self.assertEqual(metrics.evolver_selected_by_tier, {"skill": 1})
        self.assertEqual(metrics.evolver_writer_started_events, 3)
        self.assertEqual(metrics.evolver_writer_saved_events, 1)
        self.assertEqual(metrics.evolver_writer_saved_total, 1)
        self.assertEqual(metrics.evolver_writer_failed_events, 1)
        self.assertEqual(
            metrics.evolver_writer_statuses,
            {"committed": 1, "failed_no_write": 1, "no_write": 1},
        )
        self.assertEqual(metrics.evolver_written_ids, ("new-tip",))

    def test_formal_events_take_precedence_over_legacy_without_double_counting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {
                    "run_id": "same-task",
                    "event": "memory.evolver_candidates",
                    "payload": {"candidate_count": 99},
                },
                {
                    "run_id": "same-task",
                    "event": "memory.evolver_selected",
                    "payload": {"selected_count": 88, "selected_ids": ["legacy"]},
                },
                {
                    "run_id": "same-task",
                    "event": "memory.evolver_session_started",
                    "payload": {
                        "candidate_count": 2,
                        "selected_count": 1,
                        "selected_memory_ids": ["formal"],
                    },
                },
                {
                    "run_id": "same-task",
                    "event": "memory.evolver_writer_saved",
                    "payload": {"saved_count": 77, "saved_ids": ["legacy-write"]},
                },
                {
                    "run_id": "same-task",
                    "event": "memory.evolver_task_finalized",
                    "payload": {
                        "writer_status": "committed",
                        "written_memory_ids": ["formal-write"],
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_candidates_total, 2)
        self.assertEqual(metrics.evolver_selected_total, 1)
        self.assertEqual(metrics.evolver_selected_ids, ("formal",))
        self.assertEqual(metrics.evolver_writer_saved_total, 1)
        self.assertEqual(metrics.evolver_written_ids, ("formal-write",))

    def test_collects_formal_memory_tokens_and_excludes_action_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {
                    "run_id": "run-1",
                    "event": "llm.completed",
                    "payload": {
                        "phase": "react",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                },
                {
                    "run_id": "run-1",
                    "event": "opd.decision",
                    "payload": {
                        "role": "action",
                        "status": "success",
                        "prompt_token_ids": list(range(100)),
                        "completion_token_ids": list(range(50)),
                    },
                },
                {
                    "run_id": "run-1",
                    "event": "opd.decision",
                    "payload": {
                        "role": "selection",
                        "status": "success",
                        "prompt_token_ids": [1, 2, 3],
                        "completion_token_ids": [4, 5],
                    },
                },
                {
                    "run_id": "run-1",
                    "event": "opd.decision",
                    "payload": {
                        "role": "writing",
                        "status": "invalid_output",
                        "prompt_token_ids": [1, 2],
                        "completion_token_ids": [3],
                    },
                },
                {
                    "run_id": "run-1",
                    "event": "opd.decision",
                    "payload": {
                        "role": "maintenance",
                        "status": "success",
                        "prompt_token_ids": [1],
                        "completion_token_ids": [2],
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertTrue(metrics.actor_usage_available)
        self.assertTrue(metrics.memory_usage_available)
        self.assertEqual(metrics.memory_prompt_tokens, 6)
        self.assertEqual(metrics.memory_completion_tokens, 4)
        self.assertEqual(metrics.memory_total_tokens, 10)
        self.assertEqual(metrics.system_total_tokens, 25)
        self.assertNotIn("action", metrics.memory_tokens_by_role)
        self.assertEqual(
            metrics.memory_tokens_by_role["selection"]["total_tokens"],
            5,
        )

    def test_llm_error_without_token_ids_marks_memory_and_system_usage_unknown(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {
                    "run_id": "run-1",
                    "event": "llm.completed",
                    "payload": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        }
                    },
                },
                {
                    "run_id": "run-1",
                    "event": "opd.decision",
                    "payload": {
                        "role": "selection",
                        "status": "success",
                        "prompt_token_ids": [1],
                        "completion_token_ids": [2],
                    },
                },
                {
                    "run_id": "run-1",
                    "event": "opd.decision",
                    "payload": {
                        "role": "writing",
                        "status": "llm_error",
                        "prompt_token_ids": [],
                        "completion_token_ids": [],
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertFalse(metrics.memory_usage_available)
        self.assertIsNone(metrics.memory_total_tokens)
        self.assertIsNone(metrics.system_total_tokens)
        self.assertIsNone(metrics.memory_tokens_by_role["writing"]["total_tokens"])
        self.assertIn("writing", metrics.memory_usage_unavailable_reason)

    def test_all_zero_actor_usage_remains_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {
                    "run_id": "run-1",
                    "event": "llm.completed",
                    "payload": {
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                    },
                },
                {
                    "run_id": "run-1",
                    "event": "opd.decision",
                    "payload": {
                        "role": "selection",
                        "status": "success",
                        "prompt_token_ids": [1, 2],
                        "completion_token_ids": [3],
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertFalse(metrics.actor_usage_available)
        self.assertTrue(metrics.memory_usage_available)
        self.assertEqual(metrics.memory_total_tokens, 3)
        self.assertIsNone(metrics.system_total_tokens)

    def test_zero_total_actor_usage_falls_back_to_prompt_plus_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "event": "llm.completed",
                        "payload": {
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertTrue(metrics.actor_usage_available)
        self.assertEqual(metrics.total_tokens, 15)
        self.assertEqual(metrics.system_total_tokens, 15)

    def test_collects_evolver_candidate_and_selected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_candidates",
                            "payload": {
                                "candidate_count": 3,
                                "tiers": {"tip": 2, "skill": 1},
                                "selection_policy": "rule_tier_weighted_v1",
                            },
                        },
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_selected",
                            "payload": {
                                "selected_count": 2,
                                "selected_ids": ["tip-1", "skill-1"],
                                "tiers": {"tip": 1, "skill": 1},
                                "selection_policy": "rule_tier_weighted_v1",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_candidate_events, 1)
        self.assertEqual(metrics.evolver_selected_events, 1)
        self.assertEqual(metrics.evolver_candidates_total, 3)
        self.assertEqual(metrics.evolver_selected_total, 2)
        self.assertEqual(metrics.evolver_selected_by_tier, {"skill": 1, "tip": 1})
        self.assertEqual(metrics.evolver_selection_policies, {"rule_tier_weighted_v1": 1})
        self.assertEqual(metrics.to_dict()["evolver_selected_total"], 2)
        self.assertIn("Evolver selection: candidate_events=1", format_trace_metrics(metrics))

    def test_recursive_metrics_include_child_evolver_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent = base / "parent.jsonl"
            child = base / "child.jsonl"
            parent.write_text(
                json.dumps(
                    {
                        "run_id": "parent",
                        "event": "agent.completed",
                        "payload": {
                            "stop_reason": "plan_completed",
                            "child_trace_paths": [str(child)],
                        },
                    }
                ),
                encoding="utf-8",
            )
            child.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "child",
                            "event": "memory.evolver_candidates",
                            "payload": {"candidate_ids": ["a", "b"]},
                        },
                        {
                            "run_id": "child",
                            "event": "memory.evolver_selected",
                            "payload": {
                                "selected_ids": ["a"],
                                "tiers": {"tool": 1},
                                "selection_policy": "rule_tier_weighted_v1",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            shallow = collect_trace_metrics(parent, recursive=False)
            recursive = collect_trace_metrics(parent, recursive=True)

        self.assertEqual(shallow.evolver_candidates_total, 0)
        self.assertEqual(recursive.trace_files, 2)
        self.assertEqual(recursive.evolver_candidates_total, 2)
        self.assertEqual(recursive.evolver_selected_total, 1)
        self.assertEqual(recursive.evolver_selected_by_tier, {"tool": 1})

    def test_old_trace_defaults_evolver_metrics_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "event": "run.completed",
                        "payload": {"stop_reason": "done"},
                    }
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace)

        self.assertEqual(metrics.evolver_candidate_events, 0)
        self.assertEqual(metrics.evolver_selected_events, 0)
        self.assertEqual(metrics.evolver_candidates_total, 0)
        self.assertEqual(metrics.evolver_selected_total, 0)
        self.assertEqual(metrics.evolver_selected_by_tier, {})
        self.assertEqual(metrics.evolver_selection_policies, {})
        self.assertEqual(metrics.evolver_writer_started_events, 0)
        self.assertEqual(metrics.evolver_writer_saved_events, 0)
        self.assertEqual(metrics.evolver_writer_saved_total, 0)
        self.assertEqual(metrics.evolver_writer_saved_by_tier, {})
        self.assertEqual(metrics.evolver_writer_failed_events, 0)
        self.assertEqual(metrics.maintenance_runs, 0)
        self.assertEqual(metrics.maintenance_applied_runs, 0)
        self.assertEqual(metrics.maintenance_failures, 0)
        self.assertEqual(metrics.maintenance_committed_with_audit_error, 0)

    def test_collects_evolver_writer_saved_total_and_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_writer_started",
                            "payload": {
                                "mode": "fallback",
                                "outcome": "success",
                                "selected_count": 2,
                                "candidate_count": 5,
                            },
                        },
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_writer_saved",
                            "payload": {
                                "saved_count": 2,
                                "duplicate_count": 1,
                                "saved_records": [
                                    {"id": "exp_1", "tier": "skill"},
                                    {"id": "exp_2", "tier": "tool"},
                                ],
                                "tiers": {"skill": 1, "tool": 1},
                                "writer_policy": "fallback_runtime_v1",
                            },
                        },
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_writer_failed",
                            "payload": {
                                "phase": "unknown",
                                "error": "ValueError: boom",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_writer_started_events, 1)
        self.assertEqual(metrics.evolver_writer_saved_events, 1)
        self.assertEqual(metrics.evolver_writer_saved_total, 2)
        self.assertEqual(metrics.evolver_writer_saved_by_tier, {"skill": 1, "tool": 1})
        self.assertEqual(metrics.evolver_writer_failed_events, 1)
        self.assertEqual(metrics.to_dict()["evolver_writer_saved_total"], 2)
        self.assertIn("Evolver writer: started_events=1", format_trace_metrics(metrics))
        self.assertIn("saved=2", format_trace_metrics(metrics))

    def test_recursive_metrics_include_child_evolver_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent = base / "parent.jsonl"
            child = base / "child.jsonl"
            parent.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "parent",
                            "event": "memory.evolver_writer_started",
                            "payload": {"mode": "fallback"},
                        },
                        {
                            "run_id": "parent",
                            "event": "agent.completed",
                            "payload": {
                                "stop_reason": "plan_completed",
                                "child_trace_paths": [str(child)],
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            child.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "child",
                            "event": "memory.evolver_writer_saved",
                            "payload": {
                                "saved_count": 3,
                                "tiers": {"tip": 2, "skill": 1},
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            shallow = collect_trace_metrics(parent, recursive=False)
            recursive = collect_trace_metrics(parent, recursive=True)

        self.assertEqual(shallow.evolver_writer_started_events, 1)
        self.assertEqual(shallow.evolver_writer_saved_total, 0)
        self.assertEqual(recursive.trace_files, 2)
        self.assertEqual(recursive.evolver_writer_started_events, 1)
        self.assertEqual(recursive.evolver_writer_saved_total, 3)
        self.assertEqual(recursive.evolver_writer_saved_by_tier, {"skill": 1, "tip": 2})

    def test_writer_saved_count_falls_back_to_saved_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "event": "memory.evolver_writer_saved",
                        "payload": {
                            "saved_records": [
                                {"id": "exp_1", "tier": "tip"},
                                {"id": "exp_2", "tier": "skill"},
                            ],
                            "tiers": {"tip": 1, "skill": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_writer_saved_total, 2)
        self.assertEqual(metrics.evolver_writer_saved_by_tier, {"skill": 1, "tip": 1})

    def test_collects_memory_maintenance_metrics_from_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "maintenance_trace.jsonl"
            events = [
                {
                    "run_id": "maintenance-1",
                    "event": "memory.maintenance_started",
                    "payload": {"mode": "apply"},
                },
                {
                    "run_id": "maintenance-1",
                    "event": "memory.maintenance_proposed",
                    "payload": {
                        "keep": 4,
                        "delete": 1,
                        "merge": 2,
                        "promote": 1,
                        "source_entries_removed": 3,
                        "entries_added": 2,
                    },
                },
                {
                    "run_id": "maintenance-1",
                    "event": "memory.maintenance_completed",
                    "payload": {
                        "status": "committed_with_audit_error",
                        "mutation_committed": True,
                    },
                },
                {
                    "run_id": "maintenance-2",
                    "event": "memory.maintenance_started",
                    "payload": {"mode": "apply"},
                },
                {
                    "run_id": "maintenance-2",
                    "event": "memory.maintenance_failed",
                    "payload": {
                        "status": "pre_commit_failed",
                        "stage": "validation",
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.maintenance_runs, 2)
        self.assertEqual(metrics.maintenance_applied_runs, 1)
        self.assertEqual(metrics.maintenance_keep, 4)
        self.assertEqual(metrics.maintenance_delete, 1)
        self.assertEqual(metrics.maintenance_merge, 2)
        self.assertEqual(metrics.maintenance_promote, 1)
        self.assertEqual(metrics.maintenance_removed_entries, 3)
        self.assertEqual(metrics.maintenance_added_entries, 2)
        self.assertEqual(metrics.maintenance_failures, 1)
        self.assertEqual(metrics.maintenance_committed_with_audit_error, 1)
        self.assertEqual(metrics.to_dict()["maintenance_removed_entries"], 3)
        self.assertIn("Memory maintenance: runs=2", format_trace_metrics(metrics))
        self.assertIn("Maintenance actions: keep=4", format_trace_metrics(metrics))


if __name__ == "__main__":
    unittest.main()
