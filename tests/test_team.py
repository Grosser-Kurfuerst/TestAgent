from __future__ import annotations

import json
import unittest

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.llm import FakeLLM
from my_agent.plan import AgentMode, PlanValidationError, TaskType, normalize_mode, resolve_mode
from my_agent.team import (
    ExecutionStep,
    StepStatus,
    TeamPlanner,
    TeamState,
    execution_batches,
    get_executable_steps,
    topological_order,
    validate_team_graph,
)


def step(
    step_id: str,
    *,
    dependencies: list[str] | None = None,
    status: StepStatus = StepStatus.PENDING,
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        title=f"Step {step_id}",
        description=f"Do {step_id}",
        type=TaskType.ANALYSIS,
        dependencies=list(dependencies or []),
        status=status,
    )


class TeamTypeTests(unittest.TestCase):
    def test_execution_step_round_trips_through_dict(self) -> None:
        original = ExecutionStep(
            id="step_1",
            title="Inspect files",
            description="Read calculator.py",
            type=TaskType.INSPECT,
            dependencies=["step_0"],
            acceptance="file inspected",
            status=StepStatus.REVIEWING,
            result="done",
            review_summary="looks good",
            review_issues=["missing test"],
            review_suggestions=["run tests"],
            attempts=2,
            worker_name="worker-1",
            trace_path="/tmp/step_1.jsonl",
            started_at="2026-06-20T10:00:00",
        )

        restored = ExecutionStep.from_dict(json.loads(json.dumps(original.to_dict())))

        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_execution_step_accepts_paicli_style_type_aliases(self) -> None:
        self.assertEqual(ExecutionStep.from_dict({"id": "x", "description": "read", "type": "FILE_READ"}).type, TaskType.INSPECT)
        self.assertEqual(ExecutionStep.from_dict({"id": "x", "description": "write", "type": "FILE_WRITE"}).type, TaskType.EDIT)
        self.assertEqual(ExecutionStep.from_dict({"id": "x", "description": "verify", "type": "VERIFICATION"}).type, TaskType.VERIFY)

    def test_team_state_round_trips_through_dict(self) -> None:
        original = TeamState.create(goal="ship", summary="summary", steps=[step("step_1")])
        original.status = original.status.RUNNING
        original.execution_order = ["step_1"]
        original.trace_path = "/tmp/team.jsonl"

        restored = TeamState.from_dict(json.loads(json.dumps(original.to_dict())))

        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertIs(restored.get_step("step_1"), restored.steps[0])


class TeamGraphTests(unittest.TestCase):
    def test_graph_validates_and_orders_dependencies(self) -> None:
        steps = [step("step_2", dependencies=["step_1"]), step("step_1")]

        validate_team_graph(steps)

        self.assertEqual(topological_order(steps), ["step_1", "step_2"])
        self.assertEqual(execution_batches(steps), [["step_1"], ["step_2"]])

    def test_graph_batches_independent_steps(self) -> None:
        steps = [step("step_1"), step("step_2"), step("step_3", dependencies=["step_1", "step_2"])]

        self.assertEqual(execution_batches(steps), [["step_1", "step_2"], ["step_3"]])

    def test_graph_rejects_duplicate_missing_self_and_cycle(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "duplicate_task_id"):
            validate_team_graph([step("step_1"), step("step_1")])
        with self.assertRaisesRegex(PlanValidationError, "missing_dependency"):
            validate_team_graph([step("step_1", dependencies=["missing"])])
        with self.assertRaisesRegex(PlanValidationError, "self_dependency"):
            validate_team_graph([step("step_1", dependencies=["step_1"])])
        with self.assertRaises(PlanValidationError) as ctx:
            validate_team_graph([step("step_1", dependencies=["step_2"]), step("step_2", dependencies=["step_1"])])
        self.assertEqual(ctx.exception.code, "cycle_detected")

    def test_get_executable_steps_returns_pending_steps_with_completed_dependencies(self) -> None:
        steps = [
            step("step_1", status=StepStatus.COMPLETED),
            step("step_2", dependencies=["step_1"], status=StepStatus.PENDING),
            step("step_3", dependencies=["step_2"], status=StepStatus.PENDING),
            step("step_4", status=StepStatus.RUNNING),
        ]

        self.assertEqual([item.id for item in get_executable_steps(steps)], ["step_2"])


class TeamPlannerTests(unittest.TestCase):
    def test_planner_parses_steps_and_maps_original_ids(self) -> None:
        planner = TeamPlanner(FakeLLM())

        team = planner.parse_team_plan(
            "fix subtract",
            json.dumps(
                {
                    "summary": "fix and test",
                    "steps": [
                        {"id": "inspect", "title": "Inspect", "description": "Read file", "type": "FILE_READ"},
                        {
                            "id": "edit",
                            "title": "Edit",
                            "description": "Fix code",
                            "type": "FILE_WRITE",
                            "dependencies": ["inspect"],
                        },
                    ],
                }
            ),
        )

        self.assertEqual(team.summary, "fix and test")
        self.assertEqual([item.id for item in team.steps], ["step_1", "step_2"])
        self.assertEqual(team.steps[0].type, TaskType.INSPECT)
        self.assertEqual(team.steps[1].type, TaskType.EDIT)
        self.assertEqual(team.steps[1].dependencies, ["step_1"])
        self.assertEqual(team.execution_order, ["step_1", "step_2"])

    def test_planner_accepts_tasks_and_depends_on_fields(self) -> None:
        planner = TeamPlanner(FakeLLM())

        team = planner.parse_team_plan(
            "verify",
            json.dumps(
                {
                    "summary": "verify",
                    "tasks": [
                        {"id": "a", "title": "A", "description": "Inspect"},
                        {"id": "b", "title": "B", "description": "Verify", "depends_on": ["a"]},
                    ],
                }
            ),
        )

        self.assertEqual([item.id for item in team.steps], ["step_1", "step_2"])
        self.assertEqual(team.steps[1].dependencies, ["step_1"])

    def test_planner_falls_back_to_tasks_when_steps_is_empty(self) -> None:
        planner = TeamPlanner(FakeLLM())

        team = planner.parse_team_plan(
            "verify",
            json.dumps(
                {
                    "summary": "verify",
                    "steps": [],
                    "tasks": [{"id": "a", "title": "A", "description": "Inspect"}],
                }
            ),
        )

        self.assertEqual([item.id for item in team.steps], ["step_1"])

    def test_planner_validates_invalid_json_and_dependencies(self) -> None:
        planner = TeamPlanner(FakeLLM())
        with self.assertRaisesRegex(PlanValidationError, "invalid_team_plan_json"):
            planner.parse_team_plan("bad", "{not-json")
        with self.assertRaisesRegex(PlanValidationError, "invalid_team_plan_json"):
            planner.parse_team_plan("bad", json.dumps({"steps": [{"id": "a", "description": "A", "dependencies": "a"}]}))
        with self.assertRaisesRegex(PlanValidationError, "invalid_team_plan_json"):
            planner.parse_team_plan("bad", json.dumps({"steps": [{"id": "a", "description": "A", "dependencies": [1]}]}))

    def test_planner_rejects_duplicate_missing_self_cycle_and_too_many_steps(self) -> None:
        planner = TeamPlanner(FakeLLM(), max_steps=2)
        with self.assertRaisesRegex(PlanValidationError, "duplicate_task_id"):
            planner.parse_team_plan(
                "bad",
                json.dumps({"steps": [{"id": "a", "description": "A"}, {"id": "a", "description": "B"}]}),
            )
        with self.assertRaisesRegex(PlanValidationError, "missing_dependency"):
            planner.parse_team_plan(
                "bad",
                json.dumps({"steps": [{"id": "a", "description": "A", "dependencies": ["missing"]}]}),
            )
        with self.assertRaisesRegex(PlanValidationError, "self_dependency"):
            planner.parse_team_plan(
                "bad",
                json.dumps({"steps": [{"id": "a", "description": "A", "dependencies": ["a"]}]}),
            )
        with self.assertRaisesRegex(PlanValidationError, "cycle_detected"):
            planner.parse_team_plan(
                "bad",
                json.dumps(
                    {
                        "steps": [
                            {"id": "a", "description": "A", "dependencies": ["b"]},
                            {"id": "b", "description": "B", "dependencies": ["a"]},
                        ]
                    }
                ),
            )
        with self.assertRaisesRegex(PlanValidationError, "too_many_tasks"):
            planner.parse_team_plan(
                "bad",
                json.dumps(
                    {
                        "steps": [
                            {"id": "a", "description": "A"},
                            {"id": "b", "description": "B"},
                            {"id": "c", "description": "C"},
                        ]
                    }
                ),
            )

    def test_create_team_plan_uses_fake_llm_team_prompt(self) -> None:
        team = TeamPlanner(FakeLLM()).create_team_plan("修复 subtract 并运行测试", repo_context="calculator.py")

        self.assertEqual(team.summary, "Team plan generated by FakeLLM.")
        self.assertEqual([item.id for item in team.steps], ["step_1", "step_2", "step_3"])
        self.assertEqual(team.steps[1].dependencies, ["step_1"])


class TeamRoutingTests(unittest.TestCase):
    def test_team_mode_is_normalized_and_resolved(self) -> None:
        self.assertEqual(normalize_mode("team"), AgentMode.TEAM)
        self.assertEqual(resolve_mode("team", "anything"), AgentMode.TEAM)


if __name__ == "__main__":
    unittest.main()
