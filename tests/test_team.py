from __future__ import annotations

import json
import threading
import time
import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, MessageLike, messages_to_openai
from my_agent.config import AgentConfig
from my_agent.cancellation import CancellationToken
from my_agent.plan import AgentMode, PlanValidationError, TaskResult, TaskType, normalize_mode, resolve_mode
from my_agent.memory import MemoryManager, MemoryScope
from my_agent.schema import AgentState
from my_agent.team import (
    ExecutionStep,
    JsonTeamStore,
    ReviewDecision,
    StepStatus,
    SubAgent,
    TeamAgent,
    TeamPlanner,
    TeamState,
    execution_batches,
    get_executable_steps,
    parse_review_decision,
    topological_order,
    validate_team_graph,
)
from my_agent.team.types import AgentRole, TeamStatus


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


def write_runtime_repo(repo: Path) -> None:
    (repo / "calculator.py").write_text(
        "def subtract(a: int, b: int) -> int:\n"
        "    \"\"\"Return a minus b.\"\"\"\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import subtract\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_subtract(self):\n"
        "        self.assertEqual(subtract(5, 3), 2)\n",
        encoding="utf-8",
    )


def read_trace(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def agent_state(repo: Path, goal: str, *, test_command: str | None = None) -> AgentState:
    return AgentState.initial(repo_path=repo, task=goal, test_command=test_command)


def fake_config(trace_dir: Path | None = None, **overrides: object) -> AgentConfig:
    resolved_trace_dir = trace_dir or Path("traces")
    values = {
        "provider": "fake",
        "api_key": "",
        "base_url": None,
        "model": "fake",
        "temperature": 0.0,
        "max_steps": 8,
        "command_timeout": 60,
        "trace_dir": resolved_trace_dir,
        "memory_dir": resolved_trace_dir.parent / "memory",
        "use_fake_llm": True,
        "memory_auto_extract": False,
    }
    values.update(overrides)
    return AgentConfig(**values)


class RecordingLLM(FakeLLM):
    def __init__(self, responses: list[ChatResponse | str]):
        super().__init__(chat_responses=responses)
        self.requests: list[list[dict[str, object]]] = []
        self.tools_seen: list[list[dict[str, object]] | None] = []

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        self.requests.append(messages_to_openai(messages))
        self.tools_seen.append(tools)
        return super().chat(messages, tools=tools)  # type: ignore[arg-type]


class ReviewerCrashLLM(FakeLLM):
    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        if tools is None:
            raise RuntimeError("review llm unavailable")
        return super().chat(messages, tools=tools)  # type: ignore[arg-type]


class PlannerCrashLLM(FakeLLM):
    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        if tools is None:
            raise RuntimeError("planner llm unavailable")
        return super().chat(messages, tools=tools)  # type: ignore[arg-type]


class StaticPlanner:
    def __init__(self, team: TeamState):
        self.team = team

    def create_team_plan(self, goal: str, **_: object) -> TeamState:
        return TeamState.from_dict(self.team.to_dict())


class RecordingPlanner(StaticPlanner):
    def __init__(self, team: TeamState):
        super().__init__(team)
        self.repo_context = ""
        self.memory_context = ""

    def create_team_plan(self, goal: str, **kwargs: object) -> TeamState:
        self.repo_context = str(kwargs.get("repo_context", ""))
        self.memory_context = str(kwargs.get("memory_context", ""))
        return super().create_team_plan(goal, **kwargs)


class CountingPlanner(RecordingPlanner):
    def __init__(self, team: TeamState):
        super().__init__(team)
        self.calls = 0

    def create_team_plan(self, goal: str, **kwargs: object) -> TeamState:
        self.calls += 1
        return super().create_team_plan(goal, **kwargs)


class ScriptedWorker:
    name = "worker-test"

    def __init__(self, results: dict[str, list[TaskResult]]):
        self.results = results
        self.calls: list[tuple[str, str, str]] = []
        self.clear_count = 0

    def execute_step(self, state: TeamState, item: ExecutionStep, context: str, feedback: str = "") -> TaskResult:
        self.calls.append((item.id, context, feedback))
        queue = self.results.setdefault(item.id, [])
        if not queue:
            return TaskResult.failure(item.id, "No scripted worker result.")
        return queue.pop(0)

    def clear_history(self) -> None:
        self.clear_count += 1


class ScriptedReviewer:
    def __init__(self, decisions: list[ReviewDecision]):
        self.decisions = decisions
        self.calls: list[tuple[str, str, str]] = []
        self.clear_count = 0

    def review_step(self, goal: str, item: ExecutionStep, context: str, result: str) -> ReviewDecision:
        self.calls.append((item.id, context, result))
        if not self.decisions:
            return ReviewDecision(approved=True, summary="default approval")
        return self.decisions.pop(0)

    def clear_history(self) -> None:
        self.clear_count += 1


class CrashingWorker:
    name = "crashing-worker"

    def execute_step(self, state: TeamState, item: ExecutionStep, context: str, feedback: str = "") -> TaskResult:
        raise RuntimeError("worker crashed")

    def clear_history(self) -> None:
        return None


class CrashingReviewer:
    def review_step(self, goal: str, item: ExecutionStep, context: str, result: str) -> ReviewDecision:
        raise RuntimeError("reviewer crashed")

    def clear_history(self) -> None:
        return None


class ParallelWorker:
    def __init__(
        self,
        name: str,
        calls: list[tuple[str, str]],
        *,
        crash_steps: set[str] | None = None,
        active_counter: dict[str, int] | None = None,
        lock: threading.Lock | None = None,
    ):
        self.name = name
        self.calls = calls
        self.crash_steps = set(crash_steps or set())
        self.active_counter = active_counter
        self.lock = lock or threading.Lock()
        self.clear_count = 0

    def execute_step(self, state: TeamState, item: ExecutionStep, context: str, feedback: str = "") -> TaskResult:
        with self.lock:
            self.calls.append((self.name, item.id))
            if self.active_counter is not None:
                self.active_counter["current"] = self.active_counter.get("current", 0) + 1
                self.active_counter["max"] = max(
                    self.active_counter.get("max", 0),
                    self.active_counter["current"],
                )
        try:
            time.sleep(0.01)
            if item.id in self.crash_steps:
                raise RuntimeError(f"{item.id} crashed")
            return TaskResult.success(item.id, f"{self.name} completed {item.id}")
        finally:
            with self.lock:
                if self.active_counter is not None:
                    self.active_counter["current"] -= 1

    def clear_history(self) -> None:
        self.clear_count += 1


class SlowWorker:
    name = "slow-worker"

    def execute_step(self, state: TeamState, item: ExecutionStep, context: str, feedback: str = "") -> TaskResult:
        time.sleep(0.05)
        return TaskResult.success(item.id, f"late {item.id}")

    def clear_history(self) -> None:
        return None


class CooperativeCancelWorker:
    name = "cooperative-cancel-worker"

    def execute_step(
        self,
        state: TeamState,
        item: ExecutionStep,
        context: str,
        feedback: str = "",
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> TaskResult:
        deadline = time.monotonic() + 1
        while cancellation_token is None or not cancellation_token.is_cancelled():
            if time.monotonic() > deadline:
                return TaskResult.failure(item.id, "Timed out waiting for cancellation.")
            time.sleep(0.001)
        return TaskResult.failure(item.id, "Child observed cancellation.", stop_reason="cancelled")

    def clear_history(self) -> None:
        return None


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


class TeamReviewerTests(unittest.TestCase):
    def test_review_decision_parses_valid_json(self) -> None:
        decision = parse_review_decision(
            json.dumps(
                {
                    "approved": False,
                    "summary": "not enough",
                    "issues": ["missing tests"],
                    "suggestions": ["run tests"],
                }
            )
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.summary, "not enough")
        self.assertEqual(decision.issues, ("missing tests",))
        self.assertIn("missing tests", decision.feedback_text)

    def test_review_decision_rejects_invalid_empty_and_missing_approved(self) -> None:
        self.assertFalse(parse_review_decision("").approved)
        self.assertEqual(parse_review_decision("").parse_error, "empty_review")
        self.assertFalse(parse_review_decision("{not json").approved)
        self.assertIn("invalid_json", parse_review_decision("{not json").parse_error)
        missing = parse_review_decision(json.dumps({"summary": "looks ok"}))
        self.assertFalse(missing.approved)
        self.assertEqual(missing.parse_error, "missing_approved")

    def test_reviewer_uses_chat_without_tools(self) -> None:
        llm = RecordingLLM([ChatResponse(content=json.dumps({"approved": True, "summary": "ok"}), finish_reason="stop")])
        agent = SubAgent(
            name="reviewer-step_1",
            role=AgentRole.REVIEWER,
            config=fake_config(),
            llm=llm,
            repo_path=Path.cwd(),
            trace_dir=Path("traces"),
            command_timeout=60,
        )

        decision = agent.review_step("goal", step("step_1"), "dependency context", "worker result")

        self.assertTrue(decision.approved)
        self.assertEqual(llm.tools_seen, [None])
        request_text = "\n".join(str(message.get("content", "")) for message in llm.requests[0])
        self.assertIn("dependency context", request_text)
        self.assertIn("worker result", request_text)


class TeamSubAgentTests(unittest.TestCase):
    def test_worker_prompt_contains_step_context_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = RecordingLLM([ChatResponse(content="worker completed", finish_reason="stop")])
            agent = SubAgent(
                name="worker-1",
                role=AgentRole.WORKER,
                config=fake_config(base / "traces"),
                llm=llm,
                repo_path=repo,
                trace_dir=base / "traces",
                command_timeout=60,
                test_command="python -m unittest discover -s tests -q",
            )
            team = TeamState.create(goal="Fix calculator", steps=[step("step_1")])
            item = team.steps[0]
            item.acceptance = "calculator behavior is verified"
            item.attempts = 2

            result = agent.execute_step(team, item, "dependency context", feedback="fix missing verification")

            self.assertTrue(result.ok)
            self.assertIn("worker completed", result.output)
            self.assertTrue(llm.tools_seen[0])
            request_text = "\n".join(str(message.get("content", "")) for message in llm.requests[0])
            self.assertIn("You are a worker sub-agent", request_text)
            self.assertIn("- id: step_1", request_text)
            self.assertIn("- type: analysis", request_text)
            self.assertIn("calculator behavior is verified", request_text)
            self.assertIn("dependency context", request_text)
            self.assertIn("fix missing verification", request_text)


class TeamAgentTests(unittest.TestCase):
    def test_single_step_success_marks_team_succeeded_and_persists_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Inspect", summary="single", steps=[step("step_1")])
            store = JsonTeamStore(base / "teams")
            worker = ScriptedWorker({"step_1": [TaskResult.success("step_1", "inspected")]})
            reviewer = ScriptedReviewer([ReviewDecision(approved=True, summary="ok")])
            events: list[object] = []

            state = TeamAgent(
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                state_store=store,
                worker_factory=lambda _: worker,
                reviewer_factory=lambda _: reviewer,
                event_sink=events.append,
            ).run(agent_state(repo, "Inspect"))

            self.assertEqual(state.stop_reason, "team_completed")
            self.assertIn("Team succeeded", state.final_answer)
            stored_files = list((base / "teams").glob("team_*.json"))
            self.assertEqual(len(stored_files), 1)
            stored = TeamState.from_dict(json.loads(stored_files[0].read_text(encoding="utf-8")))
            self.assertEqual(stored.status, TeamStatus.SUCCEEDED)
            self.assertEqual(stored.steps[0].status, StepStatus.COMPLETED)
            event_names = [getattr(event, "event", "") for event in events]
            self.assertIn("team.step.worker_completed", event_names)
            self.assertIn("team.step.review_completed", event_names)
            self.assertIn("team.completed", event_names)

    def test_default_fake_llm_supports_serial_team_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = TeamAgent(
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
            ).run(agent_state(repo, "修复 subtract 并运行测试", test_command="python -m unittest discover -s tests -q"))

            self.assertEqual(state.stop_reason, "team_completed")
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            stored_path = next((base / "traces" / "teams").glob("team_*.json"))
            stored = TeamState.from_dict(json.loads(stored_path.read_text(encoding="utf-8")))
            self.assertEqual(stored.status, TeamStatus.SUCCEEDED)
            self.assertEqual([item.status for item in stored.steps], [StepStatus.COMPLETED] * 3)
            parent_events = read_trace(state.trace_path)
            parent_event_names = [event["event"] for event in parent_events]
            self.assertIn("agent.completed", parent_event_names)
            phases = {
                event["payload"].get("phase")
                for event in parent_events
                if event["event"] == "llm.completed"
            }
            self.assertIn("team_planner", phases)
            self.assertIn("team_reviewer", phases)
            agent_completed = [event for event in parent_events if event["event"] == "agent.completed"][-1]
            self.assertEqual(agent_completed["payload"]["mode"], "team")
            self.assertEqual(len(agent_completed["payload"]["child_trace_paths"]), 3)
            trace_paths = [Path(item.trace_path) for item in stored.steps]
            self.assertEqual(len(set(trace_paths)), 3)
            for trace_path in trace_paths:
                events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
                run_ids = {event["run_id"] for event in events if event["event"] == "run.started"}
                self.assertEqual(len(run_ids), 1)

    def test_team_agent_stops_before_planner_when_context_is_over_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Inspect", summary="single", steps=[step("step_1")])
            planner = CountingPlanner(team)

            state = TeamAgent(
                config=fake_config(
                    base / "traces",
                    context_window=8_000,
                    context_window_explicit=True,
                    response_reserve_tokens=7_000,
                    response_reserve_tokens_explicit=True,
                    compression_buffer_tokens=900,
                    compression_buffer_tokens_explicit=True,
                ),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=planner,  # type: ignore[arg-type]
            ).run(agent_state(repo, "Create a team plan."))

            self.assertEqual(state.stop_reason, "context_over_budget")
            self.assertEqual(planner.calls, 0)
            events = read_trace(state.trace_path)
            event_names = [event["event"] for event in events]
            self.assertIn("context.over_budget", event_names)
            self.assertNotIn("llm.requested", event_names)

    def test_parallel_batch_runs_independent_steps_before_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(
                goal="Inspect independently",
                steps=[
                    step("step_1"),
                    step("step_2"),
                    step("step_3", dependencies=["step_1", "step_2"]),
                ],
            )
            calls: list[tuple[str, str]] = []
            events: list[object] = []

            state = TeamAgent(
                config=fake_config(base / "traces", team_worker_count=2),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda index: ParallelWorker(f"worker-{index}", calls),
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True, summary="ok")]),
                event_sink=events.append,
            ).run(agent_state(repo, "Inspect independently"))

            self.assertEqual(state.stop_reason, "team_completed")
            batch_events = [event for event in events if getattr(event, "event", "") == "team.batch.started"]
            self.assertGreaterEqual(len(batch_events), 2)
            self.assertEqual(batch_events[0].payload["batch"], ["step_1", "step_2"])
            self.assertEqual(batch_events[1].payload["batch"], ["step_3"])
            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual([item.status for item in stored.steps], [StepStatus.COMPLETED] * 3)
            self.assertEqual({item_id for _, item_id in calls}, {"step_1", "step_2", "step_3"})

    def test_parallel_step_events_have_complete_step_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Parallel snapshots", steps=[step("step_1"), step("step_2")])
            events: list[object] = []

            TeamAgent(
                config=fake_config(base / "traces", team_worker_count=2),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda index: ParallelWorker(f"worker-{index}", []),
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
                event_sink=events.append,
            ).run(agent_state(repo, "Parallel snapshots"))

            for event in events:
                if getattr(event, "event", "") != "team.step.completed":
                    continue
                step_payload = event.payload["step"]
                self.assertEqual(step_payload["status"], "completed")
                self.assertTrue(step_payload["result"])
                self.assertFalse(step_payload["error"])

    def test_worker_pool_size_one_reuses_worker_without_concurrent_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Two independent steps", steps=[step("step_1"), step("step_2")])
            calls: list[tuple[str, str]] = []
            counter = {"current": 0, "max": 0}
            lock = threading.Lock()
            worker = ParallelWorker("worker-1", calls, active_counter=counter, lock=lock)

            state = TeamAgent(
                config=fake_config(base / "traces", team_worker_count=1),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: worker,
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
            ).run(agent_state(repo, "Two independent steps"))

            self.assertEqual(state.stop_reason, "team_completed")
            self.assertEqual(counter["max"], 1)
            self.assertEqual(worker.clear_count, 2)
            self.assertEqual([item_id for _, item_id in calls], ["step_1", "step_2"])

    def test_parallel_step_crash_does_not_block_same_batch_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(
                goal="Parallel failure",
                steps=[
                    step("step_1"),
                    step("step_2"),
                    step("step_3", dependencies=["step_1", "step_2"]),
                ],
            )
            calls: list[tuple[str, str]] = []

            state = TeamAgent(
                config=fake_config(base / "traces", team_worker_count=2),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda index: ParallelWorker(f"worker-{index}", calls, crash_steps={"step_1"}),
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
            ).run(agent_state(repo, "Parallel failure"))

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertEqual(stored.steps[1].status, StepStatus.COMPLETED)
            self.assertEqual(stored.steps[2].status, StepStatus.SKIPPED)
            self.assertIn("Worker crashed: RuntimeError: step_1 crashed", stored.steps[0].error)

    def test_parallel_batch_timeout_marks_unfinished_steps_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Timeout batch", steps=[step("step_1"), step("step_2")])
            config = fake_config(base / "traces", team_worker_count=2)
            object.__setattr__(config, "team_step_batch_timeout_seconds", 0.001)
            object.__setattr__(config, "tool_shutdown_grace_seconds", 0)

            state = TeamAgent(
                config=config,
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: SlowWorker(),
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
            ).run(agent_state(repo, "Timeout batch"))

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertIn("Step batch timed out", stored.steps[0].error)

    def test_parallel_batch_timeout_overrides_cooperative_child_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Timeout batch", steps=[step("step_1"), step("step_2")])
            config = fake_config(base / "traces", team_worker_count=2)
            object.__setattr__(config, "team_step_batch_timeout_seconds", 0.001)
            object.__setattr__(config, "tool_shutdown_grace_seconds", 1)
            root_token = CancellationToken()
            state = agent_state(repo, "Timeout batch")
            state.cancellation_token = root_token

            state = TeamAgent(
                config=config,
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: CooperativeCancelWorker(),
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
            ).run(state)

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertFalse(root_token.is_cancelled())
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertIn("Step batch timed out", stored.steps[0].error)
            self.assertEqual(stored.steps[1].status, StepStatus.FAILED)
            self.assertIn("Step batch timed out", stored.steps[1].error)

    def test_parallel_root_cancellation_marks_team_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Cancel batch", steps=[step("step_1"), step("step_2")])
            config = fake_config(base / "traces", team_worker_count=2)
            root_token = CancellationToken()
            state = agent_state(repo, "Cancel batch")
            state.cancellation_token = root_token
            timer = threading.Timer(0.01, lambda: root_token.cancel("user_cancelled"))

            timer.start()
            try:
                state = TeamAgent(
                    config=config,
                    llm=FakeLLM(),
                    trace_dir=base / "traces",
                    command_timeout=60,
                    planner=StaticPlanner(team),  # type: ignore[arg-type]
                    worker_factory=lambda _: CooperativeCancelWorker(),
                    reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
                ).run(state)
            finally:
                timer.cancel()

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_cancelled")
            self.assertEqual(stored.status, TeamStatus.CANCELLED)
            self.assertTrue(all(item.status == StepStatus.CANCELLED for item in stored.steps))

    def test_serial_root_cancellation_token_reaches_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Cancel serial", steps=[step("step_1")])
            config = fake_config(base / "traces", team_worker_count=1)
            root_token = CancellationToken()
            state = agent_state(repo, "Cancel serial")
            state.cancellation_token = root_token
            timer = threading.Timer(0.01, lambda: root_token.cancel("user_cancelled"))

            timer.start()
            try:
                state = TeamAgent(
                    config=config,
                    llm=FakeLLM(),
                    trace_dir=base / "traces",
                    command_timeout=60,
                    planner=StaticPlanner(team),  # type: ignore[arg-type]
                    worker_factory=lambda _: CooperativeCancelWorker(),
                    reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
                ).run(state)
            finally:
                timer.cancel()

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_cancelled")
            self.assertEqual(stored.status, TeamStatus.CANCELLED)
            self.assertEqual(stored.steps[0].status, StepStatus.CANCELLED)

    def test_parallel_worker_events_are_serialized_through_orchestrator_sink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Inspect in parallel", steps=[step("step_1"), step("step_2")])
            sink_lock = threading.Lock()
            overlaps: list[str] = []
            seen_events: list[str] = []

            def serialized_sink(event: object) -> None:
                acquired = sink_lock.acquire(blocking=False)
                if not acquired:
                    overlaps.append(getattr(event, "event", "unknown"))
                    return
                try:
                    seen_events.append(getattr(event, "event", ""))
                    time.sleep(0.002)
                finally:
                    sink_lock.release()

            state = TeamAgent(
                config=fake_config(base / "traces", team_worker_count=2),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                event_sink=serialized_sink,
            ).run(agent_state(repo, "Inspect in parallel", test_command="python -m unittest discover -s tests -q"))

            self.assertEqual(state.stop_reason, "team_completed")
            self.assertEqual(overlaps, [])
            self.assertIn("tool.started", seen_events)
            self.assertIn("team.step.completed", seen_events)

    def test_reviewer_rejection_retries_and_second_attempt_can_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Fix", steps=[step("step_1")])
            worker = ScriptedWorker(
                {
                    "step_1": [
                        TaskResult.success("step_1", "first result"),
                        TaskResult.success("step_1", "second result"),
                    ]
                }
            )
            reviewer = ScriptedReviewer(
                [
                    ReviewDecision(approved=False, summary="needs work", issues=("missing verification",)),
                    ReviewDecision(approved=True, summary="ok"),
                ]
            )
            events: list[object] = []

            state = TeamAgent(
                config=fake_config(base / "traces", team_max_retries=1),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: worker,
                reviewer_factory=lambda _: reviewer,
                event_sink=events.append,
            ).run(agent_state(repo, "Fix"))

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_completed")
            self.assertEqual(stored.steps[0].status, StepStatus.COMPLETED)
            self.assertEqual(stored.steps[0].attempts, 2)
            self.assertEqual(worker.calls[1][2], "Review summary: needs work\n\nIssues:\n- missing verification")
            self.assertIn("team.step.retry_started", [getattr(event, "event", "") for event in events])

    def test_allow_unapproved_results_preserves_last_worker_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(goal="Fix", steps=[step("step_1")])
            worker = ScriptedWorker({"step_1": [TaskResult.success("step_1", "unapproved worker output")]})
            reviewer = ScriptedReviewer([ReviewDecision(approved=False, summary="not good")])

            state = TeamAgent(
                config=fake_config(base / "traces", team_max_retries=0, team_allow_unapproved_results=True),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: worker,
                reviewer_factory=lambda _: reviewer,
            ).run(agent_state(repo, "Fix"))

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_completed")
            self.assertEqual(stored.steps[0].status, StepStatus.COMPLETED)
            self.assertEqual(stored.steps[0].result, "unapproved worker output")

    def test_planner_receives_memory_context_separately_from_repo_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            config = fake_config(base / "traces")
            memory = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            memory.save_fact("AgentCli team memory fact", scope=MemoryScope.PROJECT)
            team = TeamState.create(goal="AgentCli task", steps=[step("step_1")])
            planner = RecordingPlanner(team)

            state = TeamAgent(
                config=config,
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=planner,  # type: ignore[arg-type]
                worker_factory=lambda _: ScriptedWorker({"step_1": [TaskResult.success("step_1", "done")]}),
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
                memory_manager=memory,
            ).run(agent_state(repo, "AgentCli task"))

            self.assertNotIn("AgentCli team memory fact", planner.repo_context)
            self.assertIn("AgentCli team memory fact", planner.memory_context)
            prepared = [
                event["payload"]
                for event in read_trace(state.trace_path)
                if event["event"] == "memory.prepared" and event["payload"].get("phase") == "team_planner"
            ]
            self.assertTrue(prepared)
            self.assertIn("fixed_tokens", prepared[-1])
            self.assertIn("memory_budget_tokens", prepared[-1])
            self.assertIn("long_term_limit", prepared[-1])

    def test_planner_llm_failure_uses_team_planner_failed_stop_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = TeamAgent(
                config=fake_config(base / "traces"),
                llm=PlannerCrashLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
            ).run(agent_state(repo, "Plan with unavailable LLM"))

            self.assertEqual(state.stop_reason, "team_planner_failed")
            self.assertIn("Team failed", state.final_answer)
            events = [
                json.loads(line)
                for line in state.trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            validation = [event for event in events if event["event"] == "team.validation_failed"]
            self.assertEqual(validation[-1]["payload"]["code"], "team_planner_llm_failed")

    def test_reviewer_rejects_until_failed_and_dependency_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(
                goal="Fix",
                steps=[step("step_1"), step("step_2", dependencies=["step_1"])],
            )
            worker = ScriptedWorker(
                {
                    "step_1": [
                        TaskResult.success("step_1", "first result"),
                        TaskResult.success("step_1", "second result"),
                    ]
                }
            )
            reviewer = ScriptedReviewer(
                [
                    ReviewDecision(approved=False, summary="bad"),
                    ReviewDecision(approved=False, summary="still bad"),
                ]
            )

            state = TeamAgent(
                config=fake_config(base / "traces", team_max_retries=1),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: worker,
                reviewer_factory=lambda _: reviewer,
            ).run(agent_state(repo, "Fix"))

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(stored.status, TeamStatus.FAILED)
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertEqual(stored.steps[1].status, StepStatus.SKIPPED)
            self.assertIn("failed", state.final_answer)
            self.assertIn("skipped", state.final_answer)

    def test_worker_failure_fails_step_and_skips_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(
                goal="Fix",
                steps=[step("step_1"), step("step_2", dependencies=["step_1"])],
            )
            worker = ScriptedWorker({"step_1": [TaskResult.failure("step_1", "worker exploded")]})

            state = TeamAgent(
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: worker,
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
            ).run(agent_state(repo, "Fix"))

            stored = TeamState.from_dict(json.loads(next((base / "traces" / "teams").glob("team_*.json")).read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertEqual(stored.steps[0].error, "worker exploded")
            self.assertEqual(stored.steps[1].status, StepStatus.SKIPPED)

    def test_worker_exception_fails_current_step_and_persists_original_team(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(
                goal="Fix",
                steps=[step("step_1"), step("step_2", dependencies=["step_1"])],
            )

            state = TeamAgent(
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: CrashingWorker(),
                reviewer_factory=lambda _: ScriptedReviewer([ReviewDecision(approved=True)]),
            ).run(agent_state(repo, "Fix"))

            stored_files = list((base / "traces" / "teams").glob("team_*.json"))
            self.assertEqual(len(stored_files), 1)
            stored = TeamState.from_dict(json.loads(stored_files[0].read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(stored.status, TeamStatus.FAILED)
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertIn("Worker crashed: RuntimeError: worker crashed", stored.steps[0].error)
            self.assertEqual(stored.steps[1].status, StepStatus.SKIPPED)

    def test_reviewer_exception_fails_current_step_and_persists_original_team(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(
                goal="Fix",
                steps=[step("step_1"), step("step_2", dependencies=["step_1"])],
            )

            state = TeamAgent(
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: ScriptedWorker({"step_1": [TaskResult.success("step_1", "worker result")]}),
                reviewer_factory=lambda _: CrashingReviewer(),
            ).run(agent_state(repo, "Fix"))

            stored_files = list((base / "traces" / "teams").glob("team_*.json"))
            self.assertEqual(len(stored_files), 1)
            stored = TeamState.from_dict(json.loads(stored_files[0].read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(stored.status, TeamStatus.FAILED)
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertEqual(stored.steps[0].result, "worker result")
            self.assertIn("Reviewer crashed: RuntimeError: reviewer crashed", stored.steps[0].error)
            self.assertEqual(stored.steps[1].status, StepStatus.SKIPPED)

    def test_real_reviewer_llm_failure_fails_without_rerunning_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            team = TeamState.create(
                goal="Fix",
                steps=[step("step_1"), step("step_2", dependencies=["step_1"])],
            )
            worker = ScriptedWorker(
                {
                    "step_1": [
                        TaskResult.success("step_1", "worker result"),
                        TaskResult.success("step_1", "should not rerun"),
                    ]
                }
            )

            def reviewer_factory(_: str) -> SubAgent:
                return SubAgent(
                    name="reviewer-step_1",
                    role=AgentRole.REVIEWER,
                    config=fake_config(base / "traces"),
                    llm=ReviewerCrashLLM(),
                    repo_path=repo,
                    trace_dir=base / "traces",
                    command_timeout=60,
                )

            state = TeamAgent(
                config=fake_config(base / "traces", team_max_retries=1),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                planner=StaticPlanner(team),  # type: ignore[arg-type]
                worker_factory=lambda _: worker,
                reviewer_factory=reviewer_factory,
            ).run(agent_state(repo, "Fix"))

            stored_files = list((base / "traces" / "teams").glob("team_*.json"))
            self.assertEqual(len(stored_files), 1)
            stored = TeamState.from_dict(json.loads(stored_files[0].read_text(encoding="utf-8")))
            self.assertEqual(state.stop_reason, "team_failed")
            self.assertEqual(len(worker.calls), 1)
            self.assertEqual(stored.steps[0].status, StepStatus.FAILED)
            self.assertEqual(stored.steps[0].result, "worker result")
            self.assertIn("Reviewer crashed: RuntimeError: review llm unavailable", stored.steps[0].error)
            self.assertEqual(stored.steps[1].status, StepStatus.SKIPPED)


class TeamRoutingTests(unittest.TestCase):
    def test_team_mode_is_normalized_and_resolved(self) -> None:
        self.assertEqual(normalize_mode("team"), AgentMode.TEAM)
        self.assertEqual(resolve_mode("team", "anything"), AgentMode.TEAM)

    def test_auto_mode_uses_team_only_for_explicit_multi_agent_cues(self) -> None:
        self.assertEqual(resolve_mode("auto", "请让 worker 和 reviewer 分工审查这个实现"), AgentMode.TEAM)
        self.assertEqual(resolve_mode("auto", "先检查 calculator.py 再运行测试"), AgentMode.PLAN)


if __name__ == "__main__":
    unittest.main()
