from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall
from my_agent.plan import (
    AgentMode,
    InMemoryPlanStore,
    JsonPlanStore,
    PlanCancelled,
    PlanEvent,
    PlanExecutor,
    PlanExecuteAgent,
    PlanState,
    PlanStatus,
    PlanTask,
    PlanValidationError,
    Planner,
    ReActTaskRunner,
    TaskGraph,
    TaskResult,
    TaskStatus,
    TaskType,
    should_use_plan,
)
from my_agent.config import AgentConfig


def task(
    task_id: str,
    *,
    depends_on: list[str] | None = None,
    status: TaskStatus = TaskStatus.PENDING,
) -> PlanTask:
    return PlanTask(
        id=task_id,
        title=f"Task {task_id}",
        description=f"Do {task_id}",
        type=TaskType.ANALYSIS,
        depends_on=list(depends_on or []),
        status=status,
    )


def plan_with(tasks: list[PlanTask]) -> PlanState:
    return PlanState.create(goal="ship feature", summary="summary", tasks=tasks)


def write_runtime_repo(repo: Path) -> None:
    (repo / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    \"\"\"Return a minus b.\"\"\"\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add, subtract\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n\n"
        "    def test_subtract(self):\n"
        "        self.assertEqual(subtract(5, 3), 2)\n",
        encoding="utf-8",
    )


def read_trace(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fake_config(trace_dir: Path | None = None, **overrides: object) -> AgentConfig:
    values = {
        "provider": "fake",
        "api_key": "",
        "base_url": None,
        "model": "fake",
        "temperature": 0.0,
        "max_steps": 8,
        "command_timeout": 60,
        "trace_dir": trace_dir or Path("traces"),
        "use_fake_llm": True,
    }
    values.update(overrides)
    return AgentConfig(**values)


def tool_response(name: str, arguments: dict[str, object], call_id: str | None = None) -> ChatResponse:
    raw = json.dumps(arguments, ensure_ascii=False)
    return ChatResponse(
        finish_reason="tool_calls",
        tool_calls=[LLMToolCall(id=call_id or f"call_{name}", name=name, arguments=dict(arguments), arguments_json=raw)],
    )


def one_task_plan_json() -> str:
    return json.dumps(
        {
            "summary": "single task",
            "tasks": [{"id": "inspect", "title": "Inspect", "description": "Inspect files", "type": "inspect"}],
        }
    )


class RecordingTaskRunner:
    def __init__(self, fail_ids: set[str] | None = None):
        self.fail_ids = set(fail_ids or set())
        self.calls: list[str] = []

    def run_task(self, plan: PlanState, task: PlanTask) -> TaskResult:
        self.calls.append(task.id)
        if task.id in self.fail_ids:
            return TaskResult.failure(task.id, "boom", stop_reason="test_failure")
        return TaskResult.success(task.id, f"done {task.id}", trace_path=f"/tmp/{task.id}.jsonl")


class CancellingTaskRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_task(self, plan: PlanState, task: PlanTask) -> TaskResult:
        self.calls.append(task.id)
        raise PlanCancelled("user cancelled")


class FailingPlannerLLM(FakeLLM):
    supports_tools = True

    def __init__(self) -> None:
        super().__init__()
        self.tool_calls = 0

    def chat(self, messages: list[object], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        if tools is None:
            raise RuntimeError("planner unavailable")
        self.tool_calls += 1
        return super().chat(messages, tools=tools)  # type: ignore[arg-type]


class PlanTypeTests(unittest.TestCase):
    def test_task_round_trips_through_dict(self) -> None:
        original = PlanTask(
            id="task_1",
            title="Read files",
            description="Inspect calculator.py",
            type=TaskType.INSPECT,
            depends_on=["task_0"],
            acceptance="file is read",
            status=TaskStatus.RUNNING,
            result="ok",
            trace_path="/tmp/task_1.jsonl",
            max_steps=4,
            started_at="2026-06-16T10:00:00",
        )

        restored = PlanTask.from_dict(json.loads(json.dumps(original.to_dict())))

        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_plan_state_round_trips_through_dict(self) -> None:
        original = PlanState(
            id="plan_1",
            goal="fix subtract",
            summary="two tasks",
            tasks=[task("task_1"), task("task_2", depends_on=["task_1"])],
            status=PlanStatus.RUNNING,
            execution_order=["task_1", "task_2"],
            current_task_id="task_1",
            trace_path="/tmp/plan.jsonl",
        )

        restored = PlanState.from_dict(json.loads(json.dumps(original.to_dict())))

        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_task_defaults_to_pending(self) -> None:
        self.assertEqual(task("task_1").status, TaskStatus.PENDING)

    def test_task_type_rejects_unknown_or_falls_back_to_analysis(self) -> None:
        parsed = PlanTask.from_dict({"id": "task_1", "title": "A", "description": "A", "type": "unknown"})

        self.assertEqual(parsed.type, TaskType.ANALYSIS)

    def test_json_plan_store_repairs_surrogateescape_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_goal = "先检查calculator.py 再进行测试".encode("utf-8").decode(
                "ascii", errors="surrogateescape"
            )
            plan = PlanState.create(goal=raw_goal, summary="summary", tasks=[task("task_1")])
            store = JsonPlanStore(Path(tmp))

            store.save(plan)
            restored = store.get(plan.id)

        self.assertIsNotNone(restored)
        self.assertEqual(restored.goal, "先检查calculator.py 再进行测试")


class TaskGraphTests(unittest.TestCase):
    def test_graph_topological_order_is_dependency_first(self) -> None:
        graph = TaskGraph([task("task_2", depends_on=["task_1"]), task("task_1")])

        self.assertEqual(graph.topological_order(), ["task_1", "task_2"])

    def test_graph_preserves_stable_order_for_same_batch(self) -> None:
        graph = TaskGraph([task("task_b"), task("task_a")])

        self.assertEqual(graph.topological_order(), ["task_b", "task_a"])

    def test_graph_does_not_interleave_newly_ready_dependents_before_current_ready_batch(self) -> None:
        graph = TaskGraph([task("task_3", depends_on=["task_1"]), task("task_1"), task("task_2")])

        self.assertEqual(graph.execution_batches(), [["task_1", "task_2"], ["task_3"]])
        self.assertEqual(graph.topological_order(), ["task_1", "task_2", "task_3"])

    def test_graph_batches_independent_tasks(self) -> None:
        graph = TaskGraph(
            [
                task("task_1"),
                task("task_2"),
                task("task_3", depends_on=["task_1", "task_2"]),
            ]
        )

        self.assertEqual(graph.execution_batches(), [["task_1", "task_2"], ["task_3"]])

    def test_graph_rejects_duplicate_task_ids(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "duplicate_task_id"):
            TaskGraph([task("task_1"), task("task_1")]).validate()

    def test_graph_rejects_missing_dependency(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "missing_dependency"):
            TaskGraph([task("task_1", depends_on=["missing"])]).validate()

    def test_graph_rejects_self_dependency(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "self_dependency"):
            TaskGraph([task("task_1", depends_on=["task_1"])]).validate()

    def test_graph_rejects_cycle(self) -> None:
        with self.assertRaises(PlanValidationError) as ctx:
            TaskGraph([task("task_1", depends_on=["task_2"]), task("task_2", depends_on=["task_1"])]).validate()

        self.assertEqual(ctx.exception.code, "cycle_detected")
        self.assertEqual(ctx.exception.details["cycle_candidates"], ["task_1", "task_2"])

    def test_graph_rejects_too_many_tasks(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "too_many_tasks"):
            TaskGraph([task("task_1"), task("task_2")], max_tasks=1).validate()


class PlannerTests(unittest.TestCase):
    def test_planner_parses_plain_json(self) -> None:
        planner = Planner(FakeLLM())

        plan = planner.parse_plan(
            "fix subtract",
            json.dumps(
                {
                    "summary": "fix and test",
                    "tasks": [
                        {"id": "inspect", "title": "Inspect", "description": "Read file", "type": "inspect"},
                        {
                            "id": "edit",
                            "title": "Edit",
                            "description": "Fix code",
                            "type": "edit",
                            "depends_on": ["inspect"],
                        },
                    ],
                }
            ),
        )

        self.assertEqual(plan.summary, "fix and test")
        self.assertEqual([task.id for task in plan.tasks], ["task_1", "task_2"])
        self.assertEqual(plan.tasks[1].depends_on, ["task_1"])
        self.assertEqual(plan.execution_order, ["task_1", "task_2"])

    def test_planner_parses_json_code_fence(self) -> None:
        planner = Planner(FakeLLM())

        plan = planner.parse_plan(
            "inspect",
            """```json
{"summary":"inspect","tasks":[{"id":"x","title":"X","description":"Inspect","type":"inspect"}]}
```""",
        )

        self.assertEqual(plan.tasks[0].type, TaskType.INSPECT)

    def test_planner_rejects_invalid_json(self) -> None:
        with self.assertRaises(PlanValidationError) as ctx:
            Planner(FakeLLM()).parse_plan("bad", "{not-json")

        self.assertEqual(ctx.exception.code, "invalid_plan_json")

    def test_planner_validates_graph(self) -> None:
        with self.assertRaises(PlanValidationError) as ctx:
            Planner(FakeLLM()).parse_plan(
                "bad",
                json.dumps(
                    {
                        "summary": "bad",
                        "tasks": [
                            {
                                "id": "task_a",
                                "title": "A",
                                "description": "A",
                                "depends_on": ["missing"],
                            }
                        ],
                    }
                ),
            )

        self.assertEqual(ctx.exception.code, "missing_dependency")

    def test_planner_rejects_non_list_dependencies(self) -> None:
        with self.assertRaises(PlanValidationError) as ctx:
            Planner(FakeLLM()).parse_plan(
                "bad",
                json.dumps(
                    {
                        "summary": "bad",
                        "tasks": [
                            {"id": "inspect", "title": "Inspect", "description": "Inspect"},
                            {
                                "id": "edit",
                                "title": "Edit",
                                "description": "Edit",
                                "depends_on": "inspect",
                            },
                        ],
                    }
                ),
            )

        self.assertEqual(ctx.exception.code, "invalid_plan_json")

    def test_planner_rejects_non_string_dependencies(self) -> None:
        with self.assertRaises(PlanValidationError) as ctx:
            Planner(FakeLLM()).parse_plan(
                "bad",
                json.dumps(
                    {
                        "summary": "bad",
                        "tasks": [
                            {"id": "inspect", "title": "Inspect", "description": "Inspect"},
                            {"id": "edit", "title": "Edit", "description": "Edit", "depends_on": [1]},
                        ],
                    }
                ),
            )

        self.assertEqual(ctx.exception.code, "invalid_plan_json")

    def test_planner_normalizes_invalid_task_ids(self) -> None:
        planner = Planner(FakeLLM())

        plan = planner.parse_plan(
            "normalize",
            json.dumps(
                {
                    "summary": "normalize",
                    "tasks": [
                        {"id": "1 bad", "title": "A", "description": "A"},
                        {"id": "second task", "title": "B", "description": "B", "depends_on": ["1 bad"]},
                    ],
                }
            ),
        )

        self.assertEqual([task.id for task in plan.tasks], ["task_1", "task_2"])
        self.assertEqual(plan.tasks[1].depends_on, ["task_1"])

    def test_planner_create_plan_uses_llm_chat(self) -> None:
        llm = FakeLLM(
            chat_responses=[
                ChatResponse(
                    content='{"summary":"one","tasks":[{"id":"x","title":"X","description":"Do X","type":"analysis"}]}',
                    finish_reason="stop",
                )
            ]
        )

        plan = Planner(llm).create_plan("Do X", repo_context="repo")

        self.assertEqual(plan.summary, "one")
        self.assertEqual(plan.status, PlanStatus.CREATED)


class PlanExecutorTests(unittest.TestCase):
    def test_executor_runs_tasks_in_topological_order(self) -> None:
        runner = RecordingTaskRunner()
        plan = plan_with([task("task_2", depends_on=["task_1"]), task("task_1")])

        result = PlanExecutor(runner).execute(plan)

        self.assertEqual(runner.calls, ["task_1", "task_2"])
        self.assertEqual(result.status, PlanStatus.SUCCEEDED)
        self.assertEqual([task.status for task in result.tasks], [TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED])

    def test_executor_marks_downstream_skipped_after_failure(self) -> None:
        runner = RecordingTaskRunner(fail_ids={"task_1"})
        plan = plan_with([task("task_1"), task("task_2", depends_on=["task_1"]), task("task_3", depends_on=["task_2"])])

        result = PlanExecutor(runner).execute(plan)

        self.assertEqual(runner.calls, ["task_1"])
        self.assertEqual(result.status, PlanStatus.FAILED)
        self.assertEqual(result.get_task("task_1").status, TaskStatus.FAILED)
        self.assertEqual(result.get_task("task_2").status, TaskStatus.SKIPPED)
        self.assertEqual(result.get_task("task_3").status, TaskStatus.SKIPPED)

    def test_executor_stops_plan_on_task_failure(self) -> None:
        runner = RecordingTaskRunner(fail_ids={"task_1"})
        plan = plan_with([task("task_1"), task("task_2", depends_on=["task_1"])])

        result = PlanExecutor(runner).execute(plan)

        self.assertEqual(result.status, PlanStatus.FAILED)
        self.assertIn("task_1: boom", result.error)

    def test_executor_persists_plan_state_updates(self) -> None:
        store = InMemoryPlanStore()
        plan = plan_with([task("task_1")])

        result = PlanExecutor(RecordingTaskRunner(), store=store).execute(plan)
        stored = store.get(result.id)

        self.assertIsNotNone(stored)
        self.assertEqual(stored.status, PlanStatus.SUCCEEDED)
        self.assertEqual(stored.tasks[0].result, "done task_1")
        self.assertIsNot(stored, result)

    def test_json_plan_store_persists_plan_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonPlanStore(Path(tmp))
            plan = plan_with([task("task_1")])

            result = PlanExecutor(RecordingTaskRunner(), store=store).execute(plan)
            stored = store.get(result.id)

            self.assertIsNotNone(stored)
            self.assertEqual(stored.status, PlanStatus.SUCCEEDED)
            self.assertTrue((Path(tmp) / f"{result.id}.json").exists())

    def test_executor_emits_task_status_events(self) -> None:
        events: list[PlanEvent] = []
        plan = plan_with([task("task_1")])

        PlanExecutor(RecordingTaskRunner(), event_sink=events.append).execute(plan)

        self.assertEqual(
            [event.event for event in events],
            ["plan.started", "plan.task.ready", "plan.task.started", "plan.task.completed", "plan.completed"],
        )
        self.assertEqual(events[2].task_id, "task_1")
        self.assertEqual(events[2].status, "running")

    def test_cancel_marks_running_task_cancelled(self) -> None:
        plan = plan_with([task("task_1")])

        result = PlanExecutor(CancellingTaskRunner()).execute(plan)

        self.assertEqual(result.status, PlanStatus.CANCELLED)
        self.assertEqual(result.tasks[0].status, TaskStatus.CANCELLED)
        self.assertIn("user cancelled", result.error)

    def test_cancel_stops_before_running_independent_tasks(self) -> None:
        runner = CancellingTaskRunner()
        plan = plan_with([task("task_1"), task("task_2")])

        result = PlanExecutor(runner).execute(plan)

        self.assertEqual(runner.calls, ["task_1"])
        self.assertEqual(result.status, PlanStatus.CANCELLED)
        self.assertEqual(result.get_task("task_1").status, TaskStatus.CANCELLED)
        self.assertEqual(result.get_task("task_2").status, TaskStatus.CANCELLED)


class ReActTaskRunnerTests(unittest.TestCase):
    def test_react_task_runner_builds_task_prompt(self) -> None:
        plan = plan_with([task("task_1"), task("task_2", depends_on=["task_1"])])
        plan.tasks[0].status = TaskStatus.SUCCEEDED
        plan.tasks[0].result = "inspected calculator.py"
        runner = ReActTaskRunner(
            repo_path=Path("."),
            config=fake_config(),
            llm=FakeLLM(),
            trace_dir=Path("traces"),
            command_timeout=60,
        )

        prompt = runner.build_task_prompt(plan, plan.tasks[1])

        self.assertIn("Overall goal:", prompt)
        self.assertIn("Current task id: task_2", prompt)
        self.assertIn("Acceptance criteria:", prompt)
        self.assertIn("task_1 (succeeded): inspected calculator.py", prompt)

    def test_react_task_runner_uses_type_budget_capped_by_global_and_config(self) -> None:
        runner = ReActTaskRunner(
            repo_path=Path("."),
            config=fake_config(plan_task_max_steps=5),
            llm=FakeLLM(),
            trace_dir=Path("traces"),
            command_timeout=60,
            default_max_steps=8,
        )

        self.assertEqual(runner.max_steps_for_task(PlanTask("task_1", "Inspect", "Inspect", type=TaskType.INSPECT)), 4)
        self.assertEqual(runner.max_steps_for_task(PlanTask("task_2", "Edit", "Edit", type=TaskType.EDIT)), 5)

        capped = ReActTaskRunner(
            repo_path=Path("."),
            config=fake_config(plan_task_max_steps=6),
            llm=FakeLLM(),
            trace_dir=Path("traces"),
            command_timeout=60,
            default_max_steps=1,
        )
        self.assertEqual(capped.max_steps_for_task(PlanTask("task_1", "Inspect", "Inspect", type=TaskType.INSPECT)), 1)

    def test_executor_records_task_trace_path_and_child_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            plan = plan_with([PlanTask("task_1", "Read missing", "Read missing file repeatedly", max_steps=6)])
            runner = ReActTaskRunner(
                repo_path=repo,
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                test_command="python -m unittest discover -s tests -q",
            )

            result = PlanExecutor(runner).execute(plan)

            self.assertEqual(result.status, PlanStatus.SUCCEEDED)
            trace_path = Path(result.tasks[0].trace_path)
            self.assertTrue(trace_path.exists())
            event_names = [event["event"] for event in read_trace(trace_path)]
            self.assertIn("tool.started", event_names)
            self.assertIn("tool.completed", event_names)
            self.assertIn("run.completed", event_names)

    def test_executor_maps_react_failure_to_task_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    tool_response("read_file", {"path": "missing.py"}, "call_1"),
                    tool_response("read_file", {"path": "missing.py"}, "call_2"),
                    tool_response("read_file", {"path": "missing.py"}, "call_3"),
                ]
            )
            plan = plan_with([PlanTask("task_1", "Read missing", "Read missing file repeatedly", max_steps=6)])
            runner = ReActTaskRunner(
                repo_path=repo,
                config=fake_config(base / "traces", repeated_failure_window=3),
                llm=llm,
                trace_dir=base / "traces",
                command_timeout=60,
            )

            result = PlanExecutor(runner).execute(plan)

            self.assertEqual(result.status, PlanStatus.FAILED)
            self.assertEqual(result.tasks[0].status, TaskStatus.FAILED)
            self.assertIn("repeated_tool_failure", result.tasks[0].error)


class PlanExecuteAgentTests(unittest.TestCase):
    def test_plan_agent_returns_agent_state_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            agent = PlanExecuteAgent(
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
            )

            state = agent.run(
                repo_path=repo,
                goal="先检查 calculator.py，再修复 subtract，并运行测试",
                test_command="python -m unittest discover -s tests -q",
            )

            self.assertEqual(state.stop_reason, "plan_completed")
            self.assertIn("Plan:", state.plan)
            self.assertIn("status=succeeded", state.review)
            self.assertIn("Plan succeeded", state.final_answer)
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))

    def test_plan_agent_uses_configured_max_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            agent = PlanExecuteAgent(
                config=fake_config(base / "traces", plan_max_tasks=1),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
            )

            state = agent.run(repo_path=repo, goal="先检查 calculator.py，再修复 subtract，并运行测试")

            self.assertEqual(state.stop_reason, "plan_validation_failed")
            self.assertIn("too_many_tasks", state.final_answer)

    def test_plan_validation_failure_writes_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    ChatResponse(
                        content='{"summary":"bad","tasks":[{"id":"x","title":"X","description":"X","depends_on":["missing"]}]}',
                        finish_reason="stop",
                    )
                ]
            )
            agent = PlanExecuteAgent(
                config=fake_config(base / "traces"),
                llm=llm,
                trace_dir=base / "traces",
                command_timeout=60,
            )

            state = agent.run(repo_path=repo, goal="bad plan")

            self.assertEqual(state.stop_reason, "plan_validation_failed")
            event_names = [event["event"] for event in read_trace(state.trace_path)]
            self.assertIn("plan.validation_failed", event_names)

    def test_planner_llm_failure_does_not_execute_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FailingPlannerLLM()
            agent = PlanExecuteAgent(
                config=fake_config(base / "traces"),
                llm=llm,
                trace_dir=base / "traces",
                command_timeout=60,
            )

            state = agent.run(repo_path=repo, goal="bad plan")

            self.assertEqual(state.stop_reason, "plan_validation_failed")
            self.assertEqual(llm.tool_calls, 0)

    def test_task_failure_writes_plan_failed_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    ChatResponse(content=one_task_plan_json(), finish_reason="stop"),
                    tool_response("read_file", {"path": "missing.py"}, "call_1"),
                    tool_response("read_file", {"path": "missing.py"}, "call_2"),
                    tool_response("read_file", {"path": "missing.py"}, "call_3"),
                ]
            )
            agent = PlanExecuteAgent(
                config=fake_config(base / "traces", repeated_failure_window=3),
                llm=llm,
                trace_dir=base / "traces",
                command_timeout=60,
            )

            state = agent.run(repo_path=repo, goal="fail task")

            self.assertEqual(state.stop_reason, "plan_failed")
            event_names = [event["event"] for event in read_trace(state.trace_path)]
            self.assertIn("plan.failed", event_names)


class RoutingTests(unittest.TestCase):
    def test_auto_route_keeps_simple_task_on_react(self) -> None:
        self.assertFalse(should_use_plan("读取 calculator.py"))

    def test_auto_route_uses_plan_for_complex_task(self) -> None:
        self.assertTrue(should_use_plan("先检查 calculator.py，再修复 subtract，并运行测试"))

    def test_agent_mode_values(self) -> None:
        self.assertEqual(AgentMode.PLAN.value, "plan")


if __name__ == "__main__":
    unittest.main()
