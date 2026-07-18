from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from my_agent.budget import AgentBudget
from my_agent.agent_factory import AgentFactory  # noqa: F401 - initializes runtime imports
from my_agent.config import AgentConfig
from my_agent.context import ToolSchemaBudget
from my_agent.llm.types import ChatResponse, LLMToolCall
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact, TaskEvolverSession
from my_agent.memory.evolver.writing.contracts import ExperienceWriteStep
from my_agent.memory.experience_store import ExperienceStore
from my_agent.opd_data.export import load_action_execution_evidence
from my_agent.opd_data.runtime_recorder import RuntimeEvidenceRecorder
from my_agent.policy.contracts import DecisionRequest, DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.react.agent import ReActAgent
from my_agent.schema import AgentState
from my_agent.tools import ToolExecutionResult
from my_agent.tracing import TraceWriter
from my_agent.training.contracts import AuthoritativeTaskOutcome, EvaluatorIdentity
from my_agent.training.decision_log import DecisionEventContext, DecisionEventRecorder
from my_agent.training.role_views import CanonicalMessage, CanonicalTool, CanonicalToolCall


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


def _finish_tool() -> CanonicalTool:
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    return CanonicalTool(
        "finish",
        "Finish the task.",
        canonical_json_bytes(schema).decode(),
        canonical_sha256(schema),
    )


class _Policy:
    default_max_new_tokens = 32
    default_top_p = 1.0

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256({
            "messages": [message.to_dict() for message in request.messages],
            "tools": [tool.to_dict() for tool in request.tools],
        })

    def generate_decision(self, request):
        calls = ()
        completion = "{}"
        if request.role == "action":
            calls = (CanonicalToolCall(
                "call-finish",
                "finish",
                canonical_json_bytes({"summary": "done"}).decode(),
            ),)
            completion = "finish"
        return DecisionResponse(
            raw_completion=completion,
            prompt_token_ids=(1,),
            completion_token_ids=(2,),
            assistant_loss_mask=(1,),
            parsed_tool_calls=calls,
            identity=self.identity(),
        )

    def chat_response_from_decision(self, response):
        return ChatResponse(
            content="",
            tool_calls=[
                LLMToolCall(
                    id=call.call_id,
                    name=call.name,
                    arguments=json.loads(call.arguments_json),
                    arguments_json=call.arguments_json,
                )
                for call in response.parsed_tool_calls
            ],
            finish_reason="tool_calls",
        )


class _Tools:
    registry = SimpleNamespace(last_execution_summary={"groups": []})

    @staticmethod
    def execute_tools(invocations):
        return [
            ToolExecutionResult(
                id=invocation.id,
                name=invocation.name,
                ok=True,
                content="done",
            )
            for invocation in invocations
        ]


class ReactFormalExecutionEvidenceTests(unittest.TestCase):
    def test_logged_decision_id_reaches_materialized_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = _Policy()
            decision_recorder = DecisionEventRecorder(
                policy=policy,
                dataset_path=root / "dataset/decision_events.jsonl",
            )
            store = ExperienceStore.from_dir(root / "memory")
            runtime_recorder = RuntimeEvidenceRecorder(
                dataset_dir=root / "dataset",
                store=store,
                decision_recorder=decision_recorder,
            )
            session = TaskEvolverSession(
                task_id="task-1",
                task_group="group-a",
                trajectory_id="run-1",
                stream_id="stream-a",
                memory_project_key="project-a",
                policy_identity=_identity(),
                repository_revision=store.revision(),
                candidate_snapshot_hash=canonical_sha256([]),
                selected_memory_ids=(),
                rendered_memory_context="",
            )
            runtime_recorder.begin_task(
                task="Finish the public task",
                session=session,
                selection_token_budget=32,
            )
            decision_recorder.generate(
                DecisionRequest(
                    role="selection",
                    purpose="fast_loop_evidence",
                    messages=(CanonicalMessage("user", "Select nothing."),),
                    tools=(),
                    max_new_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                ),
                context=DecisionEventContext(
                    trajectory_id=session.trajectory_id,
                    turn_index=0,
                    step_index=0,
                    task_id=session.task_id,
                    task_group=session.task_group,
                    stream_id=session.stream_id,
                    memory_project_key=session.memory_project_key,
                    run_id=session.trajectory_id,
                    repository_revision=session.repository_revision,
                    candidate_snapshot_hash=session.candidate_snapshot_hash,
                ),
            )
            config = AgentConfig(
                provider="fake",
                api_key="",
                base_url=None,
                model="fake",
                max_steps=4,
                command_timeout=30,
                trace_dir=root / "traces",
                memory_dir=root / "memory-manager",
                use_fake_llm=True,
                temperature=0.0,
            )
            agent = ReActAgent(
                config=config,
                llm=policy,
                trace_dir=root / "traces",
                command_timeout=30,
            )
            state = AgentState.initial(root, "Finish the public task", run_id=session.trajectory_id)
            writer = TraceWriter(root / "trace.jsonl")
            response = agent._formal_chat(
                [{"role": "user", "content": "Finish the public task"}],
                [{
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "description": "Finish the task.",
                        "parameters": json.loads(_finish_tool().parameters_json),
                    },
                }],
                writer,
                state,
                memory=SimpleNamespace(),
                context_manager=SimpleNamespace(),
                base_messages=[],
                tool_budget=ToolSchemaBudget([], ("finish",), (), 100, 10),
                formal_session=session,
                recorder=decision_recorder,
                turn_index=0,
            )
            assert response is not None
            budget = AgentBudget.from_config(config, max_steps=4)
            agent._execute_tool_calls(
                state,
                writer,
                _Tools(),
                response.tool_calls,
                budget,
                ToolSchemaBudget([], ("finish",), (), 100, 10),
                formal_session=session,
                execution_recorder=runtime_recorder,
                decision_id=response.raw["decision_id"],
                decision_turn_index=response.raw["decision_turn_index"],
                decision_step_index=response.raw["decision_step_index"],
            )
            runtime_recorder.finish_task(
                episode=AgentEpisodeArtifact(
                    session=session,
                    trace_path=writer.path,
                    stop_reason="finish_called",
                    final_answer="done",
                    tool_history=(ExperienceWriteStep(0, "finish", {"summary": "done"}, True, "done"),),
                    task="Finish the public task",
                ),
                outcome=AuthoritativeTaskOutcome(
                    "task-1",
                    "group-a",
                    True,
                    True,
                    1.0,
                    EvaluatorIdentity("pytest", "8", canonical_sha256("pytest")),
                ),
                task_ordinal=1,
                written_memory_ids=(),
            )
            executions = load_action_execution_evidence(
                root / "dataset/tool_execution_evidence.jsonl"
            )

        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0].decision_id, response.raw["decision_id"])
        self.assertEqual(executions[0].call_id, "call-finish")
        self.assertTrue(executions[0].ok)


if __name__ == "__main__":
    unittest.main()
