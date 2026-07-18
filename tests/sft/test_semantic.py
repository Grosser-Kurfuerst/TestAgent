from __future__ import annotations

import unittest

from my_agent.memory.evolver.maintenance.formal.tools import (
    formal_maintenance_tools,
    parse_maintenance_tool_call,
)
from my_agent.memory.evolver.selection.formal import parse_selection_response
from my_agent.memory.evolver.writing.formal import parse_writing_response
from my_agent.memory.evolver.writing.validation import ExperienceProposalValidator
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.sft.contracts import deterministic_tool_call_id
from my_agent.sft.semantic import SemanticSFTSample
from my_agent.training.role_views import (
    CandidateSnapshotEntry,
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
)


class SemanticSFTTests(unittest.TestCase):
    def test_role_fixtures_round_trip_through_semantic_and_runtime_parsers(self) -> None:
        action = _action_sample()
        finish = _assistant_text_sample()
        selection = _selection_sample()
        writing = _writing_sample()
        maintenance = _maintenance_sample()

        for sample in (action, finish, selection, writing, maintenance):
            self.assertEqual(SemanticSFTSample.from_dict(sample.to_dict()), sample)

        candidates = (
            CandidateSnapshotEntry("RETRIEVED_SKILL_01", "mem-1", "skill", "tip", 1.0, 1, 2),
        )
        parsed_selection, selected = parse_selection_response(
            selection.target.content,
            candidates=candidates,
        )
        self.assertEqual(parsed_selection["selected_skills"], ["RETRIEVED_SKILL_01"])
        self.assertEqual(selected, ("mem-1",))
        self.assertEqual(
            parse_writing_response(
                writing.target.content,
                validator=ExperienceProposalValidator(),
            ),
            (),
        )
        command = parse_maintenance_tool_call(maintenance.target.tool_calls)
        self.assertEqual(command.name, "finish")

    def test_multi_call_history_preserves_observation_binding(self) -> None:
        read_tool = _tool("read_file", required=("path",))
        search_tool = _tool("search_files", required=("path", "query"))
        first = _call(0, "search_files", {"path": "src", "query": "TODO"})
        second = _call(1, "read_file", {"path": "src/bar.py"})
        sample = SemanticSFTSample.create(
            role="action",
            expected_output_kind="assistant_text",
            expected_tool_call_count=None,
            messages=(
                CanonicalMessage("system", "Use tools."),
                CanonicalMessage("user", "Inspect the result."),
                CanonicalMessage("assistant", "", tool_calls=(first, second)),
                CanonicalMessage("tool", "src/bar.py:8: TODO", tool_call_id=first.call_id),
                CanonicalMessage("tool", "def work(): pass", tool_call_id=second.call_id),
            ),
            tools=(search_tool, read_tool),
            target=CanonicalMessage("assistant", "The TODO is in src/bar.py."),
            metadata=_metadata("multi-observation"),
        )

        restored = SemanticSFTSample.from_dict(sample.to_dict())
        self.assertEqual(restored.messages[2].tool_calls, (first, second))
        self.assertEqual(restored.messages[3].tool_call_id, first.call_id)
        self.assertEqual(restored.messages[4].tool_call_id, second.call_id)

        wrong_order = list(sample.messages)
        wrong_order[3], wrong_order[4] = wrong_order[4], wrong_order[3]
        with self.assertRaisesRegex(ValueError, "next pending call ID"):
            SemanticSFTSample.create(
                role="action",
                expected_output_kind="assistant_text",
                expected_tool_call_count=None,
                messages=tuple(wrong_order),
                tools=sample.tools,
                target=sample.target,
                metadata=_metadata("wrong-order"),
            )

    def test_target_contract_rejects_wrong_call_id_and_hidden_reasoning(self) -> None:
        tool = _tool("read_file", required=("path",))
        with self.assertRaisesRegex(ValueError, "deterministic rule"):
            SemanticSFTSample.create(
                role="action",
                expected_output_kind="tool_call",
                expected_tool_call_count=1,
                messages=(CanonicalMessage("user", "Read src/a.py"),),
                tools=(tool,),
                target=CanonicalMessage(
                    "assistant",
                    "",
                    tool_calls=(CanonicalToolCall(
                        "call_wrong",
                        "read_file",
                        canonical_json_bytes({"path": "src/a.py"}).decode(),
                    ),),
                ),
                metadata=_metadata("wrong-id"),
            )
        with self.assertRaisesRegex(ValueError, "hidden reasoning"):
            SemanticSFTSample.create(
                role="action",
                expected_output_kind="tool_call",
                expected_tool_call_count=1,
                messages=(CanonicalMessage("user", "Read src/a.py"),),
                tools=(tool,),
                target=CanonicalMessage(
                    "assistant",
                    "I should inspect it.",
                    tool_calls=(_call(0, "read_file", {"path": "src/a.py"}),),
                ),
                metadata=_metadata("hidden-reason"),
            )


def _action_sample() -> SemanticSFTSample:
    return SemanticSFTSample.create(
        role="action",
        expected_output_kind="tool_call",
        expected_tool_call_count=1,
        messages=(
            CanonicalMessage("system", "Use the available repository tools."),
            CanonicalMessage("user", "Inspect src/foo.py."),
        ),
        tools=(_tool("read_file", required=("path",)),),
        target=CanonicalMessage(
            "assistant",
            "",
            tool_calls=(_call(0, "read_file", {"path": "src/foo.py"}),),
        ),
        metadata=_metadata("action"),
    )


def _assistant_text_sample() -> SemanticSFTSample:
    return SemanticSFTSample.create(
        role="action",
        expected_output_kind="assistant_text",
        expected_tool_call_count=None,
        messages=(CanonicalMessage("user", "Finish after validation."),),
        tools=(),
        target=CanonicalMessage("assistant", "Validation passed; the task is complete."),
        metadata=_metadata("finish"),
    )


def _selection_sample() -> SemanticSFTSample:
    content = canonical_json_bytes({
        "selected_skills": ["RETRIEVED_SKILL_01"],
        "selected_tips": [],
        "selected_tools": [],
        "selected_trajectories": [],
        "reasoning": "relevant",
    }).decode()
    return SemanticSFTSample.create(
        role="selection",
        expected_output_kind="selection_json",
        expected_tool_call_count=None,
        messages=(CanonicalMessage("user", "Select relevant candidates."),),
        tools=(),
        target=CanonicalMessage("assistant", content),
        metadata=_metadata("selection"),
    )


def _writing_sample() -> SemanticSFTSample:
    return SemanticSFTSample.create(
        role="writing",
        expected_output_kind="writing_json",
        expected_tool_call_count=None,
        messages=(CanonicalMessage("user", "Write reusable records or return none."),),
        tools=(),
        target=CanonicalMessage("assistant", "[]"),
        metadata=_metadata("writing"),
    )


def _maintenance_sample() -> SemanticSFTSample:
    tools = formal_maintenance_tools()
    return SemanticSFTSample.create(
        role="maintenance",
        expected_output_kind="maintenance_tool_call",
        expected_tool_call_count=1,
        messages=(CanonicalMessage("user", "Finish maintenance."),),
        tools=tools,
        target=CanonicalMessage(
            "assistant",
            "",
            tool_calls=(_call(0, "finish", {"summary": "No changes needed."}),),
        ),
        metadata=_metadata("maintenance"),
    )


def _tool(name: str, *, required: tuple[str, ...]) -> CanonicalTool:
    parameters = {
        "type": "object",
        "properties": {field: {"type": "string"} for field in required},
        "required": list(required),
        "additionalProperties": False,
    }
    return CanonicalTool(
        name,
        f"Use {name}.",
        canonical_json_bytes(parameters).decode(),
        canonical_sha256(parameters),
    )


def _call(index: int, name: str, arguments: dict) -> CanonicalToolCall:
    arguments_json = canonical_json_bytes(arguments).decode()
    return CanonicalToolCall(
        deterministic_tool_call_id(
            call_index=index,
            name=name,
            arguments=arguments_json,
        ),
        name,
        arguments_json,
    )


def _metadata(source_id: str) -> dict:
    return {
        "source": "contract_fixture",
        "source_id": source_id,
        "task_group": "fixture",
        "repository_key": "fixture-repository",
        "quality_status": "accepted",
    }


if __name__ == "__main__":
    unittest.main()
