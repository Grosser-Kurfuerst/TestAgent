from __future__ import annotations

# ruff: noqa: F401 - imported names are re-exported to split test modules

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager
from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, MessageLike
from my_agent.memory.disabled import NoopMemoryManager
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.evolver.writing.contracts import (
    ExperienceWriteProposal,
    ExperienceWriteResult,
)
from my_agent.memory.experience.serialization import experience_to_dict
from my_agent.memory.manager import MemoryManager
from my_agent.memory.store_errors import MemoryStoreLoadError
from my_agent.memory.types import MemoryScope
from my_agent.tools import ToolExecutionResult
from tests.memory.experience.fixtures import save_typed_experience


NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


def _config(memory_dir: Path, **overrides: object) -> AgentConfig:
    values: dict[str, object] = {
        "provider": "fake",
        "api_key": "",
        "base_url": None,
        "model": "fake",
        "temperature": 0.0,
        "max_steps": 8,
        "command_timeout": 60,
        "trace_dir": Path("traces"),
        "use_fake_llm": True,
        "memory_dir": memory_dir,
        "memory_context_tokens": 200,
        "memory_retrieval_limit": 8,
    }
    values.update(overrides)
    return AgentConfig(**values)


class RecordingMemoryLLM:
    supports_tools = True

    def __init__(self, *, fail_map: bool = False) -> None:
        self.fail_map = fail_map
        self.map_prompts: list[str] = []
        self.reduce_prompts: list[str] = []

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        prompt = _messages_text(messages)
        if "请压缩以下 Agent 对话片段" in prompt:
            if self.fail_map:
                raise RuntimeError("map failed")
            self.map_prompts.append(prompt)
            return ChatResponse(content=f"map summary {len(self.map_prompts)}")
        if "请合并以下多个对话摘要" in prompt:
            self.reduce_prompts.append(prompt)
            return ChatResponse(content="reduced summary")
        return ChatResponse(content="noop")


def _messages_text(messages: list[MessageLike]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, Message):
            lines.append(message.content or "")
        elif isinstance(message, dict):
            value = message.get("content", "")
            if isinstance(value, str):
                lines.append(value)
    return "\n".join(lines)


def _tool_call(name: str, call_id: str, arguments: dict[str, object] | None = None) -> LLMToolCall:
    args = arguments or {}
    return LLMToolCall(
        id=call_id,
        name=name,
        arguments=dict(args),
        arguments_json=json.dumps(args, ensure_ascii=False),
    )


def _tool_record(
    tool: str,
    *,
    ok: bool,
    output: str = "",
    reason: str = "",
    arguments: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "call": {"tool": tool, "arguments": arguments or {"command": "pytest tests/test_example.py -q"}},
        "result": {"ok": ok, "output": output or ("passed" if ok else "failed"), "reason": reason},
    }


def _append_turn(manager: MemoryManager, index: int, *, payload: str = "") -> None:
    suffix = f" {payload}" if payload else ""
    manager.append_user_message(f"user turn {index}{suffix}")
    manager.append_assistant_response(
        ChatResponse(
            content=f"assistant turn {index}{suffix}",
            tool_calls=[_tool_call("read_file", f"call_{index}", {"path": f"file_{index}.py"})],
        )
    )
    manager.append_tool_result(
        ToolExecutionResult(
            id=f"call_{index}",
            name="read_file",
            ok=True,
            content=f"result {index}{suffix}",
        )
    )


def _prepare_messages(manager: MemoryManager, **kwargs):
    return AgentContextManager(manager.context_profile).prepare_messages(memory=manager, **kwargs)

__all__ = [name for name in globals() if not name.startswith('__')]
