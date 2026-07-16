from __future__ import annotations

from typing import Any

from my_agent.llm import AgentLLM
from my_agent.llm.types import Message
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry


MAP_PROMPT = """请压缩以下 Agent 对话片段。保留：
1. 用户目标、限制和偏好
2. 已执行的工具调用及核心结果
3. 修改过的文件、测试结果、错误和未解决问题
4. 对后续任务仍有用的技术决策

只输出摘要，不要添加元说明。

=== 片段 ===
{chunk}
"""

REDUCE_PROMPT = """请合并以下多个对话摘要，去除重复，保留对后续推理仍重要的信息。
输出 1-5 段，按时间顺序组织。

=== 摘要 ===
{summaries}
"""

class MemoryCompressor:
    """Map-reduce compressor for short-term memory only."""

    def __init__(
        self,
        *,
        llm: AgentLLM | None,
        chunk_size: int,
        retain_recent_turns: int,
        max_input_chars: int,
    ) -> None:
        self.llm = llm
        self.chunk_size = max(1, chunk_size)
        self.retain_recent_turns = max(1, retain_recent_turns)
        self.max_input_chars = max(1, max_input_chars)

    def compact_entries(self, entries: list[MemoryEntry], *, focus: str = "") -> tuple[str, bool, int, bool]:
        """Compress entries with map-reduce.

        Returns ``(summary, fallback_used, map_count, reduce_used)``. The input
        should already be the old, compressible prefix chosen on a user-turn
        boundary by :class:`ShortTermMemory`.
        """
        if not entries:
            return "", False, 0, False

        summaries: list[str] = []
        fallback_used = False
        for chunk in _chunk_turn_groups(entries, self.chunk_size):
            chunk_text = _render_entries(chunk, limit=self.max_input_chars)
            prompt = MAP_PROMPT.format(chunk=chunk_text)
            if focus.strip():
                prompt += f"\n压缩重点：{focus.strip()}\n"
            summary, fallback = self._call_llm(
                system="你是一个对话摘要助手，只输出摘要本身。",
                prompt=prompt,
                fallback=lambda: _fallback_summary(chunk, self.max_input_chars),
            )
            if summary.strip():
                summaries.append(summary.strip())
            fallback_used = fallback_used or fallback

        if not summaries:
            return "", fallback_used, 0, False
        if len(summaries) == 1:
            return summaries[0], fallback_used, 1, False

        reduce_prompt = REDUCE_PROMPT.format(summaries="\n\n---\n\n".join(summaries))
        reduced, fallback = self._call_llm(
            system="你是一个摘要合并助手，只输出合并后的摘要。",
            prompt=reduce_prompt,
            fallback=lambda: "\n".join(summaries),
        )
        return reduced.strip(), fallback_used or fallback, len(summaries), True

    def _call_llm(self, *, system: str, prompt: str, fallback: Any) -> tuple[str, bool]:
        if self.llm is not None:
            try:
                response = self.llm.chat(
                    [Message(role="system", content=system), Message(role="user", content=prompt)],
                    tools=None,
                )
                if response.content.strip():
                    return response.content.strip(), False
            except Exception:
                pass
        return str(fallback()).strip(), True


def _chunk_turn_groups(entries: list[MemoryEntry], chunk_size: int) -> list[list[MemoryEntry]]:
    groups = _turn_groups(entries)
    chunks: list[list[MemoryEntry]] = []
    for idx in range(0, len(groups), max(1, chunk_size)):
        flattened: list[MemoryEntry] = []
        for group in groups[idx:idx + chunk_size]:
            flattened.extend(group)
        if flattened:
            chunks.append(flattened)
    return chunks


def _turn_groups(entries: list[MemoryEntry]) -> list[list[MemoryEntry]]:
    groups: list[list[MemoryEntry]] = []
    current: list[MemoryEntry] = []
    for entry in entries:
        if entry.source == "user" and current:
            groups.append(current)
            current = [entry]
        else:
            current.append(entry)
    if current:
        groups.append(current)
    return groups


def _render_entries(entries: list[MemoryEntry], *, limit: int) -> str:
    lines: list[str] = []
    chars = 0
    for entry in entries:
        line = f"{entry.source.upper()}({entry.type.value}): {_single_line(entry.content, 1200)}"
        lines.append(line)
        chars += len(line)
        if chars >= limit:
            lines.append("...(超长内容已截断)")
            break
    return "\n\n".join(lines)


def _fallback_summary(entries: list[MemoryEntry], limit: int) -> str:
    body = _render_entries(entries, limit=limit)
    return "[Fallback memory summary]\n" + body


def _single_line(text: str, limit: int) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... truncated"


__all__ = ["MemoryCompressor"]
