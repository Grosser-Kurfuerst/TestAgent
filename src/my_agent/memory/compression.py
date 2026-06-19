from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from my_agent.llm import AgentLLM
from my_agent.llm.types import Message
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType


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

FACT_EXTRACTION_PROMPT = """请从以下对话中提取跨会话仍然成立、未来复用仍有价值的稳定事实。
只输出 JSON 数组，每个元素包含 content、scope、confidence。

可以提取：
- 用户长期偏好和工作习惯
- 项目名称、路径、技术栈、命令、环境变量
- 已确认的重要设计决策和约定

不要提取：
- 当前这一轮的临时任务、todo、步骤
- 模型自己的猜测、道歉、提醒
- 未经用户确认的推断
- “用户让我...” 这种请求句

scope 只能是 "global" 或 "project"。
confidence 范围 0-1，低于 0.7 的事实不要输出。

=== 对话 ===
{conversation}
"""

_BAD_FACT_SNIPPETS = (
    "可能",
    "猜测",
    "推测",
    "本次任务",
    "当前这一轮",
    "临时",
    "todo",
    "待办",
    "用户让我",
    "用户请",
    "请你",
    "让我",
    "maybe",
    "possibly",
    "guess",
    "temporary",
    "todo",
    "current task",
    "this task",
    "user asked",
    "user requested",
)
_DURABLE_FACT_HINTS = (
    "用户偏好",
    "用户习惯",
    "偏好",
    "习惯",
    "喜欢",
    "倾向",
    "项目",
    "仓库",
    "路径",
    "技术栈",
    "命令",
    "配置",
    "环境变量",
    "约定",
    "规则",
    "默认",
    "使用",
)
_DURABLE_FACT_HINTS_EN = (
    "user prefers",
    "user preference",
    "project",
    "repository",
    "repo",
    "path",
    "uses",
    "built with",
    "framework",
    "library",
    "stack",
    "api",
    "command",
    "environment variable",
    "env var",
    "configuration",
    "config",
    "convention",
    "rule",
    "default",
    "python",
    "fastapi",
    "pytest",
    "unittest",
)


class MemoryCompressor:
    """Map-reduce short-term memory compressor and conservative fact extractor."""

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

    def extract_facts(
        self,
        entries: list[MemoryEntry],
        *,
        reason: str,
        project_key: str,
        run_id: str = "",
    ) -> list[MemoryEntry]:
        """Extract durable facts from conversation entries.

        Extraction is intentionally conservative. LLM errors and empty
        responses return an empty list so memory maintenance cannot interrupt
        the agent loop.
        """
        if not entries or self.llm is None:
            return []

        prompt = FACT_EXTRACTION_PROMPT.format(conversation=_render_entries(entries, limit=self.max_input_chars))
        try:
            response = self.llm.chat(
                [
                    Message(role="system", content="你是一个信息提取助手，只输出 JSON 数组。"),
                    Message(role="user", content=prompt),
                ],
                tools=None,
            )
        except Exception:
            return []

        facts = _parse_fact_response(response.content)
        entries_out: list[MemoryEntry] = []
        seen: set[str] = set()
        for content, scope, confidence in facts:
            normalized = _normalize_fact(content)
            if normalized in seen or not _is_persistent_fact_candidate(normalized, confidence):
                continue
            seen.add(normalized)
            memory_scope = MemoryScope.GLOBAL if scope == "global" else MemoryScope.PROJECT
            entries_out.append(
                MemoryEntry.build(
                    id=f"fact_auto_{uuid4().hex[:12]}",
                    content=normalized,
                    type=MemoryType.FACT,
                    scope=memory_scope,
                    source="fact_extractor",
                    token_count=estimate_tokens(normalized),
                    project_key="" if memory_scope == MemoryScope.GLOBAL else project_key,
                    run_id=run_id,
                    metadata={
                        "source": "fact_extractor",
                        "reason": reason,
                        "run_id": run_id,
                        "confidence": confidence,
                    },
                )
            )
        return entries_out

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


def _parse_fact_response(text: str) -> list[tuple[str, str, float]]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return _parse_fact_lines(stripped)

    facts: list[tuple[str, str, float]] = []
    if not isinstance(parsed, list):
        return facts
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", ""))
        scope = str(item.get("scope", "project")).strip().lower()
        confidence = _as_float(item.get("confidence"), 0.0)
        facts.append((content, "global" if scope == "global" else "project", confidence))
    return facts


def _parse_fact_lines(text: str) -> list[tuple[str, str, float]]:
    facts: list[tuple[str, str, float]] = []
    for raw in text.splitlines():
        line = _normalize_fact(raw)
        if line:
            facts.append((line, "project", 0.8))
    return facts


def _normalize_fact(text: str) -> str:
    fact = text.strip()
    fact = re.sub(r"^[-*•]\s*", "", fact)
    fact = re.sub(r"^\d+[.)]\s*", "", fact)
    return " ".join(fact.split())


def _is_persistent_fact_candidate(fact: str, confidence: float) -> bool:
    if confidence < 0.7:
        return False
    if len(fact) < 6:
        return False
    lowered = fact.lower()
    if any(snippet in fact or snippet in lowered for snippet in _BAD_FACT_SNIPPETS):
        return False
    return any(hint in fact for hint in _DURABLE_FACT_HINTS) or _has_english_durable_hint(lowered)


def _has_english_durable_hint(lowered_fact: str) -> bool:
    for hint in _DURABLE_FACT_HINTS_EN:
        pattern = r"\b" + re.escape(hint).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, lowered_fact):
            return True
    return False


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _single_line(text: str, limit: int) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... truncated"


__all__ = ["MemoryCompressor"]
