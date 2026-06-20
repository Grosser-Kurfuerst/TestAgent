from __future__ import annotations

from enum import Enum


class AgentMode(str, Enum):
    REACT = "react"
    PLAN = "plan"
    TEAM = "team"
    AUTO = "auto"


def normalize_mode(value: AgentMode | str | None, *, default: AgentMode = AgentMode.REACT) -> AgentMode:
    if isinstance(value, AgentMode):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    for mode in AgentMode:
        if mode.value == normalized:
            return mode
    allowed = ", ".join(mode.value for mode in AgentMode)
    raise ValueError(f"Unknown agent mode {value!r}. Expected one of: {allowed}.")


def resolve_mode(value: AgentMode | str | None, task: str, *, default: AgentMode = AgentMode.REACT) -> AgentMode:
    mode = normalize_mode(value, default=default)
    if mode != AgentMode.AUTO:
        return mode
    return AgentMode.PLAN if should_use_plan(task) else AgentMode.REACT


def should_use_plan(task: str) -> bool:
    text = task.strip()
    if len(text) >= 180:
        return True

    multi_step_cues = ["先", "然后", "再", "最后", "同时", "以及", "并且", "分步骤", "计划", "迁移"]
    if any(cue in text for cue in multi_step_cues):
        return True

    complex_verbs = ["实现", "修复", "重构", "设计", "生成方案", "新增", "移除", "改造", "验收"]
    if sum(1 for cue in complex_verbs if cue in text) >= 2:
        return True

    file_scope_cues = ["多个文件", "整个项目", "模块", "架构", "测试与验收", "step5-8", "阶段5-8"]
    return any(cue in text for cue in file_scope_cues)
