from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from my_agent.plan import PlanState, PlanStatus, PlanTask, TaskStatus
from my_agent.team import ExecutionStep, StepStatus, TeamState, TeamStatus
from my_agent.tools import ToolExecutionResult, ToolInvocation


TURA_CLI_BANNER = """\
╔════════════════════════════════════════════════════════════════════════╗
║         ████████╗██╗   ██╗██████╗  █████╗  ██████╗██╗     ██╗          ║
║         ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔════╝██║     ██║          ║
║            ██║   ██║   ██║██████╔╝███████║██║     ██║     ██║          ║
║            ██║   ██║   ██║██╔══██╗██╔══██║██║     ██║     ██║          ║
║            ██║   ╚██████╔╝██║  ██║██║  ██║╚██████╗███████╗██║          ║
║            ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝          ║
║                          TuraCLI   v0.1.0                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝"""


@dataclass(frozen=True)
class StartupInfo:
    version: str
    repo_path: Path
    provider: str
    model: str
    tool_summary: str
    trace_dir: Path
    limits: str


class Renderer(Protocol):
    def banner(self, info: StartupInfo) -> None:
        ...

    def user_prompt(self) -> str:
        ...

    def assistant_delta(self, text: str) -> None:
        ...

    def tool_call_started(self, invocation: ToolInvocation) -> None:
        ...

    def tool_call_completed(self, result: ToolExecutionResult) -> None:
        ...

    def plan_started(self, plan: PlanState) -> None:
        ...

    def plan_task_updated(self, task: PlanTask, *, plan_id: str) -> None:
        ...

    def plan_completed(self, plan: PlanState) -> None:
        ...

    def team_started(self, team: TeamState) -> None:
        ...

    def team_step_updated(self, step: ExecutionStep, *, team_id: str) -> None:
        ...

    def team_completed(self, team: TeamState) -> None:
        ...

    def status(self, status: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...


TASK_STATUS_ICON = {
    TaskStatus.PENDING: "○",
    TaskStatus.READY: "◌",
    TaskStatus.RUNNING: "▶",
    TaskStatus.SUCCEEDED: "✓",
    TaskStatus.FAILED: "✗",
    TaskStatus.SKIPPED: "-",
    TaskStatus.CANCELLED: "!",
}

TASK_STATUS_ASCII = {
    TaskStatus.PENDING: "[pending]",
    TaskStatus.READY: "[ready]",
    TaskStatus.RUNNING: "[running]",
    TaskStatus.SUCCEEDED: "[ok]",
    TaskStatus.FAILED: "[failed]",
    TaskStatus.SKIPPED: "[skipped]",
    TaskStatus.CANCELLED: "[cancelled]",
}

STEP_STATUS_ICON = {
    StepStatus.PENDING: "○",
    StepStatus.READY: "◌",
    StepStatus.RUNNING: "▶",
    StepStatus.REVIEWING: "?",
    StepStatus.COMPLETED: "✓",
    StepStatus.FAILED: "✗",
    StepStatus.SKIPPED: "-",
    StepStatus.CANCELLED: "!",
}

STEP_STATUS_ASCII = {
    StepStatus.PENDING: "[pending]",
    StepStatus.READY: "[ready]",
    StepStatus.RUNNING: "[running]",
    StepStatus.REVIEWING: "[reviewing]",
    StepStatus.COMPLETED: "[ok]",
    StepStatus.FAILED: "[failed]",
    StepStatus.SKIPPED: "[skipped]",
    StepStatus.CANCELLED: "[cancelled]",
}


class PlainRenderer:
    def __init__(
        self,
        output: TextIO | None = None,
        errors: TextIO | None = None,
        *,
        unicode_icons: bool = True,
    ) -> None:
        self.output = output or sys.stdout
        self.errors = errors or sys.stderr
        self.unicode_icons = unicode_icons

    def banner(self, info: StartupInfo) -> None:
        self.output.write(
            "\n".join(
                [
                    TURA_CLI_BANNER,
                    f"version: {info.version}",
                    f"repo: {info.repo_path}",
                    f"provider/model: {info.provider}/{info.model}",
                    f"tools: {info.tool_summary}",
                    f"trace dir: {info.trace_dir}",
                    f"limits: {info.limits}",
                    "",
                    "Type /help for commands, /tools for enabled tools, /quit to exit.",
                ]
            )
            + "\n"
        )

    def user_prompt(self) -> str:
        return "agentcli> "

    def assistant_delta(self, text: str) -> None:
        self.output.write(text.rstrip() + "\n")

    def tool_call_started(self, invocation: ToolInvocation) -> None:
        self.output.write(f"tool started: {invocation.name}\n")

    def tool_call_completed(self, result: ToolExecutionResult) -> None:
        status = "ok" if result.ok else "failed"
        suffix = f" ({result.error_code})" if result.error_code else ""
        self.output.write(f"tool completed: {result.name} {status}{suffix}\n")

    def plan_started(self, plan: PlanState) -> None:
        self.output.write(f"plan started: {plan.id} {len(plan.tasks)} tasks\n")
        if plan.summary:
            self.output.write(f"summary: {plan.summary}\n")
        for task in plan.tasks:
            self.output.write(f"{self._task_icon(task.status)} {task.id} {task.type.value} {task.title}\n")

    def plan_task_updated(self, task: PlanTask, *, plan_id: str) -> None:
        self.output.write(f"{self._task_icon(task.status)} {task.id} {task.status.value} {task.title}\n")

    def plan_completed(self, plan: PlanState) -> None:
        counts = _status_counts(plan)
        suffix = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        label = _plan_label(plan.status)
        self.output.write(f"plan {label}: {suffix or 'no tasks'}\n")

    def team_started(self, team: TeamState) -> None:
        self.output.write(f"team started: {team.id} {len(team.steps)} steps\n")
        if team.summary:
            self.output.write(f"summary: {team.summary}\n")
        for step in team.steps:
            self.output.write(f"{self._step_icon(step.status)} {step.id} {step.type.value} {step.title}\n")

    def team_step_updated(self, step: ExecutionStep, *, team_id: str) -> None:
        worker = f" {step.worker_name}" if step.worker_name and step.status == StepStatus.RUNNING else ""
        self.output.write(f"{self._step_icon(step.status)} {step.id} {step.status.value}{worker} {step.title}\n")

    def team_completed(self, team: TeamState) -> None:
        counts = _team_status_counts(team)
        suffix = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        self.output.write(f"team {_team_label(team.status)}: {suffix or 'no steps'}\n")

    def status(self, status: str) -> None:
        self.output.write(status.rstrip() + "\n")

    def error(self, message: str) -> None:
        self.errors.write(message.rstrip() + "\n")

    def _task_icon(self, status: TaskStatus) -> str:
        return (TASK_STATUS_ICON if self.unicode_icons else TASK_STATUS_ASCII)[status]

    def _step_icon(self, status: StepStatus) -> str:
        return (STEP_STATUS_ICON if self.unicode_icons else STEP_STATUS_ASCII)[status]


class AnsiRenderer(PlainRenderer):
    def banner(self, info: StartupInfo) -> None:
        self.output.write("\033[1m")
        super().banner(info)
        self.output.write("\033[0m")


def _status_counts(plan: PlanState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in plan.tasks:
        counts[task.status.value] = counts.get(task.status.value, 0) + 1
    return counts


def _team_status_counts(team: TeamState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in team.steps:
        counts[step.status.value] = counts.get(step.status.value, 0) + 1
    return counts


def _plan_label(status: PlanStatus) -> str:
    if status == PlanStatus.SUCCEEDED:
        return "succeeded"
    if status == PlanStatus.FAILED:
        return "failed"
    if status == PlanStatus.CANCELLED:
        return "cancelled"
    return status.value


def _team_label(status: TeamStatus) -> str:
    if status == TeamStatus.SUCCEEDED:
        return "succeeded"
    if status == TeamStatus.FAILED:
        return "failed"
    if status == TeamStatus.CANCELLED:
        return "cancelled"
    return status.value
