from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from my_agent.tools import ToolExecutionResult, ToolInvocation


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

    def status(self, status: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...


class PlainRenderer:
    def __init__(self, output: TextIO | None = None, errors: TextIO | None = None) -> None:
        self.output = output or sys.stdout
        self.errors = errors or sys.stderr

    def banner(self, info: StartupInfo) -> None:
        self.output.write(
            "\n".join(
                [
                    f"my-agent {info.version}",
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

    def status(self, status: str) -> None:
        self.output.write(status.rstrip() + "\n")

    def error(self, message: str) -> None:
        self.errors.write(message.rstrip() + "\n")


class AnsiRenderer(PlainRenderer):
    def banner(self, info: StartupInfo) -> None:
        self.output.write("\033[1m")
        super().banner(info)
        self.output.write("\033[0m")
