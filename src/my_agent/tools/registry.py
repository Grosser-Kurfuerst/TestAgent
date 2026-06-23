from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from my_agent.schema import ToolResult
from my_agent.tools.execution import ToolExecutionResult, ToolInvocation
from my_agent.tools.hooks import HookViolation, validate_tool_call_preflight
from my_agent.tools.spec import ToolContext, ToolRegistration, ToolSpec
from my_agent.tools.validation import validate_arguments_schema


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any], ToolContext], ToolResult]
    preflight: Callable[[dict[str, Any], ToolContext], None] | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description


class ToolRegistry:
    def __init__(self, context: ToolContext | None = None) -> None:
        self.context = context or ToolContext(repo_root=Path.cwd())
        self._tools: dict[str, RegisteredTool] = {}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    @property
    def tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def register(
        self,
        registration: ToolRegistration,
        *,
        allow_override: bool = False,
    ) -> None:
        existing = self._tools.get(registration.spec.name)
        if existing is not None and not allow_override:
            raise ValueError(
                f"Tool already registered: {registration.spec.name} "
                f"(existing source={existing.spec.source}, new source={registration.spec.source})."
            )
        self._tools[registration.spec.name] = RegisteredTool(
            spec=registration.spec,
            handler=registration.handler,
            preflight=registration.preflight,
        )

    def load_source(self, source: Any, context: ToolContext | None = None) -> None:
        load_context = context or self.context
        for registration in source.load(load_context):
            self.register(registration)

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.spec.as_openai_tool() for tool in self._tools.values() if tool.spec.enabled]

    def execute(self, invocations: list[ToolInvocation]) -> list[ToolExecutionResult]:
        return [self._execute_one(invocation) for invocation in invocations]

    def _execute_one(self, invocation: ToolInvocation) -> ToolExecutionResult:
        resolved = self._resolve_tool(invocation)
        if isinstance(resolved, ToolExecutionResult):
            return resolved
        tool = resolved

        parsed = self._parse_arguments(invocation)
        if isinstance(parsed, ToolExecutionResult):
            return parsed
        arguments = parsed

        schema_result = self._validate_schema(invocation, tool, arguments)
        if schema_result is not None:
            return schema_result

        policy_result = self._preflight_policy(invocation, tool, arguments)
        if policy_result is not None:
            return policy_result

        return self._execute_registered(invocation, tool, arguments)

    def _resolve_tool(self, invocation: ToolInvocation) -> RegisteredTool | ToolExecutionResult:
        tool = self._tools.get(invocation.name)
        if tool is None or not tool.spec.enabled:
            return ToolExecutionResult(
                id=invocation.id,
                name=invocation.name,
                ok=False,
                content=f"Unknown tool: {invocation.name}",
                error_code="unknown_tool",
            )
        return tool

    def _parse_arguments(self, invocation: ToolInvocation) -> dict[str, Any] | ToolExecutionResult:
        try:
            return invocation.arguments()
        except ValueError as exc:
            return ToolExecutionResult(
                id=invocation.id,
                name=invocation.name,
                ok=False,
                content=str(exc),
                error_code="invalid_arguments",
                retryable=True,
            )

    def _validate_schema(
        self,
        invocation: ToolInvocation,
        tool: RegisteredTool,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult | None:
        schema_errors = validate_arguments_schema(tool.spec.parameters, arguments)
        if not schema_errors:
            return None
        return ToolExecutionResult(
            id=invocation.id,
            name=invocation.name,
            ok=False,
            content="Tool arguments failed schema validation: " + "; ".join(schema_errors),
            error_code="invalid_arguments_schema",
            retryable=True,
        )

    def _preflight_policy(
        self,
        invocation: ToolInvocation,
        tool: RegisteredTool,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult | None:
        try:
            if tool.preflight is not None:
                tool.preflight(dict(arguments), self.context)
            else:
                validate_tool_call_preflight(
                    tool_name=tool.spec.name,
                    arguments=arguments,
                    repo_root=self.context.repo_root,
                )
        except (HookViolation, ValueError) as exc:
            return ToolExecutionResult(
                id=invocation.id,
                name=invocation.name,
                ok=False,
                content=f"[POLICY] Operation denied before approval: {exc}",
                error_code="policy_denied",
                retryable=False,
                blocked=True,
            )
        return None

    def _execute_registered(
        self,
        invocation: ToolInvocation,
        tool: RegisteredTool,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        started = time.monotonic()
        try:
            result = tool.handler(dict(arguments), self.context)
        except HookViolation as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return ToolExecutionResult.from_tool_result(
                invocation,
                ToolResult(ok=False, output=str(exc), blocked=True, reason=str(exc)),
                elapsed_ms=elapsed_ms,
                error_code="blocked",
                retryable=False,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary must convert runtime errors into observations
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return ToolExecutionResult(
                id=invocation.id,
                name=invocation.name,
                ok=False,
                content=f"Tool error: {type(exc).__name__}: {exc}",
                elapsed_ms=elapsed_ms,
                error_code="exception",
                retryable=False,
            )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        if not isinstance(result, ToolResult):
            return ToolExecutionResult(
                id=invocation.id,
                name=invocation.name,
                ok=False,
                content=f"Tool error: {invocation.name} returned an invalid result.",
                elapsed_ms=elapsed_ms,
                error_code="invalid_result",
            )
        return ToolExecutionResult.from_tool_result(invocation, result, elapsed_ms=elapsed_ms)
