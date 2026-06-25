from __future__ import annotations

from concurrent.futures import wait
from dataclasses import replace
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from my_agent.cancellation import CancelledError, CancellationToken
from my_agent.parallel import create_bounded_executor, shutdown_executor
from my_agent.schema import ToolResult
from my_agent.tools.execution import ToolExecutionResult, ToolInvocation
from my_agent.tools.hooks import HookViolation, validate_tool_call_preflight
from my_agent.tools.parallel_policy import ToolBatchGroup, build_tool_batch_groups
from my_agent.tools.spec import ToolContext, ToolRegistration, ToolSpec
from my_agent.tools.validation import validate_arguments_schema


MAX_PARALLEL_TOOLS = 4


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any], ToolContext], ToolResult]
    preflight: Callable[[dict[str, Any], ToolContext], None] | None = None
    resource_resolver: Callable[[dict[str, Any], ToolContext], set[str]] | None = None
    parallel_side_effect_safe: bool = False
    cancellation_safe: bool = False

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
        self._local = threading.local()
        self.last_execution_summary: dict[str, object] = {"groups": []}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    @property
    def tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def get_registered(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

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
            resource_resolver=registration.resource_resolver,
            parallel_side_effect_safe=registration.parallel_side_effect_safe,
            cancellation_safe=registration.cancellation_safe,
        )

    def load_source(self, source: Any, context: ToolContext | None = None) -> None:
        load_context = context or self.context
        for registration in source.load(load_context):
            self.register(registration)

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.spec.as_openai_tool() for tool in self._tools.values() if tool.spec.enabled]

    def execute(self, invocations: list[ToolInvocation]) -> list[ToolExecutionResult]:
        return self.execute_tools(invocations)

    def execute_tools(self, invocations: list[ToolInvocation]) -> list[ToolExecutionResult]:
        if not invocations:
            self.last_execution_summary = {"groups": []}
            return []
        token = self._active_context().cancellation_token
        if token is not None and token.is_cancelled():
            self.last_execution_summary = {
                "groups": [
                    {
                        "ids": [invocation.id],
                        "parallel": False,
                        "side_effect_free": False,
                        "reason": "cancelled",
                        "max_workers": 1,
                        "timeout_seconds": self._active_context().tool_batch_timeout_seconds,
                    }
                    for invocation in invocations
                ]
            }
            return [_cancelled_result(invocation, token.reason) for invocation in invocations]
        if len(invocations) == 1:
            self.last_execution_summary = {
                "groups": [
                    {
                        "ids": [invocations[0].id],
                        "parallel": False,
                        "side_effect_free": False,
                        "reason": "single_tool",
                        "max_workers": 1,
                        "timeout_seconds": self._active_context().tool_batch_timeout_seconds,
                    }
                ]
            }
            return [self._execute_one(invocations[0])]

        parsed_arguments = self._parsed_arguments_for_grouping(invocations)
        groups = build_tool_batch_groups(
            invocations,
            registry=self,
            parsed_arguments=parsed_arguments,
            context=self._active_context(),
        )
        self.last_execution_summary = {
            "groups": [
                {
                    "ids": [invocation.id for invocation in group.invocations],
                    "parallel": group.parallel and len(group.invocations) > 1,
                    "side_effect_free": group.side_effect_free,
                    "reason": group.reason,
                    "max_workers": min(
                        len(group.invocations),
                        self._active_context().max_parallel_tools,
                        MAX_PARALLEL_TOOLS,
                    )
                    if group.parallel
                    else 1,
                    "timeout_seconds": self._active_context().tool_batch_timeout_seconds,
                }
                for group in groups
            ]
        }
        results: list[ToolExecutionResult] = []
        for group in groups:
            if group.parallel and len(group.invocations) > 1:
                results.extend(self._execute_group_parallel(group))
            else:
                results.extend(self._execute_group_serial(group))
        return results

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
                tool.preflight(dict(arguments), self._active_context())
            else:
                validate_tool_call_preflight(
                    tool_name=tool.spec.name,
                    arguments=arguments,
                    repo_root=self._active_context().repo_root,
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
            context = self._active_context()
            if context.cancellation_token is not None:
                context.cancellation_token.raise_if_cancelled()
            result = tool.handler(dict(arguments), context)
        except CancelledError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return ToolExecutionResult(
                id=invocation.id,
                name=invocation.name,
                ok=False,
                content=f"Tool cancelled: {exc}",
                elapsed_ms=elapsed_ms,
                error_code="cancelled",
                retryable=False,
            )
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

    def _execute_group_serial(self, group: ToolBatchGroup) -> list[ToolExecutionResult]:
        return [self._execute_one(invocation) for invocation in group.invocations]

    def _execute_group_parallel(self, group: ToolBatchGroup) -> list[ToolExecutionResult]:
        parent_token = self._active_context().cancellation_token or CancellationToken()
        batch_token = parent_token.child()
        batch_context = replace(self._active_context(), cancellation_token=batch_token)
        max_workers = min(len(group.invocations), self._active_context().max_parallel_tools, MAX_PARALLEL_TOOLS)
        executor = create_bounded_executor(max_workers=max_workers, thread_name_prefix="agentcli-tools")
        future_by_index = {
            index: executor.submit(self._execute_one_with_context, invocation, batch_context)
            for index, invocation in enumerate(group.invocations)
        }
        try:
            done, not_done = wait(
                set(future_by_index.values()),
                timeout=self._active_context().tool_batch_timeout_seconds,
            )
            timed_out = set(not_done)
            if not_done:
                batch_token.cancel("tool_batch_timeout")
                done_after_grace, still_running = wait(
                    not_done,
                    timeout=self._active_context().tool_shutdown_grace_seconds,
                )
                done = done.union(done_after_grace)
                if still_running and not group.side_effect_free:
                    parent_token.cancel("tool_batch_timeout")
                    done_after_cancel, _ = wait(still_running)
                    done = done.union(done_after_cancel)

            results: list[ToolExecutionResult] = []
            for index, invocation in enumerate(group.invocations):
                future = future_by_index[index]
                if future in timed_out:
                    results.append(_timeout_result(invocation, self._active_context().tool_batch_timeout_seconds))
                elif future in done and not future.cancelled():
                    try:
                        results.append(future.result())
                    except Exception as exc:  # noqa: BLE001 - tool boundary converts worker failures
                        results.append(_exception_result(invocation, exc))
                else:
                    results.append(_timeout_result(invocation, self._active_context().tool_batch_timeout_seconds))
            return results
        finally:
            shutdown_executor(executor)

    def _execute_one_with_context(self, invocation: ToolInvocation, context: ToolContext) -> ToolExecutionResult:
        previous = getattr(self._local, "context", None)
        self._local.context = context
        try:
            return self._execute_one(invocation)
        finally:
            if previous is None:
                try:
                    del self._local.context
                except AttributeError:
                    pass
            else:
                self._local.context = previous

    def _active_context(self) -> ToolContext:
        return getattr(self._local, "context", self.context)

    def _parsed_arguments_for_grouping(self, invocations: list[ToolInvocation]) -> dict[str, dict[str, object]]:
        parsed: dict[str, dict[str, object]] = {}
        for invocation in invocations:
            try:
                parsed[invocation.id] = invocation.arguments()
            except ValueError:
                continue
        return parsed


def _cancelled_result(invocation: ToolInvocation, reason: str = "cancelled") -> ToolExecutionResult:
    return ToolExecutionResult(
        id=invocation.id,
        name=invocation.name,
        ok=False,
        content=f"Tool call cancelled: {reason or 'cancelled'}",
        error_code="cancelled",
        retryable=False,
    )


def _timeout_result(invocation: ToolInvocation, timeout: int) -> ToolExecutionResult:
    return ToolExecutionResult(
        id=invocation.id,
        name=invocation.name,
        ok=False,
        content=f"Tool batch timed out after {timeout}s; this tool was cancelled.",
        error_code="tool_batch_timeout",
        retryable=True,
        timed_out=True,
    )


def _exception_result(invocation: ToolInvocation, exc: Exception) -> ToolExecutionResult:
    return ToolExecutionResult(
        id=invocation.id,
        name=invocation.name,
        ok=False,
        content=f"Tool error: {type(exc).__name__}: {exc}",
        error_code="exception",
        retryable=False,
    )
