from __future__ import annotations

import argparse
import sys
import threading
from queue import Empty, Queue

from my_agent.cancellation import CancellationToken
from my_agent.cli.common import (
    CliContext,
    DEFAULT_TASK_FILE,
    load_task,
    positive_max_steps,
    section,
)


FAILURE_STOP_REASONS = {
    "plan_failed",
    "plan_validation_failed",
    "plan_cancelled",
    "team_failed",
    "team_validation_failed",
    "team_cancelled",
    "team_planner_failed",
    "context_over_budget",
}


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run", help="Run the coding agent runtime.")
    parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to a task JSON file.")
    parser.add_argument("--repo", help="Override repository path from the task file.")
    parser.add_argument("--task", help="Override task text from the task file.")
    parser.add_argument("--test-command", help="Override test command from the task file.")
    parser.add_argument("--max-steps", type=positive_max_steps, help="Maximum actor tool calls.")
    parser.add_argument("--trace-dir", help="Directory for JSONL traces.")
    run_hitl = parser.add_mutually_exclusive_group()
    run_hitl.add_argument("--hitl", dest="hitl", action="store_true", default=None, help="Enable HITL approvals.")
    run_hitl.add_argument("--no-hitl", dest="hitl", action="store_false", help="Disable HITL approvals.")
    parser.add_argument(
        "--mode",
        choices=("react", "plan", "team", "auto"),
        default=None,
        help="Execution mode. Default: AGENTCLI_AGENT_MODE or auto.",
    )
    parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    task_payload = load_task(args.task_file)
    try:
        config = ctx.config_from_env(require_env_file=False)
        config = ctx.with_hitl_flag(config, args.hitl)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    repo_path = ctx.resolve_repo_path(args.repo or task_payload["repo"])
    test_command = args.test_command if args.test_command is not None else task_payload.get("test_command")
    trace_dir = ctx.resolve_trace_dir(args.trace_dir, config.trace_dir)
    cancellation_token = CancellationToken()
    interrupted = False
    result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

    def run_in_background() -> None:
        try:
            result_queue.put(
                (
                    "ok",
                    ctx.run_agent(
                        repo_path=repo_path,
                        task=args.task or task_payload["task"],
                        test_command=test_command,
                        config=config,
                        max_steps=args.max_steps,
                        trace_dir=trace_dir,
                        mode=args.mode,
                        cancellation_token=cancellation_token,
                        event_sink=ctx.run_event_sink,
                    ),
                )
            )
        except BaseException as exc:  # noqa: BLE001 - thread boundary relays errors to main.
            result_queue.put(("error", exc))

    worker = threading.Thread(target=run_in_background, name="agentcli-run", daemon=True)
    worker.start()
    try:
        try:
            result_kind, result_payload = result_queue.get()
        except KeyboardInterrupt:
            interrupted = True
            cancellation_token.cancel("keyboard_interrupt")
            try:
                result_kind, result_payload = result_queue.get(timeout=max(0, config.tool_shutdown_grace_seconds))
            except (KeyboardInterrupt, Empty):
                print("Cancelled.", file=sys.stderr)
                return 130
        if result_kind == "error":
            if isinstance(result_payload, KeyboardInterrupt):
                cancellation_token.cancel("keyboard_interrupt")
                print("Cancelled.", file=sys.stderr)
                return 130
            if isinstance(result_payload, (RuntimeError, ValueError)):
                print(f"Error: {result_payload}", file=sys.stderr)
                return 1
            if isinstance(result_payload, BaseException):
                raise result_payload
            raise RuntimeError(result_payload)
        final_state = result_payload
        print(section("Plan", final_state.plan))
        print()
        print(section("Review", final_state.review))
        print()
        print(section("Final summary", final_state.final_answer))
        print()
        print(f"Trace: {final_state.trace_path}")
        if interrupted:
            return 130
        if getattr(final_state, "stop_reason", "") in FAILURE_STOP_REASONS:
            return 1
        return 0
    finally:
        ctx.close_mcp_servers()
