from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from my_agent.config import AgentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_FILE = PROJECT_ROOT / "examples" / "tasks" / "sample_task.json"


def load_task(path: str | Path = DEFAULT_TASK_FILE) -> dict[str, Any]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Task file must contain one JSON object.")
    for key in ("repo", "task"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Task file requires a non-empty {key!r} field.")
    return payload


def format_task(task: dict[str, Any]) -> str:
    lines = [
        f"id: {task.get('id', 'unknown')}",
        f"source: {task.get('source', 'local')}",
        f"repo: {task['repo']}",
        f"task: {task['task']}",
        f"test_command: {task.get('test_command') or 'not configured'}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal coding-agent scaffold.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_task_parser = subparsers.add_parser("load-task", help="Load and print a task manifest.")
    load_task_parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to a task JSON file.")

    index_parser = subparsers.add_parser("index", help="Placeholder for Phase 2 repository indexing.")
    index_parser.add_argument("--repo", required=True, help="Target repository path.")
    index_parser.add_argument("--query", default="", help="Optional retrieval query.")

    run_parser = subparsers.add_parser("run", help="Placeholder for the future agent runtime.")
    run_parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to a task JSON file.")

    config_parser = subparsers.add_parser("config", help="Print resolved local configuration.")
    config_parser.add_argument("--check-api-key", action="store_true", help="Validate provider and API key settings.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "load-task":
        print(format_task(load_task(args.task_file)))
        return 0

    if args.command == "index":
        print("Phase 2 placeholder: repository indexing is not implemented yet.")
        print(f"repo: {args.repo}")
        if args.query:
            print(f"query: {args.query}")
        return 0

    if args.command == "run":
        print("Phase 4 placeholder: agent runtime is not implemented yet.")
        print(format_task(load_task(args.task_file)))
        return 0

    if args.command == "config":
        config = AgentConfig.from_env()
        if args.check_api_key:
            config.require_api_key()
        print(
            json.dumps(
                {
                    "provider": config.provider,
                    "base_url": config.base_url,
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_steps": config.max_steps,
                    "command_timeout": config.command_timeout,
                    "trace_dir": str(config.trace_dir),
                    "use_fake_llm": config.use_fake_llm,
                    "api_key_configured": bool(config.api_key),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
