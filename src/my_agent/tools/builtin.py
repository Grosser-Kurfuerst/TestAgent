from __future__ import annotations

from typing import Any

from my_agent.tools.spec import ToolContext, ToolRegistration, ToolRisk, ToolSource, ToolSpec, object_schema


class BuiltinToolSource(ToolSource):
    name = "builtin"

    def __init__(self, owner: Any):
        self.owner = owner

    def load(self, context: ToolContext) -> list[ToolRegistration]:
        return [
            self._registration(
                name="list_files",
                purpose="List repository files under a file or directory path.",
                requirements=(
                    "path is optional and defaults to '.'.",
                    "path must stay inside the repository and must not target ignored or protected paths.",
                    "Use this before reading when you need to discover filenames.",
                ),
                parameters=object_schema(
                    {"path": {"type": "string", "description": "Repository-relative file or directory path."}}
                ),
                risk=ToolRisk.READ,
                handler=lambda args, ctx: self.owner._list_files(args, ctx),
                resource_resolver=lambda args, ctx: {f"path:{self.owner._resource_path(args.get('path', '.'), ctx)}"},
                cancellation_safe=True,
            ),
            self._registration(
                name="read_file",
                purpose="Read a UTF-8 text file from the repository.",
                requirements=(
                    "path is required and must be a repository-relative file path.",
                    "limit is optional and must be a positive integer.",
                    "offset is optional; when present, limit is treated as a line count.",
                    "Inspect the current file content before editing it.",
                ),
                parameters=object_schema(
                    {
                        "path": {"type": "string", "description": "Repository-relative file path."},
                        "offset": {"type": "integer", "minimum": 1, "description": "1-based start line."},
                        "limit": {"type": "integer", "minimum": 1, "description": "Character limit, or line count with offset."},
                    },
                    required=["path"],
                ),
                risk=ToolRisk.READ,
                handler=lambda args, ctx: self.owner._read_file(args, ctx),
                resource_resolver=lambda args, ctx: {f"file:{self.owner._resource_path(args['path'], ctx)}"},
                cancellation_safe=True,
            ),
            self._registration(
                name="grep",
                purpose="Search repository text files with a regular expression.",
                requirements=(
                    "pattern is required and must be a non-empty regex string.",
                    "path is optional and may be a repository-relative file or directory.",
                    "Use this to locate symbols, error text, tests, or repeated snippets.",
                ),
                parameters=object_schema(
                    {
                        "pattern": {"type": "string", "description": "Regular expression."},
                        "path": {"type": "string", "description": "Repository-relative file or directory path."},
                    },
                    required=["pattern"],
                ),
                risk=ToolRisk.READ,
                handler=lambda args, ctx: self.owner._grep(args, ctx),
                resource_resolver=lambda args, ctx: {f"path:{self.owner._resource_path(args.get('path', '.'), ctx)}"},
                cancellation_safe=True,
            ),
            self._registration(
                name="retrieve_context",
                purpose="Retrieve relevant code snippets using lexical repository search.",
                requirements=(
                    "query is required and must be a non-empty search string.",
                    "top_k is optional and must be a positive integer.",
                    "This only retrieves context; it does not edit files or call an LLM.",
                ),
                parameters=object_schema(
                    {
                        "query": {"type": "string", "description": "Search query."},
                        "top_k": {"type": "integer", "minimum": 1, "description": "Maximum snippets to return."},
                    },
                    required=["query"],
                ),
                risk=ToolRisk.READ,
                handler=lambda args, ctx: self.owner._retrieve_context(args, ctx),
                cancellation_safe=True,
            ),
            self._registration(
                name="replace_in_file",
                purpose="Make one exact text replacement in a repository file.",
                requirements=(
                    "path is required and must be a repository-relative file path.",
                    "old and new are required strings; do not use fenced markdown.",
                    "old must be copied from the file exactly and must match exactly one occurrence.",
                ),
                parameters=object_schema(
                    {
                        "path": {"type": "string", "description": "Repository-relative file path."},
                        "old": {"type": "string", "description": "Exact existing text."},
                        "new": {"type": "string", "description": "Replacement text."},
                    },
                    required=["path", "old", "new"],
                ),
                risk=ToolRisk.WRITE,
                handler=lambda args, ctx: self.owner._replace_in_file(args, ctx),
                resource_resolver=lambda args, ctx: {f"file:{self.owner._resource_path(args['path'], ctx, write=True)}"},
            ),
            self._registration(
                name="write_file",
                purpose="Overwrite or create one repository file with complete content.",
                requirements=(
                    "path is required and must be a repository-relative file path.",
                    "content is required and must be the full file content as one JSON string.",
                    "content must be a valid JSON string with escaped quotes and newlines.",
                ),
                parameters=object_schema(
                    {
                        "path": {"type": "string", "description": "Repository-relative file path."},
                        "content": {"type": "string", "description": "Complete file content."},
                    },
                    required=["path", "content"],
                ),
                risk=ToolRisk.WRITE,
                handler=lambda args, ctx: self.owner._write_file(args, ctx),
                resource_resolver=lambda args, ctx: {f"file:{self.owner._resource_path(args['path'], ctx, write=True)}"},
            ),
            self._registration(
                name="run_tests",
                purpose="Run an allowlisted test command in the repository.",
                requirements=(
                    "command is optional and defaults to 'pytest -q'.",
                    "Allowed commands include pytest, python -m pytest, python -m unittest, npm test, npm run test, pnpm test, and yarn test.",
                    "Do not include shell control syntax such as pipes, redirects, semicolons, or command substitution.",
                ),
                parameters=object_schema(
                    {"command": {"type": "string", "description": "Allowlisted test command."}}
                ),
                risk=ToolRisk.EXECUTE,
                handler=lambda args, ctx: self.owner._run_tests(args, ctx),
            ),
            self._registration(
                name="git_diff",
                purpose="Show the current repository git diff.",
                requirements=(
                    "arguments must be an empty JSON object.",
                    "Use this after edits to inspect the exact patch before finishing.",
                    "Do not include command, path, or content fields.",
                ),
                parameters=object_schema(),
                risk=ToolRisk.READ,
                handler=lambda args, ctx: self.owner._git_diff(args, ctx),
                cancellation_safe=True,
            ),
            self._registration(
                name="finish",
                purpose="Finish the task and return the final answer.",
                requirements=(
                    "summary is required and should state what changed and what was verified.",
                    "Only call this when the task is complete or when you must report a blocker.",
                    "Mention failing or unrun tests in summary when relevant.",
                ),
                parameters=object_schema(
                    {"summary": {"type": "string", "description": "Final summary."}},
                    required=["summary"],
                ),
                risk=ToolRisk.READ,
                handler=lambda args, ctx: self.owner._finish(args, ctx),
            ),
        ]

    def _registration(
        self,
        *,
        name: str,
        purpose: str,
        requirements: tuple[str, ...],
        parameters: dict[str, Any],
        risk: ToolRisk,
        handler: Any,
        resource_resolver: Any | None = None,
        parallel_side_effect_safe: bool = False,
        cancellation_safe: bool = False,
    ) -> ToolRegistration:
        return ToolRegistration(
            spec=ToolSpec(
                name=name,
                description=_native_tool_description(purpose=purpose, requirements=requirements),
                parameters=parameters,
                risk=risk,
                source=self.name,
            ),
            handler=handler,
            resource_resolver=resource_resolver,
            parallel_side_effect_safe=parallel_side_effect_safe,
            cancellation_safe=cancellation_safe,
        )


def _native_tool_description(*, purpose: str, requirements: tuple[str, ...]) -> str:
    requirements_text = " ".join(requirements)
    return f"{purpose} {requirements_text}".strip()
