from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from my_agent.cancellation import CancellationToken
from my_agent.context import estimate_tokens


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".css",
    ".html",
}


# 函数或类
@dataclass(frozen=True)
class SymbolRecord:
    kind: str
    name: str
    path: str
    line: int

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} {self.name}"


@dataclass(frozen=True)
class RepoSnapshot:
    tree: str
    file_summaries: str
    project_rules: str
    symbols: str
    retrieval_notes: str

    def as_context(self) -> str:
        return self.render_context().text

    def render_context(self, *, max_tokens: int | None = None) -> "RepoContextRender":
        sections = _display_sections(self)
        full_text = _join_sections(sections)
        full_tokens = estimate_tokens(full_text)
        if max_tokens is None or max_tokens < 1 or full_tokens <= max_tokens:
            return RepoContextRender(
                text=full_text,
                budget_tokens=max_tokens or 0,
                estimated_tokens=full_tokens,
                truncated=False,
                section_tokens={name: estimate_tokens(_section_text(title, body)) for name, title, body in sections},
                truncated_sections=(),
            )

        remaining = max_tokens
        sections = _budget_priority_sections(self)
        rendered_sections: list[tuple[str, str, str]] = []
        section_tokens: dict[str, int] = {}
        truncated_sections: list[str] = []
        for idx, (name, title, body) in enumerate(sections):
            section_budget = max(0, remaining - _separator_tokens(rendered_sections))
            section = _section_text(title, body)
            if section_budget <= 0:
                truncated_sections.extend(section_name for section_name, _, _ in sections[idx:])
                break
            elif estimate_tokens(section) <= section_budget:
                rendered_body = body
            else:
                rendered_body = _truncate_to_token_budget(
                    body,
                    section_budget - estimate_tokens(_section_text(title, "")),
                )
                truncated_sections.append(name)
                if estimate_tokens(_section_text(title, rendered_body)) > section_budget:
                    truncated_sections.extend(section_name for section_name, _, _ in sections[idx + 1 :])
                    break
            rendered_sections.append((name, title, rendered_body))
            rendered_text = _join_sections(rendered_sections)
            remaining = max(0, max_tokens - estimate_tokens(rendered_text))
            section_tokens[name] = estimate_tokens(_section_text(title, rendered_body))

        rendered_text = _join_sections(rendered_sections)
        return RepoContextRender(
            text=rendered_text,
            budget_tokens=max_tokens,
            estimated_tokens=estimate_tokens(rendered_text),
            truncated=True,
            section_tokens=section_tokens,
            truncated_sections=tuple(truncated_sections),
        )


@dataclass(frozen=True)
class RepoContextRender:
    text: str
    budget_tokens: int
    estimated_tokens: int
    truncated: bool
    section_tokens: dict[str, int]
    truncated_sections: tuple[str, ...]

    def to_trace_payload(self) -> dict[str, object]:
        return {
            "repo_context_budget_tokens": self.budget_tokens,
            "repo_context_estimated_tokens": self.estimated_tokens,
            "repo_context_truncated": self.truncated,
            "repo_context_section_tokens": dict(self.section_tokens),
            "repo_context_truncated_sections": list(self.truncated_sections),
        }


def _display_sections(snapshot: RepoSnapshot) -> list[tuple[str, str, str]]:
    return [
        ("tree", "# Repository tree", snapshot.tree),
        ("symbols", "# Symbol index", snapshot.symbols),
        ("retrieval_notes", "# Retrieval notes", snapshot.retrieval_notes),
        ("file_summaries", "# Important file previews", snapshot.file_summaries),
    ]


def _budget_priority_sections(snapshot: RepoSnapshot) -> list[tuple[str, str, str]]:
    return [
        ("retrieval_notes", "# Retrieval notes", snapshot.retrieval_notes),
        ("tree", "# Repository tree", snapshot.tree),
        ("symbols", "# Symbol index", snapshot.symbols),
        ("file_summaries", "# Important file previews", snapshot.file_summaries),
    ]


def _join_sections(sections: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(_section_text(title, body) for _, title, body in sections)


def _section_text(title: str, body: str) -> str:
    return f"{title}\n{body}"


def _separator_tokens(rendered_sections: list[tuple[str, str, str]]) -> int:
    return 1 if rendered_sections else 0


def _truncate_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    marker = "\n[Truncated: repo context budget exhausted.]"
    if estimate_tokens(marker) > token_budget:
        return ""
    max_chars = max(0, token_budget * 4 - len(marker))
    if len(text) <= max_chars and estimate_tokens(text) <= token_budget:
        return text
    truncated = text[:max_chars].rstrip() + marker
    while len(truncated) > len(marker) and estimate_tokens(truncated) > token_budget:
        max_chars = max(0, max_chars - 128)
        truncated = text[:max_chars].rstrip() + marker
    return truncated


class RepoIndexer:
    def __init__(
        self,
        repo_path: str | Path,
        max_files: int = 80,
        preview_chars: int = 1200,
        skip_predicate: Callable[[Path], bool] | None = None,
        cancellation_token: CancellationToken | None = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.max_files = max_files
        self.preview_chars = preview_chars
        self.skip_predicate = skip_predicate
        self.cancellation_token = cancellation_token
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {self.repo_path}")

    def snapshot(self, query: str | None = None, top_k: int = 8) -> RepoSnapshot:
        top_k = _validate_top_k(top_k)
        files = self._collect_files()
        return RepoSnapshot(
            tree=self._build_tree(files),
            file_summaries=self._build_summaries(files[: self.max_files]),
            project_rules=self._read_project_rules(),
            symbols=self._build_symbol_index(files),
            retrieval_notes=self.retrieve(query, top_k=top_k) if query is not None else "No task-specific retrieval query provided.",
        )

    def retrieve(self, query: str | None, top_k: int = 8, chars_per_file: int = 1400) -> str:
        top_k = _validate_top_k(top_k)
        tokens = _tokenize(query or "")
        if not tokens:
            return "No retrieval query terms available."

        scored: list[tuple[float, Path, str]] = []
        for path in self._collect_files():
            self._raise_if_cancelled()
            if not _is_text_file(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(self.repo_path).as_posix()
            score = _score_text(tokens, rel, text)
            if score > 0:
                scored.append((score, path, text))

        if not scored:
            return "No relevant files found by lexical retrieval. Use list_files or grep next."

        chunks: list[str] = []
        for score, path, text in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]:
            rel = path.relative_to(self.repo_path).as_posix()
            excerpt = _best_excerpt(tokens, text, chars_per_file)
            chunks.append(f"## {rel} score={score:.2f}\n```\n{excerpt}\n```")
        return "\n\n".join(chunks)

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for root, dirnames, filenames in os.walk(self.repo_path):
            self._raise_if_cancelled()
            root_path = Path(root)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in IGNORED_DIRS and not (root_path / name).is_symlink()
            )
            for filename in sorted(filenames):
                self._raise_if_cancelled()
                path = root_path / filename
                rel = path.relative_to(self.repo_path)
                if any(part in IGNORED_DIRS for part in rel.parts):
                    continue
                if self._is_safe_repo_file(path) and not self._should_skip(path):
                    files.append(path)
        return sorted(files, key=lambda item: item.relative_to(self.repo_path).as_posix())

    def _raise_if_cancelled(self) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.raise_if_cancelled()

    def _is_safe_repo_file(self, path: Path) -> bool:
        if path.is_symlink():
            return False
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return False
        return resolved.is_relative_to(self.repo_path) and path.is_file()

    def _should_skip(self, path: Path) -> bool:
        if self.skip_predicate is None:
            return False
        return self.skip_predicate(path)

    def _build_tree(self, files: list[Path]) -> str:
        lines: list[str] = []
        rendered_dirs: set[tuple[str, ...]] = set()
        for path in files[: self.max_files]:
            rel = path.relative_to(self.repo_path)
            for depth, part in enumerate(rel.parts[:-1]):
                dir_key = rel.parts[: depth + 1]
                if dir_key in rendered_dirs:
                    continue
                rendered_dirs.add(dir_key)
                lines.append(f"{'  ' * depth}- {part}/")
            depth = len(rel.parts) - 1
            lines.append(f"{'  ' * depth}- {rel.name}")
        if len(files) > self.max_files:
            lines.append(f"... {len(files) - self.max_files} more files omitted")
        return "\n".join(lines) or "(empty repository)"

    def _build_summaries(self, files: list[Path]) -> str:
        chunks: list[str] = []
        for path in files:
            if not _is_text_file(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            preview = text[: self.preview_chars].strip()
            if not preview:
                continue
            rel = path.relative_to(self.repo_path).as_posix()
            chunks.append(f"## {rel}\n```\n{preview}\n```")
        return "\n\n".join(chunks) or "No text file previews available."

    def _build_symbol_index(self, files: list[Path]) -> str:
        records: list[SymbolRecord] = []
        for path in files:
            if path.suffix != ".py":
                continue
            rel = path.relative_to(self.repo_path).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    records.append(SymbolRecord("class", node.name, rel, node.lineno))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    records.append(SymbolRecord("function", node.name, rel, node.lineno))
        if not records:
            return "No Python symbols found."
        return "\n".join(record.render() for record in records[:120])

    def _read_project_rules(self) -> str:
        unreadable_rule_found = False
        for filename in ("AGENT.md", "CLAUDE.md"):
            path = self.repo_path / filename
            if self._is_safe_repo_file(path):
                try:
                    return path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    unreadable_rule_found = True
                    continue
        if unreadable_rule_found:
            return "Project rules file exists but could not be read. Follow safe minimal-change defaults."
        return "No project-specific AGENT.md or CLAUDE.md found. Follow safe minimal-change defaults."


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]+", text)]


def _score_text(tokens: list[str], path: str, text: str) -> float:
    lowered_path = path.lower()
    lowered_text = text.lower()
    score = 0.0
    for token in tokens:
        body_count = lowered_text.count(token)
        path_count = lowered_path.count(token)
        definition_count = len(re.findall(rf"\b(?:async\s+def|def|class)\s+{re.escape(token)}\b", lowered_text))
        if body_count:
            score += 1.0 + math.log1p(body_count)
        if path_count:
            score += 3.0 + math.log1p(path_count)
        if definition_count:
            score += 4.0 + math.log1p(definition_count)
    return score


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be >= 1.")
    return top_k


def _best_excerpt(tokens: list[str], text: str, limit: int) -> str:
    lowered = text.lower()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    if not positions:
        return text[:limit].strip()
    center = min(positions)
    start = max(0, center - limit // 3)
    return text[start : start + limit].strip()
