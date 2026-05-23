from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


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
        return (
            "# Repository tree\n"
            f"{self.tree}\n\n"
            "# Symbol index\n"
            f"{self.symbols}\n\n"
            "# Retrieval notes\n"
            f"{self.retrieval_notes}\n\n"
            "# Important file previews\n"
            f"{self.file_summaries}\n"
        )


class RepoIndexer:
    def __init__(self, repo_path: str | Path, max_files: int = 80, preview_chars: int = 1200):
        self.repo_path = Path(repo_path).resolve()
        self.max_files = max_files
        self.preview_chars = preview_chars
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {self.repo_path}")

    def snapshot(self, query: str | None = None, top_k: int = 8) -> RepoSnapshot:
        files = self._collect_files()
        return RepoSnapshot(
            tree=self._build_tree(files),
            file_summaries=self._build_summaries(files[: self.max_files]),
            project_rules=self._read_project_rules(),
            symbols=self._build_symbol_index(files),
            retrieval_notes=self.retrieve(query, top_k=top_k) if query is not None else "No task-specific retrieval query provided.",
        )

    def retrieve(self, query: str | None, top_k: int = 8, chars_per_file: int = 1400) -> str:
        tokens = _tokenize(query or "")
        if not tokens:
            return "No retrieval query terms available."

        scored: list[tuple[float, Path, str]] = []
        for path in self._collect_files():
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
            dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
            root_path = Path(root)
            for filename in sorted(filenames):
                path = root_path / filename
                rel = path.relative_to(self.repo_path)
                if any(part in IGNORED_DIRS for part in rel.parts):
                    continue
                if path.is_file():
                    files.append(path)
        return sorted(files, key=lambda item: item.relative_to(self.repo_path).as_posix())

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
        for filename in ("AGENT.md", "CLAUDE.md"):
            path = self.repo_path / filename
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
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
        if body_count:
            score += 1.0 + math.log1p(body_count)
        if path_count:
            score += 3.0 + math.log1p(path_count)
    return score


def _best_excerpt(tokens: list[str], text: str, limit: int) -> str:
    lowered = text.lower()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    if not positions:
        return text[:limit].strip()
    center = min(positions)
    start = max(0, center - limit // 3)
    return text[start : start + limit].strip()
