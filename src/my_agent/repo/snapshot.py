from __future__ import annotations

from dataclasses import dataclass

from my_agent.context import estimate_tokens


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
