from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.indexer import RepoIndexer, SymbolRecord


SAMPLE_REPO = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"


class RepoIndexerTests(unittest.TestCase):
    def test_snapshot_contains_tree_symbols_rules_and_previews(self) -> None:
        snapshot = RepoIndexer(SAMPLE_REPO).snapshot(query="subtract")
        context = snapshot.as_context()

        self.assertIn("# Repository tree", context)
        self.assertIn("# Symbol index", context)
        self.assertIn("# Retrieval notes", context)
        self.assertIn("# Important file previews", context)
        self.assertIn("calculator.py", snapshot.tree)
        self.assertIn("tests/", snapshot.tree)
        self.assertIn("test_calculator.py", snapshot.tree)
        self.assertIn("calculator.py", snapshot.file_summaries)
        self.assertIn("function subtract", snapshot.symbols)
        self.assertIn("Make the smallest safe change", snapshot.project_rules)
        self.assertIn("subtract", snapshot.retrieval_notes)

    def test_retrieve_returns_related_file_snippet(self) -> None:
        result = RepoIndexer(SAMPLE_REPO).retrieve("subtract", top_k=2)

        self.assertIn("calculator.py", result)
        self.assertIn("subtract", result)
        self.assertIn("score=", result)

    def test_top_one_retrieval_prefers_implementation_file(self) -> None:
        result = RepoIndexer(SAMPLE_REPO).retrieve("subtract", top_k=1)

        self.assertIn("## calculator.py", result)
        self.assertNotIn("## tests/test_calculator.py", result)

    def test_ignored_directories_are_not_indexed_or_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "visible.py").write_text("def visible():\n    return 'ok'\n", encoding="utf-8")
            for dirname in (".git", ".venv", "node_modules", "build", "__pycache__"):
                path = repo / dirname
                path.mkdir()
                (path / "hidden.py").write_text("def hidden():\n    return 'needle'\n", encoding="utf-8")

            snapshot = RepoIndexer(repo).snapshot(query="needle")

            self.assertIn("visible.py", snapshot.tree)
            self.assertNotIn("hidden.py", snapshot.tree)
            self.assertNotIn("hidden.py", snapshot.file_summaries)
            self.assertIn("No relevant files found", snapshot.retrieval_notes)

    def test_symlink_files_are_not_indexed_or_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            repo = Path(repo_tmp)
            outside = Path(outside_tmp) / "outside.py"
            outside.write_text("def leaked():\n    return 'outside_secret'\n", encoding="utf-8")
            try:
                (repo / "link.py").symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink not supported: {exc}")

            snapshot = RepoIndexer(repo).snapshot(query="outside_secret")

            self.assertNotIn("link.py", snapshot.tree)
            self.assertNotIn("outside_secret", snapshot.file_summaries)
            self.assertIn("No relevant files found", snapshot.retrieval_notes)

    def test_project_rules_do_not_follow_symlink_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            repo = Path(repo_tmp)
            outside = Path(outside_tmp) / "AGENT.md"
            outside.write_text("outside_secret_rules", encoding="utf-8")
            try:
                (repo / "AGENT.md").symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink not supported: {exc}")

            snapshot = RepoIndexer(repo).snapshot()

            self.assertNotIn("outside_secret_rules", snapshot.project_rules)
            self.assertIn("No project-specific AGENT.md or CLAUDE.md found", snapshot.project_rules)

    def test_unreadable_project_rules_fall_back_to_next_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "AGENT.md").write_text("unreadable agent rules", encoding="utf-8")
            (repo / "CLAUDE.md").write_text("readable claude rules", encoding="utf-8")
            original_read_text = Path.read_text

            def fake_read_text(path: Path, *args: object, **kwargs: object) -> str:
                if path.name == "AGENT.md":
                    raise PermissionError("denied")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", fake_read_text):
                snapshot = RepoIndexer(repo).snapshot()

            self.assertEqual(snapshot.project_rules, "readable claude rules")

    def test_unreadable_project_rules_return_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "AGENT.md").write_text("unreadable agent rules", encoding="utf-8")

            with mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                snapshot = RepoIndexer(repo).snapshot()

            self.assertIn("Project rules file exists but could not be read", snapshot.project_rules)

    def test_empty_and_no_hit_queries_return_clear_messages(self) -> None:
        indexer = RepoIndexer(SAMPLE_REPO)

        self.assertIn("No retrieval query terms available", indexer.retrieve(""))
        self.assertIn("No relevant files found", indexer.retrieve("zzzz_no_such_symbol"))

    def test_top_k_must_be_positive(self) -> None:
        indexer = RepoIndexer(SAMPLE_REPO)

        with self.assertRaisesRegex(ValueError, "top_k must be >= 1"):
            indexer.retrieve("subtract", top_k=0)
        with self.assertRaisesRegex(ValueError, "top_k must be >= 1"):
            indexer.retrieve("subtract", top_k=-1)
        with self.assertRaisesRegex(ValueError, "top_k must be >= 1"):
            indexer.snapshot(query="subtract", top_k=0)

    def test_syntax_error_python_file_does_not_break_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "ok.py").write_text("async def good():\n    return 1\n", encoding="utf-8")
            (repo / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

            snapshot = RepoIndexer(repo).snapshot(query="good")

            self.assertIn("ok.py", snapshot.tree)
            self.assertIn("broken.py", snapshot.tree)
            self.assertIn("function good", snapshot.symbols)

    def test_symbol_record_render_format(self) -> None:
        record = SymbolRecord(kind="function", name="subtract", path="calculator.py", line=5)

        self.assertEqual(record.render(), "calculator.py:5: function subtract")


if __name__ == "__main__":
    unittest.main()
