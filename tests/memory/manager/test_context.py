from __future__ import annotations

# ruff: noqa: F403, F405 - shared test support exports the frozen fixtures

from tests.memory.manager.support import *

class MemoryManagerBuildContextTests(unittest.TestCase):
    def test_build_context_for_query_returns_token_bounded_injection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(manager, "用户偏好：回答中文，先给结论", tier="tip")

            ctx = manager.build_context_for_query("用户偏好 回答中文")

            self.assertTrue(ctx.injected_text.startswith("Relevant selected experience:"))
            self.assertGreater(ctx.estimated_tokens, 0)
            self.assertLessEqual(ctx.estimated_tokens, manager.config.memory_context_tokens)

    def test_build_context_for_query_injects_nothing_when_no_relevant_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(manager, "项目使用 FastAPI", tier="tip")

            ctx = manager.build_context_for_query("completely unrelated query xyz")

            self.assertEqual(ctx.injected_text, "")
            self.assertEqual(ctx.estimated_tokens, 0)

    def test_build_context_for_query_never_exceeds_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_context_tokens=1,
                    memory_evolver_mode="retrieve_select",
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(manager, "用户偏好：回答中文", tier="tip")

            ctx = manager.build_context_for_query("用户偏���")

            self.assertEqual(ctx.injected_text, "")
            self.assertEqual(ctx.estimated_tokens, 0)
class MemoryManagerEvolverContextTests(unittest.TestCase):
    def test_evolver_off_injects_no_long_term_context_and_traces_empty_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            save_typed_experience(manager, "ordinary typed tip about config", tier="tip")

            context = manager.build_context_for_query("ordinary typed config")

            self.assertEqual(context.injected_text, "")
            self.assertEqual(context.hits, [])
            self.assertIn("memory.retrieved", [event for event, _ in traces])
            self.assertNotIn("memory.evolver_selected", [event for event, _ in traces])

    def test_evolver_enabled_injects_only_selected_experience(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_selected_max_items=1,
                    memory_evolver_tier_caps={"trajectory": 1, "tip": 1, "skill": 1, "tool": 1},
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(
                manager,
                "pytest weak tip",
                tier="tip",
                source_task="task-weak",
            )
            save_typed_experience(
                manager,
                "pytest boosted skill",
                tier="skill",
                source_task="task-strong",
            )

            context = manager.build_context_for_query("pytest")

            self.assertTrue(context.injected_text.startswith("Relevant selected experience:"))
            self.assertIn("pytest boosted skill", context.injected_text)
            self.assertNotIn("pytest weak tip", context.injected_text)
            self.assertEqual([hit.entry.content for hit in context.hits], ["pytest boosted skill"])
            self.assertIsNotNone(manager.last_evolver_selection)

    def test_evolver_enabled_ignores_legacy_include_short_term_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_selected_max_items=1,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_user_message("short-term pytest note should not be injected")
            save_typed_experience(
                manager,
                "selected pytest tip",
                tier="tip",
                source_task="task-selected",
            )

            context = manager.build_context_for_query("pytest", include_short_term=True)

            self.assertTrue(context.injected_text.startswith("Relevant selected experience:"))
            self.assertIn("selected pytest tip", context.injected_text)
            self.assertNotIn("short-term pytest note", context.injected_text)
            self.assertEqual([hit.entry.content for hit in context.hits], ["selected pytest tip"])

    def test_evolver_context_traces_candidates_and_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_selected_max_items=1,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            save_typed_experience(manager, "pytest tip", tier="tip", source_task="task-tip")
            save_typed_experience(manager, "pytest skill", tier="skill", source_task="task-skill")

            context = manager.build_context_for_query("pytest")

            candidates = [payload for event, payload in traces if event == "memory.evolver_candidates"][-1]
            selected = [payload for event, payload in traces if event == "memory.evolver_selected"][-1]
            retrieved = [payload for event, payload in traces if event == "memory.retrieved"][-1]
            self.assertEqual(candidates["candidate_count"], 2)
            self.assertEqual(candidates["selection_policy"], "rule_tier_weighted_v1")
            self.assertEqual(candidates["memory_project_key"], manager.project_key)
            self.assertIn("repository_revision", candidates)
            self.assertGreaterEqual(candidates["indexed_count"], 2)
            self.assertGreaterEqual(candidates["posting_candidate_count"], 2)
            self.assertEqual(candidates["retrieval_fallback"], "")
            self.assertEqual(selected["selected_count"], 1)
            self.assertEqual(selected["estimated_tokens"], context.estimated_tokens)
            self.assertEqual(retrieved["hits"], len(context.hits))
            self.assertEqual(retrieved["mode"], "retrieve_select")
            summary = candidates["candidate_summaries"][0]
            self.assertIn("score", summary)
            self.assertIn("tokens", summary)
            self.assertIn("source_task", summary)

    def test_evolver_candidate_trace_uses_selector_visible_candidates_for_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_min_score=2.0,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            save_typed_experience(manager, "pytest tip below selector threshold", tier="tip")

            context = manager.build_context_for_query("pytest")

            candidates = [payload for event, payload in traces if event == "memory.evolver_candidates"][-1]
            self.assertEqual(context.injected_text, "")
            self.assertEqual(candidates["candidate_count"], 0)
            self.assertEqual(candidates["candidate_ids"], [])
            self.assertEqual(candidates["candidate_summaries"], [])
            self.assertEqual(candidates["tiers"], {})
            self.assertEqual(candidates["retrieved_candidate_count"], 1)
            self.assertEqual(candidates["retrieved_tiers"], {"tip": 1})

    def test_evolver_selector_failure_returns_empty_context_without_legacy_fallback(self) -> None:
        class RaisingSelector:
            def select(self, **_: object) -> object:
                raise RuntimeError("selector unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            save_typed_experience(manager, "pytest tip selected before selector fails", tier="tip")
            manager.evolver_selector = RaisingSelector()  # type: ignore[assignment]

            context = manager.build_context_for_query("pytest")

            failed = [payload for event, payload in traces if event == "memory.evolver_selection_failed"][-1]
            selected = [payload for event, payload in traces if event == "memory.evolver_selected"][-1]
            retrieved = [payload for event, payload in traces if event == "memory.retrieved"][-1]
            self.assertEqual(context.injected_text, "")
            self.assertEqual(context.hits, [])
            self.assertEqual(failed["fallback"], "empty_context")
            self.assertTrue(selected["fallback"])
            self.assertEqual(retrieved["hits"], 0)

    def test_evolver_candidate_retrieval_is_top_k_per_tier_not_global_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(manager, "pytest tip one", tier="tip")
            save_typed_experience(manager, "pytest tip two", tier="tip")
            save_typed_experience(manager, "pytest skill one", tier="skill")
            save_typed_experience(manager, "pytest skill two", tier="skill")

            candidates = manager.retrieve_evolver_candidates("pytest", top_k_per_tier=1)

            tiers = [hit.entry.tier.value for hit in candidates]
            self.assertEqual(tiers.count("tip"), 1)
            self.assertEqual(tiers.count("skill"), 1)

    def test_evolver_context_respects_project_and_global_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_a.mkdir()
            repo_b.mkdir()
            memory_dir = Path(tmp) / "memory"
            config_a = _config(memory_dir, memory_evolver_mode="retrieve_select", memory_project_key="stream:a")
            config_b = _config(memory_dir, memory_evolver_mode="retrieve_select", memory_project_key="stream:b")
            manager_a = MemoryManager.from_config(config=config_a, llm=FakeLLM(), repo_path=repo_a)
            save_typed_experience(manager_a, "pytest project a tip", tier="tip")
            save_typed_experience(
                manager_a, "pytest global skill", tier="skill", scope=MemoryScope.GLOBAL
            )
            manager_b = MemoryManager.from_config(config=config_b, llm=FakeLLM(), repo_path=repo_b)

            context_b = manager_b.build_context_for_query("pytest")

            self.assertNotIn("pytest project a tip", context_b.injected_text)
            self.assertIn("pytest global skill", context_b.injected_text)

    def test_evolver_context_can_be_disabled_by_min_experience_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_min_experience_entries=2,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(manager, "pytest one tip", tier="tip")

            context = manager.build_context_for_query("pytest")

            self.assertEqual(context.injected_text, "")
            self.assertTrue(manager.last_evolver_selection.metadata["insufficient_experience_entries"])

    def test_noop_evolver_context_returns_empty_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            config = _config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select")
            manager = NoopMemoryManager(config=config, repo_path=repo)

            context = manager.build_evolver_context_for_query("pytest")

            self.assertEqual(context.injected_text, "")
            self.assertEqual(context.hits, [])
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())
