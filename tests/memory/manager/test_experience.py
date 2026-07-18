from __future__ import annotations

# ruff: noqa: F403, F405 - shared test support exports the frozen fixtures

from tests.memory.manager.support import *

class MemoryManagerExperienceVisibilityTests(unittest.TestCase):
    def test_save_experience_persists_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            entry, created = save_typed_experience(
                manager,
                "用户偏好：回答中文",
                tier="tip",
                scope=MemoryScope.PROJECT,
            )
            self.assertTrue(created)

            same, created2 = save_typed_experience(
                manager,
                "用户偏好：回答中文",
                tier="tip",
                scope=MemoryScope.PROJECT,
            )
            self.assertFalse(created2)
            self.assertEqual(same.id, entry.id)

    def test_save_experience_survives_new_manager_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            config = _config(Path(tmp) / "memory")
            manager = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            save_typed_experience(manager, "durable tip about config", tier="tip", scope=MemoryScope.PROJECT)

            reopened = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            contents = {entry.content for entry in reopened.experience_store.all(project_key=reopened.project_key)}
            self.assertIn("durable tip about config", contents)

    def test_save_experience_global_scope_visible_to_other_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_a.mkdir()
            repo_b.mkdir()
            config = _config(Path(tmp) / "memory")
            manager_a = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo_a)
            save_typed_experience(manager_a, "global tip about config", tier="tip", scope=MemoryScope.GLOBAL)

            manager_b = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo_b)
            contents = {entry.content for entry in manager_b.experience_store.all(project_key=manager_b.project_key)}
            self.assertIn("global tip about config", contents)

    def test_memory_project_key_override_controls_project_scope_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_c = Path(tmp) / "repo_c"
            repo_a.mkdir()
            repo_b.mkdir()
            repo_c.mkdir()
            memory_dir = Path(tmp) / "memory"
            config_x = _config(memory_dir, memory_project_key="stream:x")
            config_y = _config(memory_dir, memory_project_key="stream:y")

            manager_a = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_a)
            save_typed_experience(
                manager_a,
                "stream marker VALUE equals 1",
                tier="skill",
                scope=MemoryScope.PROJECT,
            )

            manager_b = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_b)
            manager_c = MemoryManager.from_config(config=config_y, llm=FakeLLM(), repo_path=repo_c)

            contents_b = {entry.content for entry in manager_b.experience_store.all(project_key=manager_b.project_key)}
            contents_c = {entry.content for entry in manager_c.experience_store.all(project_key=manager_c.project_key)}

        self.assertEqual(manager_a.project_key, "stream:x")
        self.assertEqual(manager_b.project_key, "stream:x")
        self.assertEqual(manager_c.project_key, "stream:y")
        self.assertIn("stream marker VALUE equals 1", contents_b)
        self.assertNotIn("stream marker VALUE equals 1", contents_c)
class MemoryManagerSaveExperienceTests(unittest.TestCase):
    def test_managers_do_not_expose_legacy_fact_or_retrieval_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            config = _config(Path(tmp) / "memory")
            manager = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            noop = NoopMemoryManager(config=config, repo_path=repo)

            for candidate in (manager, noop):
                for legacy_name in ("long_term", "retriever", "retrieve_hits", "save_fact", "extract_facts"):
                    self.assertFalse(hasattr(candidate, legacy_name), legacy_name)

    def test_runtime_manager_fails_closed_on_legacy_long_term_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            (memory_dir / "long_term_memory.jsonl").write_text("{not valid json}\n", encoding="utf-8")

            with self.assertRaisesRegex(MemoryStoreLoadError, "explicit four-tier memory reset"):
                MemoryManager.from_config(config=_config(memory_dir), llm=FakeLLM(), repo_path=repo)
            self.assertTrue((memory_dir / "long_term_memory.jsonl").exists())
            self.assertFalse((memory_dir / "experience_memory.jsonl").exists())

    def test_save_experience_persists_typed_payload_and_traces(self) -> None:
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

            entry, created = save_typed_experience(
                manager,
                "Always run compileall after package migration.",
                tier=ExperienceTier.SKILL,
                payload=SkillPayload(
                    category="debugging",
                    technique="import smoke",
                    preconditions=(),
                    steps=("compileall", "import smoke"),
                ),
                source_task="task-1",
                created_by=ExperienceCreatedBy.WRITER,
                run_id="run-1",
                stream_id="stream-a",
                writer_confidence=0.8,
            )

            self.assertTrue(created)
            self.assertIsInstance(entry, ExperienceMemory)
            self.assertEqual(entry.tier, ExperienceTier.SKILL)
            self.assertEqual(entry.scope, MemoryScope.PROJECT)
            self.assertEqual(entry.project_key, str(repo.resolve()))
            self.assertEqual(entry.run_id, "run-1")
            self.assertEqual(entry.stream_id, "stream-a")
            self.assertEqual(entry.source_task, "task-1")
            self.assertEqual(entry.created_by, ExperienceCreatedBy.WRITER)
            self.assertEqual(entry.payload.technique, "import smoke")
            self.assertEqual(entry.writer_confidence, 0.8)

            events = [(event, payload) for event, payload in traces if event == "memory.evolver_saved"]
            self.assertEqual(len(events), 1)
            payload = events[0][1]
            self.assertEqual(payload["id"], entry.id)
            self.assertEqual(payload["created"], True)
            self.assertEqual(payload["tier"], "skill")
            self.assertEqual(payload["scope"], "project")
            self.assertEqual(payload["tokens"], entry.token_count)
            self.assertEqual(payload["source_task"], "task-1")
            self.assertEqual(payload["created_by"], "writer")

    def test_save_experience_dedup_and_project_visibility_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_c = Path(tmp) / "repo_c"
            repo_a.mkdir()
            repo_b.mkdir()
            repo_c.mkdir()
            memory_dir = Path(tmp) / "memory"
            config_x = _config(memory_dir, memory_project_key="stream:x")
            config_y = _config(memory_dir, memory_project_key="stream:y")
            manager_a = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_a)

            entry, created = save_typed_experience(manager_a, "same project tip", tier="tip")
            duplicate, created_duplicate = save_typed_experience(
                manager_a, "same project tip", tier="tip"
            )
            global_entry, created_global = save_typed_experience(
                manager_a,
                "global evolver skill",
                tier="skill",
                scope=MemoryScope.GLOBAL,
            )
            manager_b = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_b)
            manager_c = MemoryManager.from_config(config=config_y, llm=FakeLLM(), repo_path=repo_c)

            self.assertTrue(created)
            self.assertFalse(created_duplicate)
            self.assertEqual(duplicate.id, entry.id)
            self.assertTrue(created_global)
            self.assertEqual(global_entry.project_key, "")
            visible_b = {
                item.content for item in manager_b.experience_store.all(project_key=manager_b.project_key)
            }
            visible_c = {
                item.content for item in manager_c.experience_store.all(project_key=manager_c.project_key)
            }
            self.assertIn("same project tip", visible_b)
            self.assertIn("global evolver skill", visible_b)
            self.assertNotIn("same project tip", visible_c)
            self.assertIn("global evolver skill", visible_c)

    def test_save_experience_preserves_tool_and_trajectory_payload_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            config = _config(Path(tmp) / "memory")
            manager = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            save_typed_experience(
                manager,
                "Reusable command template for one pytest file.",
                tier="tool",
                payload=ToolPayload(
                    name="pytest_single_file",
                    language="bash",
                    code="pytest {test_path} -q",
                    input_description="test_path: path to one pytest file",
                    output_description="pytest summary",
                    command="pytest tests/test_parser.py -q",
                    args_schema={"test_path": {"type": "string"}},
                    repo_context="run from repo root",
                ),
            )
            save_typed_experience(
                manager,
                "Parser fix trajectory.",
                tier="trajectory",
                source_task="task-trajectory",
                payload=TrajectoryPayload(
                    task_description="Fix parser test",
                    steps=(ExperienceTrajectoryStep(
                        step_num=1,
                        observation="pytest failed",
                        action="run_tests",
                        action_params={"command": "pytest tests/test_parser.py -q"},
                        result="passed after fix",
                        reward=1.0,
                    ),),
                    outcome="success",
                    total_reward=1.0,
                    key_learnings=("parser strips comments before tokenization",),
                    tags=("parser",),
                ),
            )

            reopened = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            entries = {
                entry.tier.value: entry
                for entry in reopened.experience_store.all(project_key=str(repo.resolve()))
            }

            tool = entries["tool"]
            self.assertEqual(tool.payload.name, "pytest_single_file")
            self.assertEqual(tool.payload.code, "pytest {test_path} -q")
            self.assertEqual(tool.payload.args_schema, {"test_path": {"type": "string"}})
            trajectory = entries["trajectory"]
            self.assertEqual(trajectory.source_task, "task-trajectory")
            self.assertEqual(trajectory.payload.steps[0].action, "run_tests")
            self.assertEqual(trajectory.payload.key_learnings, ("parser strips comments before tokenization",))

    def test_runtime_retrieves_typed_experience_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(
                manager,
                "pytest fixture cleanup issue: clear tmp_path state before rerun",
                tier="tip",
            )

            unrelated_ctx = manager.build_context_for_query("ordinary config")
            experience_ctx = manager.build_context_for_query("pytest fixture cleanup")

            self.assertEqual(unrelated_ctx.injected_text, "")
            self.assertIn("pytest fixture cleanup issue", experience_ctx.injected_text)

    def test_noop_save_experience_returns_entry_without_writing_or_tracing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(Path(tmp) / "memory")
            manager = NoopMemoryManager(
                config=config,
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            entry, created = save_typed_experience(
                manager,
                "No-op tool template",
                tier="tool",
                payload=ToolPayload(name="run_tests", language="bash", code="pytest -q"),
                source_task="task-noop",
                created_by=ExperienceCreatedBy.MANUAL,
                run_id="run-noop",
            )

            self.assertFalse(created)
            self.assertIsInstance(entry, ExperienceMemory)
            self.assertEqual(entry.tier, ExperienceTier.TOOL)
            self.assertEqual(entry.source_task, "task-noop")
            self.assertEqual(entry.payload.name, "run_tests")
            self.assertEqual(entry.run_id, "run-noop")
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())
            self.assertNotIn("memory.evolver_saved", [event for event, _ in traces])

    def test_write_experiences_disabled_does_not_write_or_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(Path(tmp) / "memory", memory_evolver_mode="full")
            manager = MemoryManager.from_config(
                config=config,
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                stop_reason="finish_called",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual(result.saved, ())
            self.assertEqual([event for event, _ in traces if event.startswith("memory.evolver_writer")], [])
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())

    def test_write_experiences_retrieve_select_mode_does_not_run_enabled_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(
                Path(tmp) / "memory",
                memory_evolver_mode="retrieve_select",
                memory_evolver_writer_enabled=True,
            )
            manager = MemoryManager.from_config(
                config=config,
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                stop_reason="finish_called",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual(result.saved, ())
            self.assertEqual(
                [event for event, _ in traces if event.startswith("memory.evolver_writer")],
                [],
            )
            self.assertEqual(manager.experience_store.all(project_key=manager.project_key), [])

    def test_write_experiences_success_saves_writer_skill_and_tool_with_trace_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(
                Path(tmp) / "memory",
                memory_evolver_mode="full",
                memory_evolver_writer_enabled=True,
                memory_project_key="stream:a",
            )
            manager = MemoryManager.from_config(
                config=config,
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace.jsonl",
                stop_reason="finish_called",
                outcome="success",
                outcome_source="runtime",
                source_task="manifest-task-1",
                stream_id="stream-a",
                task_type="manifest",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual({entry.tier.value for entry in result.saved}, {"skill", "tool"})
            entry = result.saved[0]
            self.assertEqual(entry.created_by, ExperienceCreatedBy.WRITER)
            self.assertEqual(entry.writer_confidence, 0.8)
            self.assertEqual(entry.source_task, "manifest-task-1")
            self.assertEqual(entry.stream_id, "stream-a")
            self.assertEqual(entry.project_key, "stream:a")
            self.assertEqual(entry.run_id, "run-1")
            persisted = experience_to_dict(entry)
            for task_field in (
                "outcome", "outcome_source", "task_type", "writer_policy", "writer_reason",
                "source_trace", "candidate_memory_ids", "selected_memory_ids",
            ):
                self.assertNotIn(task_field, persisted)

            events = [event for event, _ in traces]
            self.assertIn("memory.evolver_writer_started", events)
            self.assertIn("memory.evolver_writer_proposed", events)
            self.assertIn("memory.evolver_writer_saved", events)
            saved_payload = [payload for event, payload in traces if event == "memory.evolver_writer_saved"][-1]
            self.assertEqual(saved_payload["saved_count"], 2)
            self.assertEqual(saved_payload["memory_project_key"], "stream:a")
            self.assertEqual(saved_payload["source_task"], "manifest-task-1")
            self.assertEqual(saved_payload["saved_records"][0]["tier"], "skill")

    def test_write_experiences_appends_self_describing_dataset_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "datasets" / "writer.jsonl"
            config = _config(
                Path(tmp) / "memory",
                memory_evolver_mode="full",
                memory_evolver_writer_enabled=True,
                memory_evolver_writer_dataset_path=dataset_path,
                memory_project_key="stream:a",
            )
            manager = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            save_typed_experience(
                manager, "pytest selected skill", tier="skill", source_task="task-skill"
            )
            save_typed_experience(manager, "pytest useful tip", tier="tip", source_task="task-tip")
            manager.build_evolver_context_for_query("pytest")

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace.jsonl",
                stop_reason="finish_called",
                outcome="success",
                outcome_source="runtime",
                source_task="manifest-task-1",
                stream_id="stream-a",
                task_type="manifest",
                memory_mode="shared_stream",
                tool_history=[
                    _tool_record(
                        "run_tests",
                        ok=True,
                        output="hidden_test_output should not be persisted",
                    )
                ],
            )

            rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            record = rows[0]
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["run_id"], "run-1")
            self.assertEqual(record["trace_path"], str(Path(tmp) / "trace.jsonl"))
            self.assertEqual(record["source_task"], "manifest-task-1")
            self.assertEqual(record["task_id"], "manifest-task-1")
            self.assertEqual(record["task_type"], "manifest")
            self.assertEqual(record["stream_id"], "stream-a")
            self.assertEqual(record["memory_project_key"], "stream:a")
            self.assertEqual(record["memory_mode"], "shared_stream")
            self.assertEqual(record["outcome"], "success")
            self.assertEqual(record["selected_memory_ids"], [item.candidate.id for item in manager.last_evolver_selection.selected])
            self.assertIn("skill", record["candidate_memory_ids_by_tier"])
            self.assertIn("skill", record["selected_memory_ids_by_tier"])
            self.assertEqual(record["saved_ids"], [entry.id for entry in result.saved])
            self.assertEqual(record["saved_records"][0]["tier"], "skill")
            self.assertTrue(record["proposals"])
            self.assertIn("payload", record["proposals"][0])
            self.assertIn("reason", record["proposals"][0])
            self.assertEqual(record["steps"][0]["output"], "")
            self.assertTrue(record["steps"][0]["output_redacted"])
            self.assertNotIn("hidden_test_output", json.dumps(record, ensure_ascii=False))

    def test_write_experiences_dataset_append_error_traces_failure_without_losing_saved_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "writer-as-directory"
            dataset_path.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_dataset_path=dataset_path,
                    memory_project_key="project?token=project-token-value",
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                source_task="task?api_key=plain-secret-value",
                stream_id="stream?cookie=session-cookie-value",
                task_type="manifest?password=plain-password-value",
                memory_mode="mode?secret=plain-mode-secret",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual(len(result.saved), 2)
            failed = [payload for event, payload in traces if event == "memory.evolver_writer_failed"][-1]
            self.assertEqual(failed["phase"], "dataset")
            self.assertIn("IsADirectoryError", failed["error"])
            failed_text = json.dumps(failed, ensure_ascii=False)
            traces_text = json.dumps(traces, ensure_ascii=False)
            self.assertNotIn("plain-secret-value", failed_text)
            self.assertNotIn("plain-password-value", failed_text)
            self.assertNotIn("session-cookie-value", failed_text)
            self.assertNotIn("plain-mode-secret", failed_text)
            self.assertNotIn("project-token-value", failed_text)
            self.assertNotIn("plain-secret-value", traces_text)
            self.assertNotIn("plain-password-value", traces_text)
            self.assertNotIn("session-cookie-value", traces_text)
            self.assertNotIn("plain-mode-secret", traces_text)
            self.assertNotIn("project-token-value", traces_text)
            self.assertTrue(str(failed["source_task"]).startswith("redacted_"))

    def test_write_experiences_dataset_redacts_rejected_and_result_errors(self) -> None:
        class SecretRejectedWriter:
            def propose(self, *_: object, **__: object) -> ExperienceWriteResult:
                return ExperienceWriteResult(
                    rejected=(
                        {
                            "reason": "llm_parse_failed",
                            "error": "token=plain-token-value password=plain-password-value",
                        },
                    ),
                    error="cookie=session-cookie-value",
                    llm_used=True,
                    fallback_used=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "writer.jsonl"
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_dataset_path=dataset_path,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.evolver_writer = SecretRejectedWriter()  # type: ignore[assignment]

            manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            record_text = dataset_path.read_text(encoding="utf-8")
            record = json.loads(record_text)
            self.assertNotIn("plain-token-value", record_text)
            self.assertNotIn("plain-password-value", record_text)
            self.assertNotIn("session-cookie-value", record_text)
            self.assertEqual(record["rejected"][0]["error"], "")
            self.assertTrue(record["error"].startswith("redacted_"))

    def test_write_experiences_dataset_redacts_secret_arguments_and_join_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "writer.jsonl"
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_dataset_path=dataset_path,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )

            manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace-api_key=plain-secret-value.jsonl",
                outcome="success",
                source_task="task?api_key=plain-secret-value",
                stream_id="stream-a",
                task_type="manifest",
                tool_history=[
                    _tool_record(
                        "run_tests",
                        ok=True,
                        arguments={
                            "api_key": "plain-secret-value",
                            "token": "plain-token-value",
                            "password": "plain-password-value",
                            "cookie": "session-cookie-value",
                            "secret": "generic-secret-value",
                            "access_key": "plain-access-key-value",
                            "nested": {"private_key": "nested-secret-value"},
                            "command": "pytest tests/test_example.py -q",
                        },
                    )
                ],
            )

            record_text = dataset_path.read_text(encoding="utf-8")
            record = json.loads(record_text)
            self.assertNotIn("plain-secret-value", record_text)
            self.assertNotIn("nested-secret-value", record_text)
            self.assertNotIn("plain-token-value", record_text)
            self.assertNotIn("plain-password-value", record_text)
            self.assertNotIn("session-cookie-value", record_text)
            self.assertNotIn("generic-secret-value", record_text)
            self.assertNotIn("plain-access-key-value", record_text)
            self.assertTrue(record["source_task"].startswith("redacted_"))
            self.assertTrue(record["task_id"].startswith("redacted_"))
            self.assertTrue(record["trace_path"].startswith("redacted_"))
            self.assertEqual(record["steps"][0]["arguments"]["command"], "pytest tests/test_example.py -q")
            self.assertTrue(any(key.startswith("redacted_") for key in record["steps"][0]["arguments"]))
            self.assertTrue(any(key.startswith("redacted_") for key in record["steps"][0]["arguments"]["nested"]))

    def test_write_experiences_keeps_task_context_out_of_persisted_record(self) -> None:
        class TypedProposalWriter:
            def propose(self, *_: object, **__: object) -> ExperienceWriteResult:
                return ExperienceWriteResult(
                    proposals=(
                        ExperienceWriteProposal(
                            tier=ExperienceTier.SKILL,
                            content="Use focused pytest verification after a small patch.",
                            payload=SkillPayload(
                                category="debugging",
                                technique="Focused verification",
                                preconditions=(),
                                steps=("rerun the focused test",),
                            ),
                            confidence=0.9,
                            reason="llm reason",
                        ),
                    ),
                    llm_used=True,
                )

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                    memory_project_key="stream:a",
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.evolver_writer = TypedProposalWriter()  # type: ignore[assignment]

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace.jsonl",
                outcome="success",
                source_task="manifest-task-1",
                stream_id="stream-a",
                task_type="manifest",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            saved = result.saved[0]
            self.assertEqual(saved.source_task, "manifest-task-1")
            self.assertEqual(saved.stream_id, "stream-a")
            self.assertEqual(saved.project_key, "stream:a")
            persisted = experience_to_dict(saved)
            self.assertNotIn("task_type", persisted)
            self.assertNotIn("writer_policy", persisted)
            self.assertNotIn("source_trace", persisted)
            self.assertNotIn("writer_reason", persisted)

    def test_write_experiences_llm_runtime_failure_fallback_records_combined_policy(self) -> None:
        # mode="llm" with FakeLLM: the generic assistant response is not a valid
        # JSON array, so the writer falls back to deterministic proposals. The saved
        # task/run provenance remains on trace, not on each persisted record.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_mode="llm",
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertTrue(result.llm_used)
            self.assertTrue(result.fallback_used)
            self.assertTrue(result.saved)
            for entry in result.saved:
                self.assertNotIn("writer_policy", experience_to_dict(entry))
            saved_payload = [payload for event, payload in traces if event == "memory.evolver_writer_saved"][-1]
            self.assertEqual(saved_payload["writer_policy"], "llm_then_fallback_runtime_v1")

    def test_write_experiences_failure_saves_tip_and_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                stop_reason="max_steps_reached",
                outcome="failure",
                tool_history=[
                    _tool_record("read_file", ok=True, output="code"),
                    _tool_record("run_tests", ok=False, output="failed", reason="failed"),
                ],
            )

            self.assertEqual({entry.tier.value for entry in result.saved}, {"tip", "trajectory"})

    def test_write_experiences_duplicate_content_is_reported_without_new_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            kwargs = {
                "task": "Fix focused pytest failure",
                "run_id": "run-1",
                "stop_reason": "finish_called",
                "outcome": "success",
                "tool_history": [_tool_record("run_tests", ok=True)],
            }

            first = manager.write_experiences_from_run(**kwargs)
            second = manager.write_experiences_from_run(**kwargs)

            self.assertEqual(len(first.saved), 2)
            self.assertEqual(second.saved, ())
            self.assertEqual(len(second.duplicate_ids), 2)

    def test_write_experiences_writer_exception_traces_failure_without_raising(self) -> None:
        class BrokenWriter:
            def propose(self, *_: object, **__: object) -> object:
                raise RuntimeError("writer exploded")

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="full",
                    memory_evolver_writer_enabled=True,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.evolver_writer = BrokenWriter()  # type: ignore[assignment]

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertIn("writer exploded", result.error)
            failed = [payload for event, payload in traces if event == "memory.evolver_writer_failed"][-1]
            self.assertIn("RuntimeError", failed["error"])
            self.assertEqual(failed["phase"], "unknown")

    def test_noop_write_experiences_returns_empty_result_without_writing_or_tracing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(Path(tmp) / "memory", memory_evolver_writer_enabled=True)
            manager = NoopMemoryManager(
                config=config,
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual(result.saved, ())
            self.assertEqual(traces, [])
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())
