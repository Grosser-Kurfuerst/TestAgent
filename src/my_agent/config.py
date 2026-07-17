from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


TRUE_VALUES = {"1", "true", "yes", "on"}
SUPPORTED_PROVIDERS = {"openai", "fake"}
SUPPORTED_AGENT_MODES = {"react", "plan", "team", "auto"}
SUPPORTED_MEMORY_EVOLVER_MODES = {"off", "formal", "retrieve_select", "full"}
SUPPORTED_MEMORY_EVOLVER_WRITER_MODES = {"fallback", "llm"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class AgentConfig:
    # LLM provider and process-level runtime settings.
    provider: str
    api_key: str
    base_url: str | None
    model: str
    temperature: float
    max_steps: int
    command_timeout: int
    trace_dir: Path
    use_fake_llm: bool
    tool_env_overrides: dict[str, str] = field(default_factory=dict)

    # Dynamic tool loading is opt-in for project-controlled sources.
    tool_config_paths: tuple[Path, ...] = ()
    enable_project_tools: bool = False
    enable_project_plugins: bool = False

    # ReAct loop safety budgets and stop-condition tuning.
    max_iterations: int = 50
    max_tool_calls: int = 200
    max_elapsed_seconds: int = 1800
    token_budget: int | None = None
    stagnation_window: int = 3
    repeated_failure_window: int = 3

    # Context construction and compaction budgets.
    context_window: int = 128_000
    response_reserve_tokens: int = 8_000
    compression_buffer_tokens: int = 8_000
    retain_recent_user_turns: int = 3
    max_tool_result_chars: int = 12_000
    max_summary_input_chars: int = 60_000
    repo_context_budget_tokens: int = 15_360
    tool_schema_budget_tokens: int = 10_240
    context_window_explicit: bool = False
    response_reserve_tokens_explicit: bool = False
    compression_buffer_tokens_explicit: bool = False
    repo_context_budget_tokens_explicit: bool = False
    tool_schema_budget_tokens_explicit: bool = False

    # Plan-and-execute defaults.
    plan_task_max_steps: int = 6
    plan_max_tasks: int = 12
    plan_max_replans: int = 1
    agent_mode: str = "auto"

    # Team orchestration settings; execution is wired in later stages.
    team_worker_count: int = 2
    team_max_steps: int = 12
    team_max_retries: int = 2
    team_step_max_steps: int = 6
    team_dependency_context_chars: int = 4_000
    team_parallel_enabled: bool = True
    team_allow_unapproved_results: bool = False

    # Memory storage, retrieval, and compression defaults.
    memory_enabled: bool = True
    memory_dir: Path = Path("~/.agentcli/memory").expanduser()
    # Optional visibility key for project-scope long-term memory entries.
    memory_project_key: str = ""
    # Storage hard cap only; prompt rendering uses dynamic short_term_allowed.
    memory_short_term_tokens: int = 24_000
    memory_short_term_entries: int = 500
    memory_context_tokens: int = 2_000
    memory_retrieval_limit: int = 8
    memory_compression_trigger_ratio: float = 0.8
    memory_retain_recent_turns: int = 3
    memory_map_chunk_size: int = 5
    memory_tool_result_chars: int = 500
    memory_short_term_tokens_explicit: bool = False
    memory_context_tokens_explicit: bool = False
    memory_tool_result_chars_explicit: bool = False
    memory_evolver_mode: str = "off"
    # Formal OPD fields. The older selector/writer fields below remain only for
    # the pinned legacy baseline and are rejected when formal mode is loaded.
    memory_evolver_candidate_top_k_per_tier: int = 50
    memory_evolver_selection_prompt_tokens: int = 1_800
    memory_evolver_maintenance_interval_tasks: int = 30
    memory_evolver_maintenance_max_turns: int = 8
    memory_evolver_dataset_dir: Path | None = None
    memory_evolver_collection_round: int = 0
    memory_evolver_dataset_split: str = "train"
    memory_evolver_teacher_min_score: float = 0.01
    memory_evolver_writing_top_fraction: float = 0.30
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_revision: str = ""
    policy_backend: str = "transformers"
    policy_base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    policy_base_revision: str = ""
    policy_adapter_path: Path | None = None
    policy_identity_manifest: Path | None = None
    policy_tokenizer_revision: str = ""
    policy_chat_template: str = "model_default"
    policy_dtype: str = "bfloat16"
    policy_device: str = "auto"
    memory_evolver_top_k_per_tier: int = 50
    memory_evolver_selected_max_items: int = 20
    memory_evolver_min_score: float = 0.0
    memory_evolver_min_experience_entries: int = 0
    memory_evolver_tier_caps: dict[str, int] = field(
        default_factory=lambda: {
            "trajectory": 1,
            "tip": 2,
            "skill": 2,
            "tool": 2,
        }
    )
    memory_evolver_tier_weights: dict[str, float] = field(
        default_factory=lambda: {
            "trajectory": 0.90,
            "tip": 1.00,
            "skill": 1.20,
            "tool": 1.10,
        }
    )
    memory_evolver_writer_enabled: bool = False
    memory_evolver_writer_mode: str = "fallback"
    memory_evolver_writer_min_confidence: float = 0.70
    memory_evolver_writer_max_records: int = 6
    memory_evolver_writer_max_input_chars: int = 12_000
    memory_evolver_writer_max_content_chars: int = 1_200
    memory_evolver_writer_dataset_path: Path | None = None

    # Human-in-the-loop approval; disabled by default.
    hitl_enabled: bool = False
    hitl_audit_dir: Path = Path("~/.agentcli/audit").expanduser()
    hitl_non_interactive: str = "reject"
    # Medium-risk handling: ask for approval, allow, or deny.
    hitl_medium_risk_mode: str = "ask"
    hitl_llm_judge_enabled: bool = False

    # Parallel execution and cancellation defaults.
    max_parallel_tools: int = 4
    tool_batch_timeout_seconds: int = 60
    tool_shutdown_grace_seconds: int = 2
    max_process_output_chars: int = 8_000
    plan_parallel_enabled: bool = True
    plan_max_parallel_tasks: int = 4
    plan_task_batch_timeout_seconds: int = 1_800
    team_step_batch_timeout_seconds: int = 1_800

    # MCP dynamic tool integration defaults.
    mcp_enabled: bool = True
    mcp_startup_wait_seconds: int = 8
    mcp_initialize_timeout_seconds: int = 60
    mcp_call_timeout_seconds: int = 60
    mcp_max_startup_workers: int = 8
    mcp_require_approval: bool = True
    mcp_enable_project_servers: bool = True

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        env_file: str | Path | None = None,
        *,
        require_env_file: bool = True,
        include_environment: bool = False,
    ) -> "AgentConfig":
        values = _config_values(
            env=env,
            env_file=env_file,
            require_env_file=require_env_file,
            include_environment=include_environment,
        )
        provider = values.get("MY_AGENT_LLM_PROVIDER", "openai").strip().lower()
        use_fake_llm = _as_bool(values.get("MY_AGENT_USE_FAKE_LLM", "")) or provider == "fake"

        config = cls(
            provider=provider,
            api_key=values.get("MY_AGENT_API_KEY") or values.get("OPENAI_API_KEY", ""),
            base_url=values.get("MY_AGENT_BASE_URL") or values.get("OPENAI_BASE_URL") or None,
            model=values.get("MY_AGENT_MODEL", "gpt-4o-mini"),
            temperature=float(values.get("MY_AGENT_TEMPERATURE", "0.1")),
            max_steps=int(values.get("MY_AGENT_MAX_STEPS", "8")),
            command_timeout=int(values.get("MY_AGENT_COMMAND_TIMEOUT", "60")),
            trace_dir=Path(values.get("MY_AGENT_TRACE_DIR", "traces")),
            use_fake_llm=use_fake_llm,
            tool_config_paths=_tool_config_paths(values),
            enable_project_tools=_as_bool(
                values.get("AGENTCLI_ENABLE_PROJECT_TOOLS", values.get("MY_AGENT_ENABLE_PROJECT_TOOLS", ""))
            ),
            enable_project_plugins=_as_bool(
                values.get("AGENTCLI_ENABLE_PROJECT_PLUGINS", values.get("MY_AGENT_ENABLE_PROJECT_PLUGINS", ""))
            ),
            max_iterations=_as_int(values.get("MY_AGENT_MAX_ITERATIONS"), 50),
            max_tool_calls=_as_int(values.get("MY_AGENT_MAX_TOOL_CALLS"), 200),
            max_elapsed_seconds=_as_int(values.get("MY_AGENT_MAX_ELAPSED_SECONDS"), 1800),
            token_budget=_as_optional_int(values.get("MY_AGENT_TOKEN_BUDGET")),
            stagnation_window=_as_int(values.get("MY_AGENT_STAGNATION_WINDOW"), 3),
            repeated_failure_window=_as_int(values.get("MY_AGENT_REPEATED_FAILURE_WINDOW"), 3),
            context_window=_as_int(values.get("MY_AGENT_CONTEXT_WINDOW"), 128_000),
            context_window_explicit=_has_value(values, "MY_AGENT_CONTEXT_WINDOW"),
            response_reserve_tokens=_as_int(values.get("MY_AGENT_RESPONSE_RESERVE_TOKENS"), 8_000),
            response_reserve_tokens_explicit=_has_value(values, "MY_AGENT_RESPONSE_RESERVE_TOKENS"),
            compression_buffer_tokens=_as_int(values.get("MY_AGENT_COMPRESSION_BUFFER_TOKENS"), 8_000),
            compression_buffer_tokens_explicit=_has_value(values, "MY_AGENT_COMPRESSION_BUFFER_TOKENS"),
            retain_recent_user_turns=_as_int(values.get("MY_AGENT_RETAIN_RECENT_TURNS"), 3),
            max_tool_result_chars=_as_int(values.get("MY_AGENT_MAX_TOOL_RESULT_CHARS"), 12_000),
            max_summary_input_chars=_as_int(values.get("MY_AGENT_MAX_SUMMARY_INPUT_CHARS"), 60_000),
            repo_context_budget_tokens=_as_positive_int(
                values.get("AGENTCLI_REPO_CONTEXT_BUDGET_TOKENS", values.get("MY_AGENT_REPO_CONTEXT_BUDGET_TOKENS")),
                15_360,
            ),
            repo_context_budget_tokens_explicit=_has_any_value(
                values,
                "AGENTCLI_REPO_CONTEXT_BUDGET_TOKENS",
                "MY_AGENT_REPO_CONTEXT_BUDGET_TOKENS",
            ),
            tool_schema_budget_tokens=_as_positive_int(
                values.get("AGENTCLI_TOOL_SCHEMA_BUDGET_TOKENS", values.get("MY_AGENT_TOOL_SCHEMA_BUDGET_TOKENS")),
                10_240,
            ),
            tool_schema_budget_tokens_explicit=_has_any_value(
                values,
                "AGENTCLI_TOOL_SCHEMA_BUDGET_TOKENS",
                "MY_AGENT_TOOL_SCHEMA_BUDGET_TOKENS",
            ),
            plan_task_max_steps=_as_positive_int(
                values.get("AGENTCLI_PLAN_TASK_MAX_STEPS", values.get("MY_AGENT_PLAN_TASK_MAX_STEPS")),
                6,
            ),
            plan_max_tasks=_as_positive_int(
                values.get("AGENTCLI_PLAN_MAX_TASKS", values.get("MY_AGENT_PLAN_MAX_TASKS")),
                12,
            ),
            plan_max_replans=_as_nonnegative_int(
                values.get("AGENTCLI_PLAN_MAX_REPLANS", values.get("MY_AGENT_PLAN_MAX_REPLANS")),
                1,
            ),
            agent_mode=_agent_mode(values.get("AGENTCLI_AGENT_MODE", values.get("MY_AGENT_AGENT_MODE", "auto"))),
            team_worker_count=_as_positive_int(
                values.get("AGENTCLI_TEAM_WORKERS", values.get("MY_AGENT_TEAM_WORKERS")),
                2,
            ),
            team_max_steps=_as_positive_int(
                values.get("AGENTCLI_TEAM_MAX_STEPS", values.get("MY_AGENT_TEAM_MAX_STEPS")),
                12,
            ),
            team_max_retries=_as_nonnegative_int(
                values.get("AGENTCLI_TEAM_MAX_RETRIES", values.get("MY_AGENT_TEAM_MAX_RETRIES")),
                2,
            ),
            team_step_max_steps=_as_positive_int(
                values.get("AGENTCLI_TEAM_STEP_MAX_STEPS", values.get("MY_AGENT_TEAM_STEP_MAX_STEPS")),
                6,
            ),
            team_dependency_context_chars=_as_positive_int(
                values.get(
                    "AGENTCLI_TEAM_DEPENDENCY_CONTEXT_CHARS",
                    values.get("MY_AGENT_TEAM_DEPENDENCY_CONTEXT_CHARS"),
                ),
                4_000,
            ),
            team_parallel_enabled=_as_bool(
                values.get("AGENTCLI_TEAM_PARALLEL", values.get("MY_AGENT_TEAM_PARALLEL", "")),
                default=True,
            ),
            team_allow_unapproved_results=_as_bool(
                values.get(
                    "AGENTCLI_TEAM_ALLOW_UNAPPROVED_RESULTS",
                    values.get("MY_AGENT_TEAM_ALLOW_UNAPPROVED_RESULTS", ""),
                )
            ),
            memory_enabled=_as_bool(
                values.get("AGENTCLI_MEMORY", values.get("MY_AGENT_MEMORY", "")),
                default=True,
            ),
            memory_dir=_memory_dir(values),
            memory_project_key=str(
                values.get("AGENTCLI_MEMORY_PROJECT_KEY") or values.get("MY_AGENT_MEMORY_PROJECT_KEY") or ""
            ).strip(),
            memory_short_term_tokens=_as_positive_int(
                values.get("AGENTCLI_MEMORY_SHORT_TERM_TOKENS", values.get("MY_AGENT_MEMORY_SHORT_TERM_TOKENS")),
                24_000,
            ),
            memory_short_term_tokens_explicit=_has_any_value(
                values,
                "AGENTCLI_MEMORY_SHORT_TERM_TOKENS",
                "MY_AGENT_MEMORY_SHORT_TERM_TOKENS",
            ),
            memory_short_term_entries=_as_positive_int(
                values.get("AGENTCLI_MEMORY_SHORT_TERM_ENTRIES", values.get("MY_AGENT_MEMORY_SHORT_TERM_ENTRIES")),
                500,
            ),
            memory_context_tokens=_as_positive_int(
                values.get("AGENTCLI_MEMORY_CONTEXT_TOKENS", values.get("MY_AGENT_MEMORY_CONTEXT_TOKENS")),
                2_000,
            ),
            memory_context_tokens_explicit=_has_any_value(
                values,
                "AGENTCLI_MEMORY_CONTEXT_TOKENS",
                "MY_AGENT_MEMORY_CONTEXT_TOKENS",
            ),
            memory_retrieval_limit=_as_positive_int(
                values.get("AGENTCLI_MEMORY_RETRIEVAL_LIMIT", values.get("MY_AGENT_MEMORY_RETRIEVAL_LIMIT")),
                8,
            ),
            memory_compression_trigger_ratio=_as_ratio(
                values.get("AGENTCLI_MEMORY_COMPRESSION_TRIGGER_RATIO", values.get("MY_AGENT_MEMORY_COMPRESSION_TRIGGER_RATIO")),
                0.8,
            ),
            memory_retain_recent_turns=_as_positive_int(
                values.get("AGENTCLI_MEMORY_RETAIN_RECENT_TURNS", values.get("MY_AGENT_MEMORY_RETAIN_RECENT_TURNS")),
                3,
            ),
            memory_map_chunk_size=_as_positive_int(
                values.get("AGENTCLI_MEMORY_MAP_CHUNK_SIZE", values.get("MY_AGENT_MEMORY_MAP_CHUNK_SIZE")),
                5,
            ),
            memory_tool_result_chars=_as_positive_int(
                values.get("AGENTCLI_MEMORY_TOOL_RESULT_CHARS", values.get("MY_AGENT_MEMORY_TOOL_RESULT_CHARS")),
                500,
            ),
            memory_tool_result_chars_explicit=_has_any_value(
                values,
                "AGENTCLI_MEMORY_TOOL_RESULT_CHARS",
                "MY_AGENT_MEMORY_TOOL_RESULT_CHARS",
            ),
            memory_evolver_mode=_memory_evolver_mode(values),
            memory_evolver_candidate_top_k_per_tier=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_CANDIDATE_TOP_K_PER_TIER",
                    values.get("MY_AGENT_MEMORY_EVOLVER_CANDIDATE_TOP_K_PER_TIER"),
                ),
                50,
                1,
            ),
            memory_evolver_selection_prompt_tokens=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_SELECTION_PROMPT_TOKENS",
                    values.get("MY_AGENT_MEMORY_EVOLVER_SELECTION_PROMPT_TOKENS"),
                ),
                1_800,
                1,
            ),
            memory_evolver_maintenance_interval_tasks=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_INTERVAL_TASKS",
                    values.get("MY_AGENT_MEMORY_EVOLVER_MAINTENANCE_INTERVAL_TASKS"),
                ),
                30,
                1,
            ),
            memory_evolver_maintenance_max_turns=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_MAX_TURNS",
                    values.get("MY_AGENT_MEMORY_EVOLVER_MAINTENANCE_MAX_TURNS"),
                ),
                8,
                1,
            ),
            memory_evolver_dataset_dir=_optional_path(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_DATASET_DIR",
                    values.get("MY_AGENT_MEMORY_EVOLVER_DATASET_DIR"),
                )
            ),
            memory_evolver_collection_round=_as_nonnegative_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_COLLECTION_ROUND",
                    values.get("MY_AGENT_MEMORY_EVOLVER_COLLECTION_ROUND"),
                ),
                0,
            ),
            memory_evolver_dataset_split=_opd_dataset_split(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_DATASET_SPLIT",
                    values.get("MY_AGENT_MEMORY_EVOLVER_DATASET_SPLIT"),
                )
            ),
            memory_evolver_teacher_min_score=_as_min_float(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_TEACHER_MIN_SCORE",
                    values.get("MY_AGENT_MEMORY_EVOLVER_TEACHER_MIN_SCORE"),
                ),
                0.01,
                0.0,
            ),
            memory_evolver_writing_top_fraction=_as_ratio(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_WRITING_TOP_FRACTION",
                    values.get("MY_AGENT_MEMORY_EVOLVER_WRITING_TOP_FRACTION"),
                ),
                0.30,
            ),
            embedding_model=str(
                values.get("AGENTCLI_EMBEDDING_MODEL")
                or values.get("MY_AGENT_EMBEDDING_MODEL")
                or "Qwen/Qwen3-Embedding-0.6B"
            ).strip(),
            embedding_revision=str(
                values.get("AGENTCLI_EMBEDDING_REVISION")
                or values.get("MY_AGENT_EMBEDDING_REVISION")
                or ""
            ).strip(),
            policy_backend=str(
                values.get("AGENTCLI_POLICY_BACKEND")
                or values.get("MY_AGENT_POLICY_BACKEND")
                or "transformers"
            ).strip().lower(),
            policy_base_model=str(
                values.get("AGENTCLI_POLICY_BASE_MODEL")
                or values.get("MY_AGENT_POLICY_BASE_MODEL")
                or "Qwen/Qwen3-4B-Instruct-2507"
            ).strip(),
            policy_base_revision=str(
                values.get("AGENTCLI_POLICY_BASE_REVISION")
                or values.get("MY_AGENT_POLICY_BASE_REVISION")
                or ""
            ).strip(),
            policy_adapter_path=_optional_path(
                values.get("AGENTCLI_POLICY_ADAPTER_PATH", values.get("MY_AGENT_POLICY_ADAPTER_PATH"))
            ),
            policy_identity_manifest=_optional_path(
                values.get(
                    "AGENTCLI_POLICY_IDENTITY_MANIFEST",
                    values.get("MY_AGENT_POLICY_IDENTITY_MANIFEST"),
                )
            ),
            policy_tokenizer_revision=str(
                values.get("AGENTCLI_POLICY_TOKENIZER_REVISION")
                or values.get("MY_AGENT_POLICY_TOKENIZER_REVISION")
                or ""
            ).strip(),
            policy_chat_template=str(
                values.get("AGENTCLI_POLICY_CHAT_TEMPLATE")
                or values.get("MY_AGENT_POLICY_CHAT_TEMPLATE")
                or "model_default"
            ).strip(),
            policy_dtype=str(
                values.get("AGENTCLI_POLICY_DTYPE")
                or values.get("MY_AGENT_POLICY_DTYPE")
                or "bfloat16"
            ).strip().lower(),
            policy_device=str(
                values.get("AGENTCLI_POLICY_DEVICE")
                or values.get("MY_AGENT_POLICY_DEVICE")
                or "auto"
            ).strip(),
            memory_evolver_top_k_per_tier=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER",
                    values.get("MY_AGENT_MEMORY_EVOLVER_TOP_K_PER_TIER"),
                ),
                50,
                1,
            ),
            memory_evolver_selected_max_items=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS",
                    values.get("MY_AGENT_MEMORY_EVOLVER_SELECTED_MAX_ITEMS"),
                ),
                20,
                1,
            ),
            memory_evolver_min_score=_as_min_float(
                values.get("AGENTCLI_MEMORY_EVOLVER_MIN_SCORE", values.get("MY_AGENT_MEMORY_EVOLVER_MIN_SCORE")),
                0.0,
                0.0,
            ),
            memory_evolver_min_experience_entries=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES",
                    values.get("MY_AGENT_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES"),
                ),
                0,
                0,
            ),
            memory_evolver_writer_enabled=_as_bool(
                values.get("AGENTCLI_MEMORY_EVOLVER_WRITER", values.get("MY_AGENT_MEMORY_EVOLVER_WRITER", ""))
            ),
            memory_evolver_writer_mode=_memory_evolver_writer_mode(values),
            memory_evolver_writer_min_confidence=_as_probability(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE",
                    values.get("MY_AGENT_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE"),
                ),
                0.70,
            ),
            memory_evolver_writer_max_records=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS",
                    values.get("MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_RECORDS"),
                ),
                6,
                0,
            ),
            memory_evolver_writer_max_input_chars=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS",
                    values.get("MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS"),
                ),
                12_000,
                0,
            ),
            memory_evolver_writer_max_content_chars=_as_min_int(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS",
                    values.get("MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS"),
                ),
                1_200,
                0,
            ),
            memory_evolver_writer_dataset_path=_optional_path(
                values.get(
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_DATASET_PATH",
                    values.get("MY_AGENT_MEMORY_EVOLVER_WRITER_DATASET_PATH"),
                )
            ),
            hitl_enabled=_as_bool(values.get("AGENTCLI_HITL", values.get("MY_AGENT_HITL", ""))),
            hitl_audit_dir=_hitl_audit_dir(values),
            hitl_non_interactive=_hitl_non_interactive(
                values.get("AGENTCLI_HITL_NON_INTERACTIVE", values.get("MY_AGENT_HITL_NON_INTERACTIVE", "reject"))
            ),
            hitl_medium_risk_mode=_hitl_medium_risk_mode(
                values.get("AGENTCLI_HITL_MEDIUM_RISK_MODE", values.get("MY_AGENT_HITL_MEDIUM_RISK_MODE", "ask"))
            ),
            hitl_llm_judge_enabled=_as_bool(
                values.get("AGENTCLI_HITL_LLM_JUDGE", values.get("MY_AGENT_HITL_LLM_JUDGE", ""))
            ),
            max_parallel_tools=_as_positive_int(
                values.get("AGENTCLI_MAX_PARALLEL_TOOLS", values.get("MY_AGENT_MAX_PARALLEL_TOOLS")),
                4,
            ),
            tool_batch_timeout_seconds=_as_positive_int(
                values.get(
                    "AGENTCLI_TOOL_BATCH_TIMEOUT_SECONDS",
                    values.get("MY_AGENT_TOOL_BATCH_TIMEOUT_SECONDS"),
                ),
                60,
            ),
            tool_shutdown_grace_seconds=_as_nonnegative_int(
                values.get(
                    "AGENTCLI_TOOL_SHUTDOWN_GRACE_SECONDS",
                    values.get("MY_AGENT_TOOL_SHUTDOWN_GRACE_SECONDS"),
                ),
                2,
            ),
            max_process_output_chars=max(
                1_000,
                _as_positive_int(
                    values.get("AGENTCLI_MAX_PROCESS_OUTPUT_CHARS", values.get("MY_AGENT_MAX_PROCESS_OUTPUT_CHARS")),
                    8_000,
                ),
            ),
            plan_parallel_enabled=_as_bool(
                values.get("AGENTCLI_PLAN_PARALLEL", values.get("MY_AGENT_PLAN_PARALLEL", "")),
                default=True,
            ),
            plan_max_parallel_tasks=_as_positive_int(
                values.get("AGENTCLI_PLAN_MAX_PARALLEL_TASKS", values.get("MY_AGENT_PLAN_MAX_PARALLEL_TASKS")),
                4,
            ),
            plan_task_batch_timeout_seconds=_as_positive_int(
                values.get(
                    "AGENTCLI_PLAN_TASK_BATCH_TIMEOUT_SECONDS",
                    values.get("MY_AGENT_PLAN_TASK_BATCH_TIMEOUT_SECONDS"),
                ),
                1_800,
            ),
            team_step_batch_timeout_seconds=_as_positive_int(
                values.get(
                    "AGENTCLI_TEAM_STEP_BATCH_TIMEOUT_SECONDS",
                    values.get("MY_AGENT_TEAM_STEP_BATCH_TIMEOUT_SECONDS"),
                ),
                1_800,
            ),
            mcp_enabled=_as_bool(values.get("AGENTCLI_MCP", values.get("MY_AGENT_MCP", "")), default=True),
            mcp_startup_wait_seconds=_as_nonnegative_int(
                values.get("AGENTCLI_MCP_STARTUP_WAIT_SECONDS", values.get("MY_AGENT_MCP_STARTUP_WAIT_SECONDS")),
                8,
            ),
            mcp_initialize_timeout_seconds=_as_positive_int(
                values.get(
                    "AGENTCLI_MCP_INITIALIZE_TIMEOUT_SECONDS",
                    values.get("MY_AGENT_MCP_INITIALIZE_TIMEOUT_SECONDS"),
                ),
                60,
            ),
            mcp_call_timeout_seconds=_as_positive_int(
                values.get("AGENTCLI_MCP_CALL_TIMEOUT_SECONDS", values.get("MY_AGENT_MCP_CALL_TIMEOUT_SECONDS")),
                60,
            ),
            mcp_max_startup_workers=_as_positive_int(
                values.get("AGENTCLI_MCP_MAX_STARTUP_WORKERS", values.get("MY_AGENT_MCP_MAX_STARTUP_WORKERS")),
                8,
            ),
            mcp_require_approval=_as_bool(
                values.get("AGENTCLI_MCP_REQUIRE_APPROVAL", values.get("MY_AGENT_MCP_REQUIRE_APPROVAL", "")),
                default=True,
            ),
            mcp_enable_project_servers=_as_bool(
                values.get(
                    "AGENTCLI_MCP_ENABLE_PROJECT_SERVERS",
                    values.get("MY_AGENT_MCP_ENABLE_PROJECT_SERVERS", ""),
                ),
                default=True,
            ),
        )
        _validate_formal_evolver_config(config, values)
        return config

    def require_valid_provider(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise RuntimeError(f"Unsupported MY_AGENT_LLM_PROVIDER={self.provider!r}. Supported providers: {supported}.")

    def require_valid_formal_evolver(self) -> None:
        """Validate intrinsic formal-mode requirements for every runtime entry point."""

        _validate_formal_evolver_config(self, {})

    def require_api_key(self) -> None:
        self.require_valid_provider()
        if self.use_fake_llm:
            return
        if not self.api_key:
            raise RuntimeError(
                "No API key configured. Set MY_AGENT_API_KEY in .env, "
                "or set MY_AGENT_LLM_PROVIDER=fake in .env for local tests."
            )


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in TRUE_VALUES


def _has_value(values: Mapping[str, str], key: str) -> bool:
    value = values.get(key)
    return value is not None and bool(value.strip())


def _has_any_value(values: Mapping[str, str], *keys: str) -> bool:
    return any(_has_value(values, key) for key in keys)


def _as_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _as_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _as_positive_int(value: str | None, default: int) -> int:
    parsed = _as_int(value, default)
    if parsed < 1:
        raise ValueError("plan configuration values must be >= 1.")
    return parsed


def _as_nonnegative_int(value: str | None, default: int) -> int:
    parsed = _as_int(value, default)
    if parsed < 0:
        raise ValueError("plan configuration values must be >= 0.")
    return parsed


def _agent_mode(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized not in SUPPORTED_AGENT_MODES:
        supported = ", ".join(sorted(SUPPORTED_AGENT_MODES))
        raise ValueError(f"Unsupported AGENTCLI_AGENT_MODE={value!r}. Supported modes: {supported}.")
    return normalized


def _memory_evolver_mode(values: Mapping[str, str]) -> str:
    raw_mode = values.get("AGENTCLI_MEMORY_EVOLVER_MODE", values.get("MY_AGENT_MEMORY_EVOLVER_MODE"))
    if raw_mode is not None and raw_mode.strip():
        normalized = raw_mode.strip().lower()
    elif _as_bool(values.get("AGENTCLI_MEMORY_EVOLVER", values.get("MY_AGENT_MEMORY_EVOLVER", ""))):
        normalized = "retrieve_select"
    else:
        normalized = "off"
    if normalized not in SUPPORTED_MEMORY_EVOLVER_MODES:
        supported = ", ".join(sorted(SUPPORTED_MEMORY_EVOLVER_MODES))
        raise ValueError(f"Unsupported AGENTCLI_MEMORY_EVOLVER_MODE={raw_mode!r}. Supported modes: {supported}.")
    return normalized


def _memory_evolver_writer_mode(values: Mapping[str, str]) -> str:
    raw_mode = values.get("AGENTCLI_MEMORY_EVOLVER_WRITER_MODE", values.get("MY_AGENT_MEMORY_EVOLVER_WRITER_MODE"))
    normalized = (raw_mode or "fallback").strip().lower()
    if normalized not in SUPPORTED_MEMORY_EVOLVER_WRITER_MODES:
        supported = ", ".join(sorted(SUPPORTED_MEMORY_EVOLVER_WRITER_MODES))
        raise ValueError(
            f"Unsupported AGENTCLI_MEMORY_EVOLVER_WRITER_MODE={raw_mode!r}. Supported modes: {supported}."
        )
    return normalized


def _opd_dataset_split(value: str | None) -> str:
    normalized = (value or "train").strip().lower()
    if normalized not in {"train", "validation", "test"}:
        raise ValueError("OPD dataset split must be train, validation, or test")
    return normalized


def _validate_formal_evolver_config(
    config: AgentConfig,
    values: Mapping[str, str],
) -> None:
    if config.memory_evolver_mode != "formal":
        return
    from my_agent.training.formal_contract import (
        FORMAL_FORBIDDEN_LEGACY_CONFIG_KEYS,
        FORMAL_LEGACY_CONFIG_DEFAULTS,
    )

    forbidden = sorted(
        key for key in FORMAL_FORBIDDEN_LEGACY_CONFIG_KEYS if _has_value(values, key)
    )
    if forbidden:
        raise ValueError(
            "formal memory evolver configuration rejects legacy rule fields: "
            + ", ".join(forbidden)
        )
    non_default_legacy_fields = sorted(
        field_name
        for field_name, default_value in FORMAL_LEGACY_CONFIG_DEFAULTS.items()
        if getattr(config, field_name) != default_value
    )
    if non_default_legacy_fields:
        raise ValueError(
            "formal memory evolver configuration rejects non-default legacy rule fields: "
            + ", ".join(non_default_legacy_fields)
        )
    if config.policy_backend != "transformers":
        raise ValueError("formal memory evolver requires policy_backend=transformers")
    for field_name in ("policy_base_revision", "policy_tokenizer_revision", "embedding_revision"):
        value = str(getattr(config, field_name)).strip()
        if not value:
            raise ValueError(f"formal memory evolver requires {field_name}")
        if value.lower() in {"main", "master", "head", "latest"}:
            raise ValueError(f"formal memory evolver requires an immutable {field_name}")
    if not config.policy_base_model or not config.embedding_model:
        raise ValueError("formal memory evolver requires policy and embedding model names")
    if config.policy_identity_manifest is None:
        raise ValueError("formal memory evolver requires policy_identity_manifest")


def _as_ratio(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    parsed = float(value)
    if not (0.0 < parsed <= 1.0):
        raise ValueError("memory_compression_trigger_ratio must be in (0, 1].")
    return parsed


def _as_min_int(value: str | None, default: int, minimum: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _as_min_float(value: str | None, default: float, minimum: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _as_probability(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, min(1.0, parsed))


def _optional_path(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    return Path(value).expanduser()


def _memory_dir(values: Mapping[str, str]) -> Path:
    raw = values.get("AGENTCLI_MEMORY_DIR", values.get("MY_AGENT_MEMORY_DIR", ""))
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path("~/.agentcli/memory").expanduser()


def _hitl_audit_dir(values: Mapping[str, str]) -> Path:
    raw = values.get("AGENTCLI_HITL_AUDIT_DIR", values.get("MY_AGENT_HITL_AUDIT_DIR", ""))
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path("~/.agentcli/audit").expanduser()


def _hitl_medium_risk_mode(value: str | None) -> str:
    normalized = (value or "ask").strip().lower()
    if normalized not in {"ask", "allow", "deny"}:
        raise ValueError("AGENTCLI_HITL_MEDIUM_RISK_MODE must be one of: ask, allow, deny.")
    return normalized


def _hitl_non_interactive(value: str | None) -> str:
    normalized = (value or "reject").strip().lower()
    if normalized != "reject":
        raise ValueError("AGENTCLI_HITL_NON_INTERACTIVE currently supports only 'reject'.")
    return normalized


def _config_values(
    env: Mapping[str, str] | None,
    env_file: str | Path | None,
    *,
    require_env_file: bool,
    include_environment: bool,
) -> dict[str, str]:
    values = _read_env_file(Path(env_file) if env_file is not None else DEFAULT_ENV_FILE, required=require_env_file)
    if include_environment:
        values.update({key: value for key, value in os.environ.items()})
    if env is None:
        return values
    values.update(env)
    return values


def _read_env_file(path: Path, *, required: bool = True) -> dict[str, str]:
    if not path.exists():
        if not required:
            return {}
        raise FileNotFoundError(f"Configuration file not found: {path}. Copy .env.example to .env and fill it in.")
    return {key: value or "" for key, value in dotenv_values(path).items()}


def _tool_config_paths(values: Mapping[str, str]) -> tuple[Path, ...]:
    raw = values.get("AGENTCLI_TOOL_CONFIGS") or values.get("MY_AGENT_TOOL_CONFIGS")
    if raw:
        return tuple(Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip())
    return (Path("~/.config/agentcli/tools.json").expanduser(),)
