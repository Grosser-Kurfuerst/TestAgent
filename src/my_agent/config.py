from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


TRUE_VALUES = {"1", "true", "yes", "on"}
SUPPORTED_PROVIDERS = {"openai", "fake"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str
    temperature: float
    max_steps: int
    command_timeout: int
    trace_dir: Path
    use_fake_llm: bool
    tool_config_paths: tuple[Path, ...] = ()
    enable_project_tools: bool = False
    enable_project_plugins: bool = False
    max_iterations: int = 50
    max_tool_calls: int = 200
    max_elapsed_seconds: int = 1800
    token_budget: int | None = None
    stagnation_window: int = 3
    repeated_failure_window: int = 3
    context_window: int = 128_000
    response_reserve_tokens: int = 8_000
    compression_buffer_tokens: int = 8_000
    retain_recent_user_turns: int = 3
    max_tool_result_chars: int = 12_000
    max_summary_input_chars: int = 60_000

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

        return cls(
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
            response_reserve_tokens=_as_int(values.get("MY_AGENT_RESPONSE_RESERVE_TOKENS"), 8_000),
            compression_buffer_tokens=_as_int(values.get("MY_AGENT_COMPRESSION_BUFFER_TOKENS"), 8_000),
            retain_recent_user_turns=_as_int(values.get("MY_AGENT_RETAIN_RECENT_TURNS"), 3),
            max_tool_result_chars=_as_int(values.get("MY_AGENT_MAX_TOOL_RESULT_CHARS"), 12_000),
            max_summary_input_chars=_as_int(values.get("MY_AGENT_MAX_SUMMARY_INPUT_CHARS"), 60_000),
        )

    def require_valid_provider(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise RuntimeError(f"Unsupported MY_AGENT_LLM_PROVIDER={self.provider!r}. Supported providers: {supported}.")

    def require_api_key(self) -> None:
        self.require_valid_provider()
        if self.use_fake_llm:
            return
        if not self.api_key:
            raise RuntimeError(
                "No API key configured. Set MY_AGENT_API_KEY in .env, "
                "or set MY_AGENT_LLM_PROVIDER=fake in .env for local tests."
            )


def _as_bool(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def _as_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _as_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


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
