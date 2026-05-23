from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


TRUE_VALUES = {"1", "true", "yes", "on"}
SUPPORTED_PROVIDERS = {"openai", "fake"}


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

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AgentConfig":
        values = os.environ if env is None else env
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
                "No API key configured. Set MY_AGENT_API_KEY or OPENAI_API_KEY, "
                "or set MY_AGENT_LLM_PROVIDER=fake for local tests."
            )


def _as_bool(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES
