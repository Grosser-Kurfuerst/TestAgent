from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from my_agent.config import AgentConfig
from my_agent.llm import FakeLLM
from my_agent.memory import ExperienceStore, MemoryManager
from my_agent.policy.identity import PolicyIdentity, policy_identity_manifest_payload
from my_agent.runtime import CodingAgentRuntime


def _identity(*, revision: str = "model-revision-1") -> PolicyIdentity:
    return PolicyIdentity(
        base_model="model",
        base_revision=revision,
        checkpoint_hash="sha256:" + "1" * 64,
        adapter_hash=None,
        tokenizer_revision="tokenizer-revision-1",
        tokenizer_hash="sha256:" + "2" * 64,
        chat_template_hash="sha256:" + "3" * 64,
    )


def _formal_config(manifest_path: Path) -> AgentConfig:
    return AgentConfig.from_env(
        {
            "MY_AGENT_LLM_PROVIDER": "fake",
            "AGENTCLI_MEMORY_EVOLVER_MODE": "formal",
            "AGENTCLI_POLICY_BASE_MODEL": "model",
            "AGENTCLI_POLICY_BASE_REVISION": "model-revision-1",
            "AGENTCLI_POLICY_TOKENIZER_REVISION": "tokenizer-revision-1",
            "AGENTCLI_POLICY_IDENTITY_MANIFEST": str(manifest_path),
            "AGENTCLI_EMBEDDING_REVISION": "embedding-revision-1",
            "AGENTCLI_MEMORY_DIR": str(manifest_path.parent / "memory"),
        },
        env_file=manifest_path.parent / ".env.test",
        require_env_file=False,
    )


class _TrainablePolicyStub:
    supports_tools = True

    def __init__(self, identity: PolicyIdentity) -> None:
        self._identity = identity

    def chat(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("not used")

    def generate_decision(self, request):
        del request
        raise AssertionError("not used")

    def identity(self) -> PolicyIdentity:
        return self._identity

    def tokenize(self, request):
        del request
        raise AssertionError("not used")

    def forward_logits(self, batch):
        del batch
        raise AssertionError("not used")


class FormalRuntimeValidationTests(unittest.TestCase):
    def test_direct_formal_config_is_revalidated_at_runtime_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "identity.json"
            manifest_path.write_text(
                json.dumps(policy_identity_manifest_payload(_identity())),
                encoding="utf-8",
            )
            invalid = replace(_formal_config(manifest_path), policy_base_revision="main")

            with self.assertRaisesRegex(ValueError, "immutable policy_base_revision"):
                CodingAgentRuntime(config=invalid, llm=_TrainablePolicyStub(_identity()))

    def test_injected_non_trainable_policy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "identity.json"
            manifest_path.write_text(
                json.dumps(policy_identity_manifest_payload(_identity())),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "TrainablePolicy"):
                CodingAgentRuntime(config=_formal_config(manifest_path), llm=FakeLLM())

    def test_injected_trainable_policy_must_match_identity_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "identity.json"
            manifest_path.write_text(
                json.dumps(policy_identity_manifest_payload(_identity())),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                CodingAgentRuntime(
                    config=_formal_config(manifest_path),
                    llm=_TrainablePolicyStub(_identity(revision="other-revision")),
                )

    def test_injected_memory_manager_binding_checks_identity_and_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            manifest_path = root / "identity.json"
            identity = _identity()
            manifest_path.write_text(
                json.dumps(policy_identity_manifest_payload(identity)),
                encoding="utf-8",
            )
            config = _formal_config(manifest_path)
            store = ExperienceStore.from_dir(config.memory_dir)
            manager = object.__new__(MemoryManager)
            manager.config = config
            manager.experience_store = store
            manager.project_key = str(repo.resolve())
            manager.embedding_retriever = object()
            manager.evolver_coordinator = SimpleNamespace(
                store=store,
                project_key=manager.project_key,
                policy_identity=identity,
                retriever=manager.embedding_retriever,
                top_k_per_tier=config.memory_evolver_candidate_top_k_per_tier,
                selected_max_items=config.memory_evolver_selected_max_items,
                selection_token_budget=config.memory_evolver_selection_prompt_tokens,
            )

            manager.require_formal_runtime_binding(
                config=config,
                policy_identity=identity,
                repo_path=repo,
            )
            manager.evolver_coordinator.policy_identity = _identity(revision="other-revision")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                manager.require_formal_runtime_binding(
                    config=config,
                    policy_identity=identity,
                    repo_path=repo,
                )


if __name__ == "__main__":
    unittest.main()
