from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import yaml

from scripts.train_opd_evolver import (
    _configure_gradient_checkpointing,
    _required_bool,
)
from my_agent.training.opd_trainer import OPDTrainerConfig


ROOT = Path(__file__).resolve().parents[2]


class GradientCheckpointingConfigurationTests(unittest.TestCase):
    def test_enables_checkpointing_input_grads_and_disables_cache(self) -> None:
        class BaseModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(use_cache=True)
                self.checkpointing_enabled = False
                self.input_grads_enabled = False

            def gradient_checkpointing_enable(self) -> None:
                self.checkpointing_enabled = True

            def enable_input_require_grads(self) -> None:
                self.input_grads_enabled = True

        base = BaseModel()
        wrapper = SimpleNamespace(get_base_model=lambda: base)

        _configure_gradient_checkpointing(wrapper, enabled=True)

        self.assertTrue(base.checkpointing_enabled)
        self.assertTrue(base.input_grads_enabled)
        self.assertFalse(base.config.use_cache)

    def test_disabled_mode_leaves_model_untouched(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(use_cache=True))

        _configure_gradient_checkpointing(model, enabled=False)

        self.assertTrue(model.config.use_cache)

    def test_gradient_checkpointing_config_requires_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            _required_bool("true", "policy.gradient_checkpointing")


class SingleGpuConfigTests(unittest.TestCase):
    def test_16gb_train_config_keeps_bf16_shared_lora_contract(self) -> None:
        payload = yaml.safe_load((ROOT / "configs/opd_16gb_train.yaml").read_text())
        policy = payload["policy"]
        trainer = OPDTrainerConfig.from_mapping(payload["trainer"])

        self.assertEqual(policy["dtype"], "bfloat16")
        self.assertEqual(policy["device"], "cuda")
        self.assertTrue(policy["gradient_checkpointing"])
        self.assertEqual(trainer.batch_size, 1)
        self.assertEqual(trainer.gradient_accumulation_steps, 16)
        self.assertEqual(trainer.vocab_chunk_size, 256)
        self.assertEqual(trainer.samples_per_epoch, 16)
        self.assertEqual(
            trainer.shared_adapter.adapter_config_hash,
            "sha256:fc2d911dc40bbf3965a70afab1547eea4102a2c1c54bd0a44cbf5b40cbc5f91c",
        )
        self.assertNotIn("quantization_bit", policy)

    def test_16gb_recollection_pins_models_tasks_and_token_limit(self) -> None:
        payload = yaml.safe_load(
            (ROOT / "configs/opd_16gb_recollection.yaml").read_text()
        )

        self.assertEqual(
            payload["environment"]["AGENTCLI_EMBEDDING_REVISION"],
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        )
        self.assertEqual(
            payload["environment"]["AGENTCLI_POLICY_BASE_REVISION"],
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        self.assertEqual(
            payload["environment"]["AGENTCLI_POLICY_TOKENIZER_REVISION"],
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        self.assertEqual(
            payload["environment"]["AGENTCLI_POLICY_IDENTITY_MANIFEST"],
            "outputs/opd/M0/policy_identity_manifest.json",
        )
        self.assertEqual(
            payload["environment"]["AGENTCLI_POLICY_ADAPTER_PATH"],
            "outputs/opd/M0/adapter",
        )
        for round_index in (0, 1):
            commands = payload["rounds"][round_index]["collection_commands"]
            task_command = commands[0]
            task_path = task_command[task_command.index("--tasks") + 1]
            self.assertTrue(task_path.startswith("data/sft_raw/mbpp_train/tasks/"))
            self.assertNotIn("REPLACE_WITH", task_path)
            build_command = commands[2]
            token_index = build_command.index("--max-new-tokens")
            self.assertEqual(build_command[token_index + 1], "1024")

    def test_paper_recollection_preset_remains_separate(self) -> None:
        payload = yaml.safe_load(
            (ROOT / "configs/opd_paper_recollection.yaml").read_text()
        )

        self.assertEqual(payload["root"], "../runs/opd-paper-recollection")
        self.assertEqual(payload["numerical_reproduction"]["training_tasks_total"], 7000)


if __name__ == "__main__":
    unittest.main()
