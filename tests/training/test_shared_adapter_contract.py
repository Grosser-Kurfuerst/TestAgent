from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest

import yaml

from my_agent.training.opd_trainer import (
    OPDTrainerConfig,
    SharedAdapterConfig,
    canonical_adapter_payload,
    validate_shared_adapter_config,
)


ROOT = Path(__file__).resolve().parents[2]


class _TaskType(Enum):
    CAUSAL_LM = "CAUSAL_LM"


def _peft_config(**overrides):
    values = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": {"q_proj", "k_proj", "v_proj", "o_proj"},
        "task_type": _TaskType.CAUSAL_LM,
        "bias": "none",
        "modules_to_save": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SharedAdapterContractTests(unittest.TestCase):
    def test_none_and_empty_modules_to_save_normalize_identically(self) -> None:
        expected = SharedAdapterConfig()
        self.assertEqual(canonical_adapter_payload(_peft_config()), expected.canonical_payload)
        self.assertEqual(
            canonical_adapter_payload(_peft_config(modules_to_save=[])),
            expected.canonical_payload,
        )
        self.assertEqual(
            expected.adapter_config_hash,
            "sha256:fc2d911dc40bbf3965a70afab1547eea4102a2c1c54bd0a44cbf5b40cbc5f91c",
        )

    def test_existing_adapter_is_validated_exactly(self) -> None:
        expected = SharedAdapterConfig()
        model = SimpleNamespace(peft_config={"default": _peft_config()})
        self.assertEqual(validate_shared_adapter_config(model, expected), "default")
        model.peft_config["default"] = _peft_config(lora_alpha=16)
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            validate_shared_adapter_config(model, expected)

    def test_formal_contract_rejects_wildcards_and_modules_to_save(self) -> None:
        with self.assertRaisesRegex(ValueError, "wildcards"):
            SharedAdapterConfig(target_modules=("all",))
        with self.assertRaisesRegex(ValueError, "must be empty"):
            SharedAdapterConfig(modules_to_save=("lm_head",))

    def test_sft_and_opd_presets_share_the_same_adapter_hash(self) -> None:
        sft = yaml.safe_load((ROOT / "configs/sft_warm_start_qwen3_4b.yaml").read_text())
        opd = yaml.safe_load((ROOT / "configs/opd_paper_train.yaml").read_text())
        sft_adapter = dict(sft["adapter"])
        sft_hash = sft_adapter.pop("adapter_config_hash")
        opd_adapter = dict(opd["trainer"]["shared_adapter"])
        opd_hash = opd_adapter.pop("adapter_config_hash")
        sft_config = SharedAdapterConfig(**sft_adapter)
        opd_config = OPDTrainerConfig.from_mapping({"shared_adapter": opd_adapter}).shared_adapter
        self.assertEqual(sft_config.canonical_payload, opd_config.canonical_payload)
        self.assertEqual(sft_hash, sft_config.adapter_config_hash)
        self.assertEqual(opd_hash, opd_config.adapter_config_hash)
        self.assertEqual(
            sft["model"]["base_revision"],
            "cdbee75f17c01a7cc42f958dc650907174af0554",
        )
        self.assertEqual(
            sft["model"]["base_revision"],
            sft["model"]["tokenizer_revision"],
        )

        opd_adapter["adapter_config_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "adapter_config_hash"):
            OPDTrainerConfig.from_mapping({"shared_adapter": opd_adapter})


if __name__ == "__main__":
    unittest.main()
