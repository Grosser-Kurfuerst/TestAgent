#!/usr/bin/env python3
"""Run the patched LLaMA-Factory converter and supervised encoder."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import argparse
import json
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--dataset-info", required=True)
    parser.add_argument("--cutoff-len", required=True, type=int)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    converter_module, supervised_module = _load_locked_modules(Path(args.checkout))

    rows = _load_array(Path(args.input))
    dataset_info = _load_object(Path(args.dataset_info))
    dataset_config = dataset_info["agentcli_sft_train"]
    columns = dataset_config["columns"]
    tags = dataset_config["tags"]
    attr = SimpleNamespace(
        load_from="file",
        formatting=dataset_config["formatting"],
        ranking=False,
        messages=columns["messages"],
        tools=columns["tools"],
        system=None,
        images=None,
        videos=None,
        audios=None,
        chosen=None,
        rejected=None,
        kto_tag=None,
        role_tag=tags["role_tag"],
        content_tag=tags["content_tag"],
        user_tag=tags["user_tag"],
        assistant_tag=tags["assistant_tag"],
        observation_tag=tags["observation_tag"],
        function_tag=tags["function_tag"],
        system_tag=tags["system_tag"],
    )
    data_args = SimpleNamespace(
        media_dir="",
        train_on_prompt=False,
        mask_history=True,
        cutoff_len=args.cutoff_len,
    )
    converter = converter_module.get_dataset_converter(
        "agentcli_openai_tools",
        attr,
        data_args,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    processor = object.__new__(supervised_module.SupervisedDatasetProcessor)
    processor.tokenizer = tokenizer
    processor.data_args = data_args

    results: list[dict[str, Any]] = []
    for row in rows:
        aligned = converter(row)
        messages = aligned.get("_agentcli_messages")
        tools = aligned.get("_agentcli_tools")
        _require_object_arguments(messages)
        input_ids, labels = processor._encode_agentcli_example(messages, tools)
        results.append({
            "sample_id": row["sample_id"],
            "input_ids": input_ids,
            "labels": labels,
            "normalized_template_input_hash": _canonical_sha256({
                "messages": messages[:-1],
                "target": messages[-1],
                "tools": tools,
            }),
            "arguments_are_objects": True,
        })
    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _load_locked_modules(checkout: Path):
    source = checkout / "src/llamafactory"
    packages = {
        "llamafactory": source,
        "llamafactory.extras": source / "extras",
        "llamafactory.data": source / "data",
        "llamafactory.data.processor": source / "data/processor",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    constants = _load_module(
        "llamafactory.extras.constants",
        source / "extras/constants.py",
    )
    logging_module = _load_module(
        "llamafactory.extras.logging",
        source / "extras/logging.py",
    )
    sys.modules["llamafactory.extras"].constants = constants
    sys.modules["llamafactory.extras"].logging = logging_module
    _load_module("llamafactory.data.data_utils", source / "data/data_utils.py")
    converter = _load_module("llamafactory.data.converter", source / "data/converter.py")
    _load_module(
        "llamafactory.data.processor.processor_utils",
        source / "data/processor/processor_utils.py",
    )
    supervised = _load_module(
        "llamafactory.data.processor.supervised",
        source / "data/processor/supervised.py",
    )
    return converter, supervised


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load locked LLaMA-Factory module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_object_arguments(messages: Any) -> None:
    if not isinstance(messages, list):
        raise ValueError("patched ingestion did not preserve messages")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("patched ingestion message must be an object")
        for call in message.get("tool_calls", []):
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("arguments"), dict):
                raise ValueError("patched ingestion function.arguments must be an object")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(payload).hexdigest()


def _load_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("parity input must be an array of objects")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dataset info must be an object")
    return value


if __name__ == "__main__":
    main()
