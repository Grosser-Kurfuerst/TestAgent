#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_DIR="${DATASET_DIR:-data/llamafactory}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/coding_agent_lora}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
MODEL_REVISION="${MODEL_REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${BASE_MODEL}}"

TEMPLATE="${TEMPLATE:-qwen3_5_nothink}"
FINETUNING_TYPE="${FINETUNING_TYPE:-lora}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET="${LORA_TARGET:-q_proj,k_proj,v_proj,o_proj}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
CUTOFF_LEN="${CUTOFF_LEN:-8192}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
TRAIN_ON_PROMPT="${TRAIN_ON_PROMPT:-false}"
MASK_HISTORY="${MASK_HISTORY:-true}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SEED="${SEED:-42}"
BF16="${BF16:-true}"
FP16="${FP16:-false}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LLAMAFACTORY_CMD="${LLAMAFACTORY_CMD:-llamafactory-cli}"
LLAMAFACTORY_VERSION="${LLAMAFACTORY_VERSION:-0.9.6.dev0}"
PYTHON_CMD="${PYTHON_CMD:-python}"

export CUDA_VISIBLE_DEVICES

if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
  echo "Missing ${DATASET_DIR}/dataset_info.json. Run export-alpaca before training." >&2
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/train_alpaca.json" ]]; then
  echo "Missing ${DATASET_DIR}/train_alpaca.json. Run export-alpaca before training." >&2
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/val_alpaca.json" ]]; then
  echo "Missing ${DATASET_DIR}/val_alpaca.json. Run export-alpaca before training." >&2
  exit 1
fi
if ! command -v "${LLAMAFACTORY_CMD}" >/dev/null 2>&1; then
  echo "Missing ${LLAMAFACTORY_CMD}. Install LLaMA-Factory before running this script." >&2
  exit 1
fi
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON_CMD}. A Python interpreter is required to validate datasets." >&2
  exit 1
fi

if [[ "${BASE_MODEL}" != "Qwen/Qwen3.5-4B" \
  || "${MODEL_REVISION}" != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" \
  || "${MODEL_NAME_OR_PATH}" != "${BASE_MODEL}" \
  || -n "${LOCAL_MODEL_DIR}" ]]; then
  echo "OPD warm-start requires the pinned Qwen3.5-4B base model and revision; local model overrides are not supported." >&2
  exit 1
fi

LLAMAFACTORY_VERSION_OUTPUT="$("${LLAMAFACTORY_CMD}" version 2>&1)"
if [[ "${LLAMAFACTORY_VERSION_OUTPUT}" != *"${LLAMAFACTORY_VERSION}"* ]]; then
  echo "LLaMA-Factory version mismatch. Expected ${LLAMAFACTORY_VERSION}." >&2
  echo "${LLAMAFACTORY_VERSION_OUTPUT}" >&2
  exit 1
fi

"${PYTHON_CMD}" -c '
import json
import sys
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"SFT dataset must be a non-empty JSON array: {path}")
' "${DATASET_DIR}/train_alpaca.json" "${DATASET_DIR}/val_alpaca.json"

if [[ "${FINETUNING_TYPE}" != "lora" \
  || "${LORA_RANK}" != "16" \
  || "${LORA_ALPHA}" != "32" \
  || "${LORA_DROPOUT}" != "0" && "${LORA_DROPOUT}" != "0.0" \
  || "${LORA_TARGET}" != "q_proj,k_proj,v_proj,o_proj" \
  || "${TEMPLATE}" != "qwen3_5_nothink" \
  || "${TRAIN_ON_PROMPT}" != "false" \
  || "${MASK_HISTORY}" != "true" ]]; then
  echo "SFT LoRA settings must match the OPD shared adapter contract." >&2
  exit 1
fi

"${LLAMAFACTORY_CMD}" train \
  --stage sft \
  --do_train true \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --model_revision "${MODEL_REVISION}" \
  --trust_remote_code true \
  --dataset_dir "${DATASET_DIR}" \
  --dataset coding_agent_train \
  --eval_dataset coding_agent_val \
  --template "${TEMPLATE}" \
  --finetuning_type "${FINETUNING_TYPE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target "${LORA_TARGET}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir true \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --learning_rate "${LEARNING_RATE}" \
  --train_on_prompt "${TRAIN_ON_PROMPT}" \
  --mask_history "${MASK_HISTORY}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --cutoff_len "${CUTOFF_LEN}" \
  --max_samples "${MAX_SAMPLES}" \
  --lr_scheduler_type cosine \
  --warmup_ratio "${WARMUP_RATIO}" \
  --seed "${SEED}" \
  --do_eval true \
  --eval_strategy steps \
  --save_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --bf16 "${BF16}" \
  --fp16 "${FP16}" \
  --report_to none \
  --plot_loss true

"${PYTHON_CMD}" -c '
import hashlib
import json
import os
import sys

(
    output_dir,
    base_model,
    model_revision,
    template,
    rank,
    alpha,
    dropout,
    target_modules,
    llamafactory_version,
) = sys.argv[1:]
adapter_config = {
    "rank": int(rank),
    "alpha": int(alpha),
    "dropout": float(dropout),
    "target_modules": sorted(target_modules.split(",")),
    "task_type": "CAUSAL_LM",
    "bias": "none",
    "modules_to_save": [],
}
canonical = json.dumps(
    adapter_config,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
manifest = {
    "schema_version": "agentcli-legacy-sft-training-v1",
    "base_model": base_model,
    "model_revision": model_revision,
    "tokenizer_revision": model_revision,
    "template": template,
    "llamafactory_version": llamafactory_version,
    "adapter_config": adapter_config,
    "adapter_config_hash": "sha256:" + hashlib.sha256(canonical).hexdigest(),
}
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "sft_training_manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
    handle.write("\n")
' "${OUTPUT_DIR}" "${BASE_MODEL}" "${MODEL_REVISION}" "${TEMPLATE}" \
  "${LORA_RANK}" "${LORA_ALPHA}" "${LORA_DROPOUT}" "${LORA_TARGET}" \
  "${LLAMAFACTORY_VERSION}"
