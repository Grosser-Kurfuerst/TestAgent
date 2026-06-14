#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_DIR="${DATASET_DIR:-data/llamafactory}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/coding_agent_lora}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${BASE_MODEL}}"

if [[ -n "${LOCAL_MODEL_DIR}" ]]; then
  MODEL_NAME_OR_PATH="${LOCAL_MODEL_DIR}"
fi

TEMPLATE="${TEMPLATE:-qwen}"
FINETUNING_TYPE="${FINETUNING_TYPE:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET="${LORA_TARGET:-all}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
CUTOFF_LEN="${CUTOFF_LEN:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
BF16="${BF16:-true}"
FP16="${FP16:-false}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LLAMAFACTORY_CMD="${LLAMAFACTORY_CMD:-llamafactory-cli}"

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

"${LLAMAFACTORY_CMD}" train \
  --stage sft \
  --do_train true \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --trust_remote_code true \
  --dataset_dir "${DATASET_DIR}" \
  --dataset coding_agent_train \
  --eval_dataset coding_agent_val \
  --template "${TEMPLATE}" \
  --finetuning_type "${FINETUNING_TYPE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_target "${LORA_TARGET}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir true \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --cutoff_len "${CUTOFF_LEN}" \
  --max_samples "${MAX_SAMPLES}" \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --bf16 "${BF16}" \
  --fp16 "${FP16}" \
  --plot_loss true
