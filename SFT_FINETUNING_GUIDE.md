# SFT Fine-Tuning Guide

本文档说明如何在 `my-Agent` 中准备 SFT 数据、使用 LLaMA-Factory 进行 LoRA SFT 微调，以及如何使用微调后的模型进行测试。

所有命令默认在仓库根目录执行：

```bash
cd /home/kurfuerst/Coding/work/Coding-Agent/my-Agent
```

## 1. 总体流程

完整链路是：

```text
原始任务 / Agent trace
  -> SFT JSONL
  -> LLaMA-Factory Alpaca 数据
  -> LoRA SFT 训练
  -> protocol 评估
  -> 可选：MBPP 端到端评估
```

其中：

- SFT JSONL 是本项目内部的统一监督数据格式。
- Alpaca 数据是 LLaMA-Factory 训练需要的格式。
- LoRA 训练产物是 adapter checkpoint，不是完整模型。
- `eval_sft_protocol.py` 测的是模型输出协议质量。
- `eval_mbpp.py` 测的是完整 coding agent 在 MBPP repo 上修代码并跑测试的端到端效果。

## 2. 准备依赖

安装项目数据和开发依赖：

```bash
uv sync --extra data --extra dev
```

真实 LoRA 训练还需要安装 LLaMA-Factory，并确保命令可用：

```bash
llamafactory-cli version
```

如果你的命令不是 `llamafactory-cli`，训练时可以用 `LLAMAFACTORY_CMD` 指定。

真实模型推理评估还需要 `torch`、`transformers` 和 `peft`。这些依赖不在当前 `pyproject.toml` 的基础依赖中，需要在你的训练/推理环境中单独安装。

## 3. 构建 SFT JSONL 数据

### 3.1 从 MBPP 构建

```bash
uv run python run_agent.py build-mbpp \
  --limit 500 \
  --output-dir data/sft_raw
```

输出包括：

```text
data/sft_raw/
  repos/mbpp/
  tasks/mbpp_tasks.jsonl
  sft/mbpp_sft.jsonl
```

`mbpp_sft.jsonl` 主要是 `write_file` 风格样本，用于让模型学习根据任务和测试生成 `solution.py`。

### 3.2 从 HumanEval 构建

```bash
uv run python run_agent.py build-humaneval \
  --limit 100 \
  --output-dir data/sft_raw
```

输出包括：

```text
data/sft_raw/
  repos/humaneval/
  tasks/humaneval_tasks.jsonl
  sft/humaneval_sft.jsonl
```

### 3.3 从任务 manifest 构建 strategy SFT

如果已经有任务 manifest，例如 `mbpp_tasks.jsonl`，可以转成 strategy 样本：

```bash
uv run python run_agent.py tasks-to-sft \
  --input data/sft_raw/tasks/mbpp_tasks.jsonl \
  --output data/sft_raw/sft/mbpp_strategy_sft.jsonl
```

这类样本用于让模型学习如何选择下一步策略，而不是直接生成完整代码。

### 3.4 从 Agent trace 构建 SFT

把已有 agent 运行 trace 转成 SFT 样本：

```bash
uv run python run_agent.py traces-to-sft \
  --input traces \
  --output data/sft_raw/sft/agent_traces_sft.jsonl
```

trace 转换会保留成功的非 `finish` 工具调用，用于让模型学习真实 agent 轨迹中的工具调用格式和上下文。

## 4. 导出 LLaMA-Factory Alpaca 数据

把一个或多个 SFT JSONL 文件导出为 LLaMA-Factory 可训练格式：

```bash
uv run python run_agent.py export-alpaca \
  --inputs \
    data/sft_raw/sft/mbpp_sft.jsonl \
    data/sft_raw/sft/humaneval_sft.jsonl \
    data/sft_raw/sft/mbpp_strategy_sft.jsonl \
    data/sft_raw/sft/agent_traces_sft.jsonl \
  --output-dir data/llamafactory \
  --train-ratio 0.95
```

导出后目录应包含：

```text
data/llamafactory/
  dataset_info.json
  train_alpaca.json
  val_alpaca.json
  dataset_stats.json
```

训练前建议检查 `dataset_stats.json`：

```bash
cat data/llamafactory/dataset_stats.json
```

重点看：

- `total`：总样本数。
- `train` / `val`：训练集和验证集数量。
- `skipped`：被跳过的坏样本数量。
- `source_counts`：各数据来源的样本数量。

## 5. 启动 LoRA SFT 训练

使用默认配置训练：

```bash
DATASET_DIR=data/llamafactory \
OUTPUT_DIR=outputs/coding_agent_lora \
BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
LEARNING_RATE=1e-4 \
NUM_TRAIN_EPOCHS=1 \
CUTOFF_LEN=4096 \
bash scripts/train_llamafactory_lora.sh
```

如果 base model 已经下载到本地：

```bash
LOCAL_MODEL_DIR=/path/to/Qwen2.5-Coder-7B-Instruct \
DATASET_DIR=data/llamafactory \
OUTPUT_DIR=outputs/coding_agent_lora \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_llamafactory_lora.sh
```

常用可调参数：

```bash
LORA_RANK=8
LORA_ALPHA=32
LORA_TARGET=all
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=1e-4
NUM_TRAIN_EPOCHS=1
CUTOFF_LEN=4096
SAVE_STEPS=50
EVAL_STEPS=50
BF16=true
FP16=false
```

训练脚本实际调用的是：

```text
llamafactory-cli train
```

并使用：

```text
--stage sft
--finetuning_type lora
--template qwen
--dataset coding_agent_train
--eval_dataset coding_agent_val
```

训练完成后，LoRA adapter 会在：

```text
outputs/coding_agent_lora/
```

常见结构：

```text
outputs/coding_agent_lora/
  adapter_config.json
  adapter_model.safetensors
  checkpoint-50/
  checkpoint-100/
  trainer_state.json
```

如果存在多个 `checkpoint-*`，当前评估代码会自动选择编号最大的 checkpoint。

## 6. 使用微调模型做 protocol 评估

这个评估比较 base model 和 LoRA adapter 在验证集上的输出协议质量。

```bash
uv run python scripts/eval_sft_protocol.py \
  --val-data data/llamafactory/val_alpaca.json \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-dir outputs/coding_agent_lora \
  --output-dir outputs/sft_protocol_eval \
  --limit 100 \
  --max-new-tokens 512 \
  --device auto \
  --dtype bfloat16
```

如果 base model 在本地：

```bash
uv run python scripts/eval_sft_protocol.py \
  --val-data data/llamafactory/val_alpaca.json \
  --base-model /path/to/Qwen2.5-Coder-7B-Instruct \
  --adapter-dir outputs/coding_agent_lora \
  --output-dir outputs/sft_protocol_eval \
  --limit 100 \
  --max-new-tokens 512
```

输出目录：

```text
outputs/sft_protocol_eval/
  base_responses.json
  sft_responses.json
  metrics_summary.json
  detailed_results.json
  experiment_report.md
```

核心指标：

- `json_valid_rate`：输出是否是严格 JSON object。
- `field_hit_rate`：必填字段是否完整。
- `tool_accuracy`：工具调用样本中工具名是否正确。
- `file_mention_rate`：输出是否提到关键文件路径。
- `rouge_l`：与参考输出的弱文本相似度。

这个评估只说明模型是否更会遵守 agent 输出协议，不直接证明模型具备复杂真实代码修复能力。

## 7. 使用响应文件做指标回放

如果已经有 `base_responses.json` 和 `sft_responses.json`，可以只跑指标计算，不加载模型：

```bash
uv run python scripts/eval_sft_protocol.py \
  --val-data data/llamafactory/val_alpaca.json \
  --base-responses outputs/sft_protocol_eval/base_responses.json \
  --sft-responses outputs/sft_protocol_eval/sft_responses.json \
  --output-dir outputs/sft_protocol_eval_replay
```

这适合：

- 快速验证指标逻辑。
- 在无 GPU 环境下复算结果。
- 对同一批响应生成不同报告。

## 8. 使用微调模型跑 MBPP 端到端测试

`eval_sft_protocol.py` 直接加载 base model + LoRA adapter。

但 `scripts/eval_mbpp.py` 走的是 agent runtime，runtime 通过 OpenAI-compatible API 调模型。因此，想用微调模型跑 MBPP，需要先把模型服务成 OpenAI-compatible API。

常见方式：

1. 用 LLaMA-Factory / vLLM / 其他推理服务加载 base model + LoRA adapter。
2. 或先把 LoRA adapter merge 到 base model，再服务 merged model。

服务启动后，配置 `.env`：

```bash
MY_AGENT_LLM_PROVIDER=openai
MY_AGENT_API_KEY=dummy
MY_AGENT_BASE_URL=http://127.0.0.1:8000/v1
MY_AGENT_MODEL=your-sft-served-model-name
MY_AGENT_TEMPERATURE=0.1
MY_AGENT_MAX_STEPS=10
MY_AGENT_COMMAND_TIMEOUT=60
MY_AGENT_TRACE_DIR=traces
```

检查配置：

```bash
uv run python run_agent.py config --check-api-key
```

跑少量 MBPP：

```bash
uv run python scripts/eval_mbpp.py \
  --limit 10 \
  --output-dir evaluationResults/mbpp_sft_eval_run1 \
  --max-steps 10
```

跑全量 MBPP：

```bash
uv run python scripts/eval_mbpp.py \
  --limit 500 \
  --output-dir evaluationResults/mbpp_sft_eval_full \
  --max-steps 10
```

建议每次评估使用新的 `--output-dir`，避免 `results.jsonl` 追加到旧结果里。

## 9. 推荐实验顺序

第一次跑通建议：

1. `build-mbpp --limit 20` 和 `build-humaneval --limit 20`，先构建小数据。
2. `export-alpaca`，确认 `dataset_stats.json` 正常。
3. 用 `NUM_TRAIN_EPOCHS=1` 跑一次小规模 LoRA。
4. 跑 `eval_sft_protocol.py --limit 50`，检查协议指标。
5. 如果协议指标有提升，再服务 LoRA 模型。
6. 跑 `eval_mbpp.py --limit 10`，检查端到端通过率。
7. 最后再扩大样本数、训练 epoch 和 MBPP 评测规模。

## 10. 常见问题

### `llamafactory-cli` 找不到

确认 LLaMA-Factory 是否安装，并且当前 shell 能找到命令：

```bash
which llamafactory-cli
llamafactory-cli version
```

如果命令路径不同：

```bash
LLAMAFACTORY_CMD=/path/to/llamafactory-cli \
bash scripts/train_llamafactory_lora.sh
```

### 缺少 `dataset_info.json`

说明还没有执行 `export-alpaca`，或者 `DATASET_DIR` 指错了。

检查：

```bash
ls data/llamafactory
```

重新导出：

```bash
uv run python run_agent.py export-alpaca \
  --inputs data/sft_raw/sft/mbpp_sft.jsonl \
  --output-dir data/llamafactory
```

### 显存不够

优先降低：

```bash
BATCH_SIZE=1
CUTOFF_LEN=2048
```

也可以增大：

```bash
GRADIENT_ACCUMULATION_STEPS=32
```

如果仍然不够，换更小的 base model 或使用量化训练方案。

### Protocol 指标提升但 MBPP 不提升

这通常说明模型更会输出格式，但端到端修题能力仍不足。优先检查：

- `traces/` 中是否还有大量 `invalid_tool_call`。
- 是否经常没有读取测试文件就编辑。
- 是否经常使用大段 `write_file` 生成非法 JSON。
- 是否工具 description 和 actor prompt 足够明确。
- MBPP 失败是协议问题、语义问题还是 API transient error。

### MBPP 端到端评测调用的不是微调模型

确认 `.env` 中：

```bash
MY_AGENT_BASE_URL=http://127.0.0.1:8000/v1
MY_AGENT_MODEL=your-sft-served-model-name
MY_AGENT_LLM_PROVIDER=openai
MY_AGENT_USE_FAKE_LLM=false
```

然后运行：

```bash
uv run python run_agent.py config --check-api-key
```

确认输出里 `use_fake_llm` 是 `false`，并且 `model` 是你服务出来的微调模型名。
