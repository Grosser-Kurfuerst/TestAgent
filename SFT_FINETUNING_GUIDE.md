# SFT Warm-Start and OPD Integration Guide

本文档说明如何复用 AgentCli 原有的 legacy Alpaca SFT 流程，训练 Qwen3-4B LoRA，并将训练结果注册为 OPD 的初始检查点 M0。

## 1. 数据流

```text
task manifest / passed Agent traces
  -> legacy SFT JSONL
  -> export-alpaca --tool-calls-only
  -> data/llamafactory
  -> train_llamafactory_lora.sh
  -> trainer-output
  -> register_sft_checkpoint.py
  -> outputs/opd/M0
```

原始 SFT schema 保持不变：

```json
{
  "instruction": "根据用户任务、计划和已有工具轨迹，选择下一步工具调用。",
  "input": {
    "task": "修复测试失败",
    "plan": "",
    "history": []
  },
  "output": {
    "tool": "read_file",
    "arguments": {"path": "src/example.py"},
    "reason": "inspect"
  }
}
```

当前 runtime 会把顶层 `tool` 当作 `name` 的 legacy 别名，因此这个输出能够转换为真实 `CanonicalToolCall` 并执行。

## 2. 构建原有 SFT 数据

从可运行任务生成 strategy 样本：

```bash
UV_CACHE_DIR=/tmp/agentcli-uv-cache uv run my-agent tasks-to-sft \
  --input data/sft_raw/tasks/tasks.jsonl \
  --output data/sft_raw/sft/task_strategy_sft.jsonl
```

从已经通过 benchmark 的 trace 生成 next-tool-call 样本：

```bash
UV_CACHE_DIR=/tmp/agentcli-uv-cache uv run my-agent traces-to-sft \
  --input traces \
  --output data/sft_raw/sft/agent_traces_sft.jsonl
```

`traces-to-sft` 只转换包含 `benchmark_result.status=passed` 的 trace，并保留成功的非 `finish` 工具调用。

也可以继续使用 `build-mbpp`、`build-humaneval` 和 `swebench-to-sft` 生成原有 schema 的数据。

## 3. 导出 LLaMA-Factory Alpaca 数据

OPD warm-start 只需要工具调用样本，因此使用 `--tool-calls-only` 排除 strategy 和 repair-plan 输出：

```bash
UV_CACHE_DIR=/tmp/agentcli-uv-cache uv run my-agent export-alpaca \
  --inputs \
    data/sft_raw/sft/mbpp_sft.jsonl \
    data/sft_raw/sft/humaneval_sft.jsonl \
    data/sft_raw/sft/agent_traces_sft.jsonl \
  --output-dir data/llamafactory \
  --train-ratio 0.95 \
  --seed 42 \
  --tool-calls-only
```

输出：

```text
data/llamafactory/
  dataset_info.json
  train_alpaca.json
  val_alpaca.json
  dataset_stats.json
```

训练前检查：

```bash
jq '{total, train, val, skipped, filtered, tool_calls_only, source_counts}' \
  data/llamafactory/dataset_stats.json
```

`train` 和 `val` 都必须大于零。默认不传 `--tool-calls-only` 时，`export-alpaca` 仍保持原有行为。

## 4. 安装 LLaMA-Factory

使用独立 Python 3.11 环境安装 LLaMA-Factory 0.9.4，避免覆盖 AgentCli 的依赖：

```bash
git clone --depth 1 --branch v0.9.4 \
  https://github.com/hiyouga/LLaMA-Factory.git \
  ../LLaMA-Factory-v0.9.4

cd ../LLaMA-Factory-v0.9.4
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
llamafactory-cli version
```

版本输出必须包含 `0.9.4`。该 legacy Alpaca 流程不需要 AgentCli 自定义 ingestion patch。

## 5. 训练 Qwen3-4B LoRA

回到 AgentCli 仓库根目录后执行：

```bash
DATASET_DIR=data/llamafactory \
OUTPUT_DIR=outputs/coding_agent_lora \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_llamafactory_lora.sh
```

默认训练合同：

```text
base model: Qwen/Qwen3-4B-Instruct-2507
revision: cdbee75f17c01a7cc42f958dc650907174af0554
template: qwen3_nothink
LoRA: rank=16, alpha=32, dropout=0
targets: q_proj,k_proj,v_proj,o_proj
learning rate: 2e-5
cutoff length: 8192
batch size: 1
gradient accumulation: 16
BF16: true
```

脚本会拒绝与 OPD shared adapter 不一致的 LoRA 参数，也会拒绝空的 train/validation 数据或错误的 LLaMA-Factory 版本。

为保证 M0 身份可以追溯，训练脚本不允许使用 `LOCAL_MODEL_DIR` 或其他本地模型覆盖。训练完成后会额外写入：

```text
outputs/coding_agent_lora/sft_training_manifest.json
```

其中固定记录实际训练使用的 base model、immutable revision、tokenizer revision、`qwen3_nothink` 模板、LLaMA-Factory 版本以及 canonical adapter config/hash。

训练完成后，最终 adapter 必须直接位于：

```text
outputs/coding_agent_lora/
  adapter_config.json
  adapter_model.safetensors
  sft_training_manifest.json
```

中间 `checkpoint-*` 可以保留用于训练审计，但注册脚本不会自动选择它们。

## 6. Protocol 评估

```bash
UV_CACHE_DIR=/tmp/agentcli-uv-cache uv run --extra opd-train python \
  scripts/eval_sft_protocol.py \
  --val-data data/llamafactory/val_alpaca.json \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-dir outputs/coding_agent_lora \
  --output-dir outputs/sft_protocol_eval
```

除原有 JSON、字段和工具准确率外，报告还包含：

```text
runtime_tool_call_parse_rate
```

该指标表示生成结果能否被当前 AgentCli runtime 解析为至少一个真实工具调用。

## 7. 注册为 M0

```bash
UV_CACHE_DIR=/tmp/agentcli-uv-cache uv run --extra opd-train python \
  scripts/register_sft_checkpoint.py \
  --trainer-output outputs/coding_agent_lora \
  --output outputs/opd/M0 \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --base-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --opd-config configs/opd_paper_train.yaml
```

注册步骤会：

- 校验 trainer-output 根目录中的最终 adapter；
- 校验 `sft_training_manifest.json` 与注册参数、PEFT adapter 和 OPD 配置一致；
- 校验 adapter 与 OPD shared adapter 合同一致；
- 使用 `TransformersPolicy` 加载 base model 和 adapter；
- 复制干净的 deployable adapter；
- 写入 `policy_identity_manifest.json`。

输出：

```text
outputs/opd/M0/
  adapter/
    adapter_config.json
    adapter_model.safetensors
  sft_training_manifest.json
  policy_identity_manifest.json
```

## 8. 从 M0 进入 OPD

OPD round 0 使用：

```text
checkpoint: outputs/opd/M0/adapter
identity manifest: outputs/opd/M0/policy_identity_manifest.json
```

示例：

```bash
UV_CACHE_DIR=/tmp/agentcli-uv-cache uv run --extra opd-train python \
  scripts/train_opd_evolver.py \
  --config configs/opd_paper_train.yaml \
  --checkpoint outputs/opd/M0/adapter \
  --identity-manifest outputs/opd/M0/policy_identity_manifest.json \
  --learner-dataset outputs/opd/round-0/learner_dataset.jsonl \
  --export-manifest outputs/opd/round-0/export_manifest.json \
  --output-dir outputs/opd/M1
```

OPD trainer 会继续训练 M0 中唯一的 LoRA adapter，不会创建第二个 adapter。

`data/` 和 `outputs/` 均为运行产物目录，不应提交到 Git。
