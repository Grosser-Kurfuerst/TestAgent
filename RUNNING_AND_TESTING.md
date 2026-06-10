# my-Agent Running and Testing Guide

本文档说明如何启动 `my-Agent`，以及如何按不同层级测试它的效果。

所有命令默认在仓库根目录执行：

```bash
cd /home/kurfuerst/Coding/work/Coding-Agent/my-Agent
```

## 1. 安装依赖

只运行基础 Agent：

```bash
uv sync
```

运行单元测试：

```bash
uv sync --extra dev
```

运行 MBPP/HumanEval/SWE-bench 数据构建或 MBPP 评测：

```bash
uv sync --extra data --extra dev
```

说明：

- 基础依赖只包含 Agent 运行必需项。
- `dev` 提供 `pytest`。
- `data` 提供 HuggingFace `datasets`，用于真实 benchmark 数据。

## 2. 配置 LLM

运行 `run` 命令前必须有 `.env` 文件。可以从示例复制：

```bash
cp .env.example .env
```

### 2.1 本地 deterministic fake LLM

用于开发验证，不调用真实 API：

```bash
MY_AGENT_LLM_PROVIDER=fake
MY_AGENT_MAX_STEPS=8
MY_AGENT_COMMAND_TIMEOUT=60
MY_AGENT_TRACE_DIR=traces
```

检查配置：

```bash
uv run python run_agent.py config
```

确认输出里：

```text
"provider": "fake"
"use_fake_llm": true
```

### 2.2 真实 OpenAI-compatible API

用于真实 LLM 测试：

```bash
MY_AGENT_LLM_PROVIDER=openai
MY_AGENT_API_KEY=sk-your-api-key
MY_AGENT_BASE_URL=https://api.openai.com/v1
MY_AGENT_MODEL=gpt-4o-mini
MY_AGENT_TEMPERATURE=0.1
MY_AGENT_MAX_STEPS=8
MY_AGENT_COMMAND_TIMEOUT=60
MY_AGENT_TRACE_DIR=traces
```

如果使用本地 OpenAI-compatible 服务，例如 vLLM 或网关服务：

```bash
MY_AGENT_LLM_PROVIDER=openai
MY_AGENT_API_KEY=dummy
MY_AGENT_BASE_URL=http://127.0.0.1:8000/v1
MY_AGENT_MODEL=your-local-model-name
MY_AGENT_TEMPERATURE=0.1
MY_AGENT_MAX_STEPS=8
MY_AGENT_COMMAND_TIMEOUT=60
MY_AGENT_TRACE_DIR=traces
```

检查真实 API 配置：

```bash
uv run python run_agent.py config --check-api-key
```

确认输出里：

```text
"provider": "openai"
"use_fake_llm": false
"api_key_configured": true
```

## 3. 不调用 LLM 的上下文测试

这两类命令只验证索引和检索，不会调用 LLM，也不会改文件。

预览仓库上下文：

```bash
uv run python run_agent.py index \
  --repo examples/sample_repo \
  --query "subtract bug" \
  --top-k 3
```

单独运行词法检索：

```bash
uv run python run_agent.py retrieve \
  --repo examples/sample_repo \
  --query "subtract function" \
  --top-k 3
```

期望看到 `calculator.py` 和 `tests/test_calculator.py` 相关片段。

## 4. Agent Smoke Test

### 4.1 使用默认任务文件

默认任务在 `examples/tasks/sample_task.json`：

```bash
uv run python run_agent.py load-task
```

运行 Agent：

```bash
uv run python run_agent.py run --max-steps 8 --trace-dir traces
```

### 4.2 使用命令行覆盖任务

```bash
uv run python run_agent.py run \
  --repo examples/sample_repo \
  --task "Fix the subtract function so it returns the first number minus the second number." \
  --test-command "python -m unittest discover -s tests -q" \
  --max-steps 8 \
  --trace-dir traces
```

运行后会输出：

- `Plan`
- `Review`
- `Final summary`
- `Trace: ...jsonl`

注意：`run` 会让 Agent 调用工具，可能修改目标 repo。运行后用下面命令检查改动：

```bash
git diff -- examples/sample_repo
```

## 5. Trace 统计

查看一个 trace 文件：

```bash
uv run python run_agent.py stats --trace traces/agent_trace_xxx.jsonl
```

查看整个 trace 目录：

```bash
uv run python run_agent.py stats --trace traces
```

JSON 格式输出：

```bash
uv run python run_agent.py stats --trace traces --json
```

重点看这些指标：

- `Tool success rate`: 工具调用成功率。
- `Blocked tool calls`: 被安全 hook 拦截的工具调用。
- `Test pass rate`: `run_tests` 的通过率。
- `Edit count`: 成功编辑次数。
- `Tool distribution`: 工具调用分布。

## 6. 单元测试

安装 dev 依赖后运行：

```bash
uv sync --extra dev
uv run python -m unittest discover -s tests -q
```

如果要用 pytest：

```bash
uv sync --extra dev
uv run pytest -q
```

这个层级主要验证代码实现、hook 边界、配置解析、CLI 和 trace 统计，不评估真实 LLM 修复能力。

## 7. MBPP 真实评测

MBPP 评测会：

1. 从 HuggingFace 加载 MBPP 数据。
2. 为每条任务生成临时 Python repo。
3. 调用 Agent 进行修复。
4. 运行 pytest 判断是否通过。
5. 写入逐题结果和汇总评分。

准备依赖：

```bash
uv sync --extra data --extra dev
```

建议先跑 1 到 3 条，确认 API、网络和费用可控：

```bash
uv run python scripts/eval_mbpp.py \
  --limit 3 \
  --output-dir evaluationResults/mbpp_eval \
  --max-steps 10
```

默认情况下，LLM API 空响应、429/5xx、超时等 transient 错误会自动重试。重试耗尽后会记录为 `status=transient_error`，默认不计入解题能力评分分母。

跑更多任务：

```bash
uv run python scripts/eval_mbpp.py \
  --limit 10 \
  --output-dir evaluationResults/mbpp_eval \
  --max-steps 10
```

如果要评估端到端稳定性，把 API 抖动也计入失败分母：

```bash
uv run python scripts/eval_mbpp.py \
  --limit 10 \
  --output-dir evaluationResults/mbpp_eval_e2e \
  --max-steps 10 \
  --count-transient-errors
```

断点续跑：

```bash
uv run python scripts/eval_mbpp.py \
  --start 10 \
  --limit 10 \
  --output-dir evaluationResults/mbpp_eval \
  --max-steps 10
```

`--start 0` 会覆盖旧的 `results.jsonl`，避免新旧结果 schema 混合；`--start > 0` 会继续追加，适合断点续跑。

结果文件：

```text
evaluationResults/mbpp_eval/results.jsonl
```

每行包含：

- `task_id`
- `status`: `passed`、`failed`、`error` 或 `transient_error`
- `scored`: 是否计入解题能力评分分母
- `test_output`
- `agent_steps`
- `agent_done`
- `agent_stop_reason`
- `error`
- `elapsed_sec`
- `attempts`

汇总里同时输出：

- `Solve rate`: `passed / scored`，默认剔除 API transient 错误，适合看解题能力。
- `End-to-end rate`: `passed / total`，包含所有样本，适合看完整系统稳定性。

## 8. 数据构建测试

构建少量 MBPP 任务 repo 和 SFT 样本：

```bash
uv sync --extra data --extra dev
uv run python run_agent.py build-mbpp --limit 5 --output-dir /tmp/my_agent_data
```

构建 HumanEval：

```bash
uv run python run_agent.py build-humaneval --limit 5 --output-dir /tmp/my_agent_data
```

构建 SWE-bench Lite manifest：

```bash
uv run python run_agent.py build-swebench --limit 5 --output-dir /tmp/my_agent_data
```

把本地任务 manifest 转成 strategy SFT：

```bash
uv run python run_agent.py tasks-to-sft \
  --input /tmp/my_agent_data/tasks/mbpp_tasks.jsonl \
  --output /tmp/my_agent_data/sft/mbpp_strategy_sft.jsonl
```

把 trace 转成 SFT：

```bash
uv run python run_agent.py traces-to-sft \
  --input traces \
  --output /tmp/my_agent_data/sft/agent_traces_sft.jsonl
```

导出 LLaMA-Factory alpaca 格式：

```bash
uv run python run_agent.py export-alpaca \
  --inputs /tmp/my_agent_data/sft/mbpp_sft.jsonl /tmp/my_agent_data/sft/mbpp_strategy_sft.jsonl \
  --output-dir /tmp/my_agent_data/llamafactory
```

## 9. SFT 训练与协议评估

Phase 7 依赖 Phase 6 导出的 LLaMA-Factory 数据目录，目录中至少需要：

```text
dataset_info.json
train_alpaca.json
val_alpaca.json
dataset_stats.json
```

启动 LoRA SFT 训练：

```bash
DATASET_DIR=/tmp/my_agent_data/llamafactory \
OUTPUT_DIR=/tmp/my_agent_sft_lora \
BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
BATCH_SIZE=1 \
LEARNING_RATE=1e-4 \
NUM_TRAIN_EPOCHS=1 \
CUTOFF_LEN=4096 \
bash scripts/train_llamafactory_lora.sh
```

训练脚本会调用本机已有的 `llamafactory-cli`。如果没有安装 LLaMA-Factory，脚本会直接报错，不会静默跳过。

对比 base model 与 SFT adapter：

```bash
uv run python scripts/eval_sft_protocol.py \
  --val-data /tmp/my_agent_data/llamafactory/val_alpaca.json \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-dir /tmp/my_agent_sft_lora \
  --output-dir /tmp/my_agent_sft_eval \
  --max-new-tokens 512
```

如果只想验证指标计算逻辑，可以使用预生成响应文件，不加载真实模型：

```bash
uv run python scripts/eval_sft_protocol.py \
  --val-data /tmp/my_agent_data/llamafactory/val_alpaca.json \
  --base-responses /tmp/my_agent_sft_eval/base_responses.json \
  --sft-responses /tmp/my_agent_sft_eval/sft_responses.json \
  --output-dir /tmp/my_agent_sft_eval_metric_only
```

评估输出：

- `metrics_summary.json`：base、SFT 和 delta 的汇总指标。
- `detailed_results.json`：每条样本的任务类型、参考输出、base 输出、SFT 输出和单样本指标。
- `experiment_report.md`：自动生成的实验报告草稿。
- `base_responses.json` / `sft_responses.json`：用于复现指标的原始响应。

核心指标含义：

- `json_valid_rate`：输出是否为严格 JSON object。
- `field_hit_rate`：不同任务类型的必填字段是否完整。
- `tool_accuracy`：工具调用样本中工具名是否和参考输出一致。
- `file_mention_rate`：输出是否提到参考输出或输入里的关键文件路径。
- `rouge_l`：与参考输出的弱文本相似度。

报告模板在：

```text
templates/sft-experiment-report-template.md
```

注意：这些指标主要用于衡量结构化输出协议对齐，不直接证明复杂真实代码修复能力。端到端能力仍需要通过 agent 运行、diff 审查和测试结果单独验证。

## 10. 推荐测试顺序

第一次验证建议按这个顺序：

```bash
uv sync --extra data --extra dev
cp .env.example .env
```

先使用 fake LLM：

```bash
uv run python run_agent.py config
uv run python run_agent.py index --repo examples/sample_repo --query "subtract bug" --top-k 3
uv run python run_agent.py retrieve --repo examples/sample_repo --query "subtract function" --top-k 3
uv run python run_agent.py run --max-steps 8 --trace-dir traces
uv run python run_agent.py stats --trace traces
uv run python -m unittest discover -s tests -q
```

再切换真实 API：

```bash
uv run python run_agent.py config --check-api-key
uv run python run_agent.py run --max-steps 8 --trace-dir traces
uv run python scripts/eval_mbpp.py --limit 3 --output-dir /tmp/mbpp_eval --max-steps 10
```

## 11. 常见问题

### `Configuration file not found`

缺少 `my-Agent/.env`。执行：

```bash
cp .env.example .env
```

### `No API key configured`

真实 API 模式下没有配置 key。检查：

```bash
MY_AGENT_LLM_PROVIDER=openai
MY_AGENT_API_KEY=...
```

如果只是本地 smoke test，可以改成：

```bash
MY_AGENT_LLM_PROVIDER=fake
```

### `datasets` 找不到

运行：

```bash
uv sync --extra data
```

### `llamafactory-cli` 找不到

Phase 7 的训练脚本依赖外部安装的 LLaMA-Factory。先安装并确认命令可用：

```bash
llamafactory-cli version
```

也可以用 `LLAMAFACTORY_CMD` 指向自定义启动命令。

### `torch`、`transformers` 或 `peft` 找不到

`scripts/eval_sft_protocol.py` 只有在需要真实模型推理时才导入这些依赖。若只是检查指标，传入 `--base-responses` 和 `--sft-responses` 即可避免加载模型。

### `pytest` 找不到

运行：

```bash
uv sync --extra dev
```

### 测试命令被 hook 拦截

`run_tests` 只允许仓库范围内的安全测试命令。推荐使用：

```bash
python -m unittest discover -s tests -q
python -m pytest -q
pytest -q
```

不要在测试命令里使用管道、重定向、`curl`、`wget`、`rm -rf`、仓库外路径等。

### 真实 LLM 输出不是 JSON

Actor 阶段要求模型只输出一个 JSON 对象：

```json
{"tool": "read_file", "arguments": {"path": "solution.py"}, "reason": "Inspect the file before editing."}
```

如果模型连续输出自然语言，Agent 会记录 `invalid_tool_call` 并停止。可以换更强的指令遵循模型，或降低温度。
