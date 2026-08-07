# AgentCli AI Agent 高价值面试指南：通用精选与 Memory 专题

> 适用范围：AI Agent、Coding Agent、Agent Runtime、RAG、Tool Calling、Memory Evolver、SFT/OPD 相关面试。
> 代码基线：当前 AgentCli 仓库实现。
> 文档结构：第 0 节从合并后的 73 道公开面经题及项目追问中精选 30 题详解；第 1～6 节是 13 道 AgentCli Memory 项目深挖题；第 7 节根据简历中的 CODINCLI 项目补充 16 道 HITL、上下文管理、SFT 与 OPD 定制追问。
> 真实性边界：公开面经属于候选人自述，不是公司官方题库；代码答案严格区分已实现、部分实现和建议，不虚构个人贡献、上线状态或效果指标。

## 0. 从 73 题中精选并扩展的 30 道高价值问题

### 0.1 信源说明

| 编号 | 信源 | 证据等级 |
|---|---|---|
| N1 | 牛客《29届第四范式 Agent 开发实习面经》 | B：当前研究记录保留原帖题名，未保留稳定链接 |
| N2 | [牛客《大模型 Agent 校招面经——阿里淘天》](https://www.nowcoder.com/discuss/911218249889480704) | A：候选人原帖 |
| N3 | 牛客《美团大模型算法岗完整面经》 | B：当前研究记录保留原帖题名，未保留稳定链接 |
| N4 | [牛客《字节跳动 Agent 开发一面》](https://www.nowcoder.com/feed/main/detail/2879521) | A：候选人原帖 |
| N5 | [牛客《腾讯 AI Agent 应用开发一面凉经》](https://www.nowcoder.com/feed/main/detail/2854750) | A：候选人原帖 |
| N6 | 牛客《字节 AI Agent 算法一面》 | B：当前研究记录保留原帖题名，未保留稳定链接 |
| X1 | Zero2PM《小红书 AI 产品实习面试真题》等小红书相关二次复盘 | B：二次整理，不作为候选人原帖 |
| X2 | 二次整理的小红书 Agent 开发面经 | B：二次整理，原始候选人帖子未核验 |
| X3 | [小红书《淘天 AI Agent 三轮面经》](https://www.xiaohongshu.com/search_result/69aedd14000000001a035aad?xsec_token=AByLOhiFGN6uz355J-mbVtaXZkF0yJfBaXFDDBIup5JgQ=) | A：采集时读取到正文；签名链接可能过期 |
| N11 | [牛客《百度 AI Agent 开发一面面经》](https://www.nowcoder.com/feed/main/detail/fb9d83f0b6cd44b1ac3e3cafed6a1cf0) | B：搜索索引摘要 |
| N12 | [牛客《智谱 AI Agent 产品实习一面》](https://www.nowcoder.com/feed/main/detail/5d0657578dbb479caa3547c95e4a937b) | B：搜索索引摘要 |

选择标准：优先保留能够同时考察原理、系统设计、生产可靠性和项目实现边界的问题；不选择仅靠背定义即可回答、与 AgentCli 完全无关，或与后文 Memory 专题高度重复的题目。

---

### 精选 1：Workflow 和 Agent 的核心区别是什么？分别适合什么场景？

> 公司：美团、阿里淘天｜信源：N3、X3｜原题编号：3
> AgentCli 状态：已实现 ReAct、Plan-and-Execute、Team 三种运行模式。

Workflow 的执行图、分支和异常处理主要由开发者预先确定，优点是可预测、易测试、易审计；Agent 则让模型根据当前 Observation 动态选择下一步，适合步骤无法提前穷举的开放任务，但不确定性、成本和风险更高。

AgentCli 中：

- ReAct 最接近动态 Agent Loop，模型每轮决定回答还是调用工具。
- Plan 模式先让 Planner 生成 DAG，再由执行器按依赖运行，更像“动态生成的 Workflow”。
- Team 模式在计划与执行之外加入 Worker 和 Reviewer，适合可拆分且需要复核的任务。

选择时看三个条件：路径是否稳定、错误是否可恢复、结果能否确定性验证。稳定审批流优先 Workflow；代码修复、资料研究等开放任务可以使用 Agent；高风险动作应由 Workflow/HITL 包住 Agent，而不是把全部控制权交给模型。

代码：[运行模式工厂](src/my_agent/runtime/factory.py#L40)、[ReAct Loop](src/my_agent/react/agent.py#L168)、[任务图验证](src/my_agent/plan/graph.py#L23)

### 精选 2：Modular Agent 如何实现多步规划？

> 公司：阿里淘天｜信源：N2｜原题编号：2
> AgentCli 状态：已实现结构化计划、DAG 校验和依赖调度。

核心不是让模型输出一段自然语言计划，而是输出可验证的结构化任务契约。AgentCli Planner 要求 JSON 中包含任务描述、类型、依赖和验收条件；随后统一规范化任务 ID，并检查重复 ID、未知依赖、自依赖、环和任务数量上限。

执行器对 DAG 做拓扑分批：同一批内没有依赖冲突的任务可以并行，前置任务失败时后继任务会被跳过。每个任务由独立 Runner 执行并返回 `ok/output/trace_path/stop_reason`，使计划层只依赖统一结果契约。

比较完整的生产设计还应加入：计划版本、重规划触发条件、每步输入输出 Schema、失败传播策略和幂等边界。当前 Plan 链路有状态保存和批次超时，但自动重规划能力仍有限。

代码：[Planner](src/my_agent/plan/planner.py#L59)、[TaskGraph](src/my_agent/plan/graph.py#L23)、[PlanExecutor](src/my_agent/plan/executor.py#L74)

### 精选 3：单 Agent 和多 Agent 应该如何选择？

> 公司：字节跳动、百度｜信源：N4、N11｜原题编号：4
> AgentCli 状态：已实现 Team 的 Planner、Worker、Reviewer 编排。

任务短、步骤强依赖、需要共享大量隐含上下文时优先单 Agent。任务可以拆成独立子问题、存在明显并行机会、需要角色隔离或独立复核时，再使用多 Agent。

多 Agent 的收益来自分工和上下文隔离，不是 Agent 数量本身。AgentCli 的 Team 模式由 Planner 生成步骤，Worker 执行，Reviewer 按验收条件判断是否通过；依赖满足的步骤可以并行。代价是多次 LLM 调用、上下文复制、结论冲突和更复杂的失败恢复。

实践中可以用一个简单阈值：如果拆分后的每个子任务都有独立输入、独立产物和确定性验收，就适合多 Agent；如果子任务间每一步都要共享完整思考过程，拆分通常得不偿失。

代码：[TeamAgent](src/my_agent/team/agent.py#L55)、[SubAgentRunner](src/my_agent/team/sub_agent.py#L74)、[Team 并行执行](src/my_agent/team/agent.py#L506)

### 精选 4：请设计一个完整的 AI Agent 系统架构。

> 公司：阿里淘天｜信源：X3｜原题编号：11
> AgentCli 状态：单机主链路基本具备，分布式与高可用能力不足。

可以按八层回答：

1. 接入层：鉴权、会话、限流和任务受理。
2. 上下文层：任务、仓库、规则、历史、Memory 和 Token Budget。
3. 规划层：ReAct、Plan 或 Team 路由，生成步骤和验收条件。
4. 执行编排层：依赖调度、并发、超时、取消和状态流转。
5. Tool/MCP 层：Schema、注册、权限、执行和结果标准化。
6. 安全层：静态策略、Sandbox、HITL、敏感信息和副作用控制。
7. 状态与恢复层：事件日志、Checkpoint、幂等键和恢复确认。
8. 可观测评测层：Trace、指标、Bad Case、回放和版本对照。

AgentCli 已覆盖单机运行时、三种 Agent 模式、Tool/MCP、Memory、HITL 和 Trace。主要缺口是持久化任务队列、跨进程 Resume、多 Provider 故障转移、租户隔离、完整 Sandbox 和外部副作用事务化。

代码：[AgentBase](src/my_agent/runtime/base.py#L65)、[MemoryManager](src/my_agent/memory/manager.py#L50)、[TraceWriter](src/my_agent/observability/tracing.py#L11)、[HITL Policy](src/my_agent/hitl/policy.py#L40)

### 精选 5：Agent 如何自动拆分任务？任务粒度如何确定？

> 公司：阿里淘天、百度｜信源：X3、N11｜原题编号：14
> AgentCli 状态：已实现任务 Schema 和 DAG 校验，粒度主要由 Prompt 约束。

拆分后的每一步应满足四个条件：单一目标、显式依赖、有限执行预算和独立验收条件。粒度过粗会让 Worker 在一个步骤里重新做规划；粒度过细则会增加调用次数、上下文传递和状态同步成本。

AgentCli Planner 输出 `title/description/type/depends_on/acceptance/max_steps` 等字段，TaskGraph 负责结构正确性，但“是否拆得合理”仍主要依赖模型。可进一步增加确定性规则：限制每步修改范围、要求产物路径、检查验收条件是否可执行，并根据历史 Trace 统计步骤失败率和平均成本。

面试中可以给出判断标准：一个步骤应能由一个 Worker 在固定工具轮数内完成，产物能由 Reviewer 不依赖 Worker 隐含思路独立验证。

代码：[计划输出解析](src/my_agent/plan/planner.py#L59)、[PlanTask](src/my_agent/plan/types.py#L15)、[Team Planner](src/my_agent/team/planner.py#L72)

### 精选 6：多个工具调用链路如何调度？是否有异常回退？

> 公司：阿里淘天｜信源：N2｜原题编号：5
> AgentCli 状态：已实现并发安全分组、超时、取消和结构化错误；业务级回退不足。

调度前先根据工具风险、资源和依赖关系分组。AgentCli 会并行执行允许并发的只读工具；写工具只有显式声明 `parallel_side_effect_safe` 且能够解析资源时才可能并行；无法判断或存在副作用的调用形成串行屏障。

每个结果统一返回错误码、是否超时、是否可重试等字段。批次有最大并发数、总超时和取消信号，未知工具、参数错误、策略拒绝和执行异常会转为 Observation 交给模型。

当前回退主要是“错误反馈给模型后重新决策”，缺少工具级替代路由、Retry Budget、指数退避和熔断。生产环境应把认证错误、参数错误、限流、瞬时网络错误、超时未知状态和业务拒绝分开处理。

代码：[工具批量执行](src/my_agent/tools/registry.py#L92)、[并发策略](src/my_agent/tools/parallel_policy.py#L21)、[结构化执行结果](src/my_agent/tools/execution.py#L47)

### 精选 7：Agent 陷入工具调用死循环时，如何检测和终止？

> 公司：腾讯、阿里淘天｜信源：N5、X3｜原题编号：40
> AgentCli 状态：已实现预算与重复失败终止，语义循环检测不足。

AgentCli 通过最大步骤数、最大 Tool Call 数、墙钟时间和重复失败窗口限制循环。如果最近若干次失败具有相同工具和错误特征，Budget 会返回 `repeated_tool_failure`；达到工具或步骤上限则停止，并把 `stop_reason` 写入 Trace。

这能处理“同一个调用连续失败”，但不能完整识别参数略有变化、实际没有新信息的语义循环。建议对每次决策计算规范化签名：`tool_name + normalized_args + relevant_state_hash`，结合 Observation 信息增益判断是否真正前进。达到阈值后先要求模型总结阻塞原因，再选择替代工具、请求人工输入或返回部分结果。

代码：[RuntimeBudget](src/my_agent/runtime/budget.py#L14)、[重复失败检查](src/my_agent/runtime/budget.py#L68)、[ReAct 停止处理](src/my_agent/react/agent.py#L504)

### 精选 8：如何设计 Agent 的 Tool Registry？

> 公司：阿里淘天｜信源：X3｜原题编号：44
> AgentCli 状态：已实现统一注册、Schema、风险和执行边界。

Tool Registry 至少要保存：稳定名称、用途描述、JSON Schema、来源、风险等级、超时、执行器和并发/资源信息。注册时校验 Schema，执行时再次解析参数，不能只相信模型输出。

AgentCli 的 `ToolSpec` 负责名称、描述、参数 Schema、风险、来源和超时；`ToolRegistration` 再绑定 Handler、资源解析器和并发安全标记。Registry 统一处理未知工具、参数校验、策略拒绝、执行异常和 Trace 事件。

进一步建议加入工具版本、Owner、SLA、权限 Scope、幂等能力、补偿动作和替代工具列表。工具描述应突出“何时用、何时不用、必需参数、返回语义”，避免多个工具只有名称不同、描述高度重叠。

代码：[ToolSpec](src/my_agent/tools/spec.py#L34)、[Tool 注册](src/my_agent/tools/registry.py#L45)、[工具执行边界](src/my_agent/tools/registry.py#L161)

### 精选 9：Agent State 应该如何管理？

> 公司：阿里淘天｜信源：X3｜原题编号：35
> AgentCli 状态：已实现内存状态、计划状态与 Trace，缺少统一持久化状态机。

State 不应只是聊天记录，而应至少包含：原始目标、约束、当前计划、步骤状态、工具结果、外部资源 ID、预算消耗、审批、错误、重试、环境版本和最终结果。

AgentCli 的 `AgentState` 保存任务、运行 ID、仓库上下文、计划、工具历史、步数、Trace、结束原因和元数据；Plan/Team 还有更细的任务状态。Trace 记录状态变化的事件证据。

不足是这些状态没有被统一成可跨进程恢复的事件溯源模型。建议区分三类数据：Prompt Context 只保留当前推理需要的信息；Durable State 保存恢复所需事实；Trace 保存完整审计证据。三者不能混成一个无限增长的消息列表。

代码：[AgentState](src/my_agent/schema.py#L82)、[PlanState](src/my_agent/plan/types.py#L135)、[TraceEvent](src/my_agent/schema.py#L131)

### 精选 10：Agent 如何实现 Checkpoint 和 Resume？

> 公司：阿里淘天｜信源：X3｜原题编号：36
> AgentCli 状态：部分实现；有状态文件，但没有完整任务恢复协议。

真正的 Checkpoint 必须保存恢复后继续执行所需的最小充分状态：目标与约束、已完成步骤、未完成计划、工具结果、文件/进程状态、外部资源 ID、审批记录、预算和代码/环境版本。恢复时还要确认上次超时操作究竟是失败、成功还是 `UNKNOWN`，否则直接重试可能产生重复副作用。

AgentCli 的 Plan/Team Store 会持久化计划或团队状态，Trace 也保存事件；Memory Store 有较强的事务恢复。但主 Agent 没有公开的 Resume 入口，也没有把文件快照、外部资源和幂等键统一纳入 Checkpoint，因此不能把当前能力描述成完整的跨进程恢复。

建议采用“事件日志 + 周期快照”：每个副作用步骤分配 `idempotency_key`，提交前记录 Intent，完成后记录 Result；恢复时先查询或验证外部状态，再决定继续、补偿或人工确认。

代码：[JsonPlanStore](src/my_agent/plan/store.py#L31)、[Team Store](src/my_agent/team/store.py#L31)、[TraceWriter](src/my_agent/observability/tracing.py#L11)

### 精选 11：什么是 RAG？如何评价一个 RAG 系统是否有效？

> 公司：阿里淘天｜信源：N2、X3｜原题编号：17
> AgentCli 状态：部分实现仓库检索和经验检索，不是通用知识库 RAG。

RAG 把问题拆成检索和生成两段：先从外部知识中召回证据，再让模型基于证据回答。评测也必须分层：

- 检索层：Recall@K、MRR/NDCG、命中文档率、重复率。
- 上下文层：证据覆盖率、噪声率、Token 占用。
- 生成层：事实正确率、引用一致率、拒答正确率。
- 任务层：最终成功率、测试通过率、延迟和成本。

AgentCli 的仓库检索主要是词法打分，长期经验支持词法或 Embedding Cosine。它可以作为 RAG 基础，但缺少通用文档解析、向量数据库、混合召回和标准 Rerank。

代码：[仓库索引](src/my_agent/repo/indexer.py#L75)、[长期经验词法检索](src/my_agent/memory/experience/retrieval/lexical.py#L99)、[Embedding 检索](src/my_agent/memory/experience/retrieval/embedding.py#L74)

### 精选 12：混合检索的完整流程是什么？召回后为什么需要 Rerank？

> 公司：字节跳动、阿里淘天｜信源：N6、X3｜原题编号：20
> AgentCli 状态：有两种独立 Retriever，尚未形成通用 Hybrid RAG。

推荐流程是：查询规范化或改写 → 元数据过滤 → BM25/词法与向量并行召回 → 去重 → 分数归一化或 RRF 融合 → Cross-Encoder/LLM Rerank → 按 Token Budget 选择上下文。

词法检索擅长代码符号、错误码和专有名词，向量检索擅长语义近似。第一阶段目标是高召回，会带来噪声；Rerank 使用更强的查询—文档联合建模修正排序，因此通常更准但更慢。

AgentCli 当前 Legacy 和 Formal 分别使用词法与 Embedding，尚未把两路候选统一融合。建议先实现 RRF，避免直接比较不可校准的原始分数；随后在固定验证集上调 TopK 和 Rerank 数量。

代码：[Lexical Retriever](src/my_agent/memory/experience/retrieval/lexical.py#L99)、[Embedding Retriever](src/my_agent/memory/experience/retrieval/embedding.py#L74)、[Selection Service](src/my_agent/memory/evolver/selection/service.py#L41)

### 精选 13：Agent 的评估体系包含哪些维度？

> 公司：阿里淘天｜信源：N2｜原题编号：46
> AgentCli 状态：已有 Trace/协议指标和任务评测，尚非完整线上评测平台。

建议分五层评估：

1. Outcome：任务成功率、测试通过率、最终答案正确性。
2. Decision：计划合法率、工具选择准确率、参数正确率、停止决策。
3. Reliability：超时、重复失败、恢复正确率、重复副作用率。
4. Safety：越权调用、审批触发与拒绝、敏感信息泄露。
5. Efficiency：Token、LLM/Tool Call 数、端到端延迟和成本。

AgentCli 已能从 Trace 汇总工具成功率、阻塞调用、编辑/测试结果和停止原因；Formal Role Protocol 还能评估 Selection、Action、Writing、Maintenance 的决策成功率、工具解析率和未知引用率。

不足是规划质量、幻觉率、恢复正确率和副作用重复率尚未形成统一指标。LLM-as-Judge 只能辅助语义评分，关键结果仍应由测试、Schema、文件 Diff 和权限策略确定性验证。

代码：[Trace Metrics](src/my_agent/observability/trace_metrics.py#L112)、[Protocol Metrics](src/my_agent/evaluation/protocol_metrics.py#L137)、[Formal Role Protocol](src/my_agent/evaluation/formal_role_protocol.py#L27)

### 精选 14：Agent 回复准确率突然下降，应该如何分层排查？

> 公司：小红书｜信源：X1｜原题编号：48
> AgentCli 状态：具备 Trace 和离线指标基础，自动根因定位与确定性回放不足。

先确认指标口径和数据分布是否变化，再按链路分层：

```text
输入/任务分布
→ Prompt 与上下文裁剪
→ Memory/RAG 召回
→ 模型与采样参数
→ Tool Schema/参数/权限
→ 外部服务和环境
→ Evaluator 与阈值
```

每个 Bad Case 应绑定 `run_id`，比较正常与异常版本的模型身份、Prompt/Tool Schema 哈希、检索候选、工具输入输出、错误码、审批和最终 Diff。AgentCli 的 JSONL Trace 与聚合指标可以完成事后定位，但尚缺一键回放、版本 Diff 和自动聚类。

建议建立 Bad Case 生命周期：自动采集 → 失败分类 → 相似案例聚类 → 最小复现 → 修复 → 固定为回归集。不要直接凭几个失败样本改 Prompt，否则容易修复局部、破坏整体。

代码：[TraceWriter](src/my_agent/observability/tracing.py#L11)、[Trace 聚合](src/my_agent/observability/trace_metrics.py#L112)、[Runtime Trace 绑定](src/my_agent/runtime/base.py#L113)

### 精选 15：三个工具且请求频率很高，如何降低整体延迟？

> 公司：阿里淘天｜信源：N2｜原题编号：49
> AgentCli 状态：已实现安全并行和批次超时，缓存与模型路由不足。

先拆延迟：模型首轮、工具排队、工具执行、结果序列化、模型二轮。优化顺序通常是：

1. 无依赖且无冲突的工具并行调用。
2. 缓存确定性只读结果，并设计版本化 Cache Key。
3. 合并同源请求、复用连接和批量查询。
4. 裁剪 Tool Result，避免下一轮 LLM 输入过长。
5. 简单路由/抽取使用更小模型，复杂推理再升级模型。
6. 设置每阶段 Timeout、总 Deadline 和部分结果降级。

AgentCli 已根据风险和资源冲突并行工具，也支持 Plan/Team 批量并发和有界线程池。未实现通用 Tool Cache、Provider 级批处理和大小模型动态路由。优化时应同时观测 P50/P95/P99、成功率和 Token，不能只看平均延迟。

代码：[Tool 并行策略](src/my_agent/tools/parallel_policy.py#L21)、[有界并发执行](src/my_agent/tools/registry.py#L311)、[Plan 批次执行](src/my_agent/plan/executor.py#L130)

### 精选 16：Tool Calling 连续失败时，如何设计重试、熔断和降级？

> 公司：第四范式｜信源：N1｜原题编号：50
> AgentCli 状态：有错误分类、超时和重复失败终止；没有完整重试与熔断框架。

第一步是按错误类型决定动作：参数错误应修参，不应原样重试；认证和权限错误应立即停止；限流与瞬时网络错误可以指数退避；业务拒绝通常不可重试；带副作用的超时必须标为 `UNKNOWN`，先查询结果再决定是否重试。

推荐机制：

```text
classified error
→ retryable?
→ Retry Budget
→ capped exponential backoff + jitter
→ circuit breaker
→ alternate tool/provider
→ partial result or HITL
```

所有副作用调用需要 `idempotency_key`，否则重试可能重复创建资源或重复修改数据。AgentCli 已输出 `retryable/error_code/timed_out`，并在重复失败窗口后终止，但未实现通用 Backoff、熔断、Provider 替代和运行时幂等键。

代码：[ToolExecutionResult](src/my_agent/tools/execution.py#L47)、[超时结果](src/my_agent/tools/registry.py#L393)、[重复失败预算](src/my_agent/runtime/budget.py#L68)

### 精选 17：SFT 的完整流程是什么？训练数据集应该如何构建？

> 公司：阿里淘天｜信源：N2｜原题编号：54
> AgentCli 状态：已实现 Trace 到 SFT、Alpaca 导出和 LoRA 训练链路。

完整流程是：定义目标能力和评测集 → 收集/清洗高质量轨迹 → 转为对话或指令样本 → 划分 Train/Validation/Test → 按模型模板渲染 → LoRA/SFT 训练 → 离线评测 → Agent Runtime 回归。

AgentCli 的 `traces-to-sft` 只转换包含 `benchmark_result.status=passed` 的 Trace，并保留成功的非 `finish` 工具调用；随后可以导出 Alpaca 数据并使用 LLaMA-Factory 训练 Qwen3.5 LoRA。

数据质量比数量更重要。应去除密钥、隐藏答案、重复样本和失败噪声；按仓库或任务族划分数据，避免同一任务变体跨 Train/Test 泄漏；保留 Prompt、Tool Schema 和模型模板版本。只训练成功轨迹会缺少“如何从失败恢复”，可额外构造经过验证的纠错样本，但不能把未验证失败直接作为答案。

代码：[Trace 转 SFT](src/my_agent/data/builders.py#L316)、[Alpaca 导出](src/my_agent/data/converters.py#L75)、[LoRA 训练脚本](scripts/train_llamafactory_lora.sh#L101)

### 精选 18：PPO、DPO 和 GRPO 的主要区别是什么？AgentCli 使用了哪一种？

> 公司：阿里淘天、腾讯｜信源：N2、N5｜原题编号：55
> AgentCli 状态：未实现 PPO/DPO/GRPO；实现的是 OPD。

- PPO：在线采样，使用奖励模型、参考模型和 Value/Critic，通过裁剪策略梯度更新；控制能力强，但训练链复杂、显存和稳定性成本高。
- DPO：直接使用偏好对 `(chosen, rejected)` 优化相对概率，不需要训练时在线 Rollout 和 Critic；实现简单，但高度依赖离线偏好数据质量。
- GRPO：对同一 Prompt 采样一组回答，用组内相对 Reward 标准化优势，通常省去 Critic；仍需要在线生成、多样本 Reward 和较高采样成本。

AgentCli 的 OPD 不是上述三者。它在 Selection、Action、Writing、Maintenance 等角色上构造 Teacher/Student 视图，以 `KL(p_teacher || q_student)` 蒸馏策略，并对角色进行平衡采样。面试时应明确说“项目实现 OPD，可对比 PPO/DPO/GRPO，但不能声称项目已跑过这些算法”。

代码：[OPD KL Loss](src/my_agent/training/opd_loss.py#L51)、[角色平衡数据集](src/my_agent/training/opd_dataset.py#L159)、[OPD Collator](src/my_agent/training/opd_collator.py#L28)

### 精选 19：哪些任务适合交给 Agent，哪些任务仍然需要人工干预？

> 公司：小红书｜信源：X1｜原题编号：64
> AgentCli 状态：已实现按风险分级的 HITL，环境事务化不足。

适合 Agent 的任务通常可验证、可回滚、权限有限，例如代码检索、生成候选方案、只读分析和受测试约束的修改。涉及资金、删除、发布、权限提升、隐私、不可逆外部动作或低置信度高影响判断时，需要人工确认。

AgentCli 将只读工具、写工具、命令执行和 MCP 工具映射到不同策略：只读通常直接执行；EXECUTE 和高风险副作用要求审批；中风险写操作可配置 ask/allow/deny；MCP 默认要求批准。用户可以批准、拒绝、跳过或修改参数，并留下审计事件。

不足是审批解决“谁允许执行”，不等于解决“执行失败如何回滚”。生产系统还需要最小权限、Sandbox、预执行 Diff、补偿动作、外部资源确认和审批时效。

代码：[HITL Policy](src/my_agent/hitl/policy.py#L40)、[Approval 流程](src/my_agent/hitl/registry.py#L167)、[HITL Audit](src/my_agent/hitl/audit.py#L21)

### 精选 20：结合一个项目，说明你具体做了什么、产生了什么结果。

> 公司：智谱 AI｜信源：N12｜原题编号：69
> AgentCli 状态：可提供项目证据结构，但个人职责和指标必须由候选人补充。

推荐按“问题—方案—个人动作—验证—边界”回答，而不是按代码目录罗列功能：

1. 问题：现有 Coding Agent 在长任务中面临上下文超限、工具失败和经验无法复用等问题。
2. 方案：构建 ReAct/Plan/Team Runtime，并加入 Tool Registry、预算、Trace、HITL 和四层长期经验。
3. 个人动作：只描述自己真实负责的模块、关键设计取舍和解决过的故障，不把整个仓库都说成个人完成。
4. 验证：说明实际跑过的测试、协议或 Benchmark；没有实测数字就不要编造提升比例。
5. 边界：默认长期 Evolver 关闭，跨进程 Resume、通用混合 RAG 和生产级熔断仍需完善。

一个可复述模板是：

> 我负责的是「具体模块」。它解决「明确问题」，核心设计是「两到三个关键机制」。我用「真实测试或评测协议」验证正确性，并通过 Trace 定位过「真实 Bad Case」。目前已完成「代码事实」，尚未证明或尚未上线的是「边界」。

项目入口：[运行时工厂](src/my_agent/runtime/factory.py#L40)、[工具注册](src/my_agent/tools/registry.py#L45)、[上下文管理](src/my_agent/context/manager.py#L45)、[可观测性](src/my_agent/observability/tracing.py#L11)

### 精选 21：除了准确率，评价 Agent 还应该关注哪些指标？

> 公司：小红书｜信源：X1｜原题编号：47
> AgentCli 状态：已有部分运行、工具、Token 和测试指标。

准确率只回答“最后结果对不对”，不能解释 Agent 是否稳定、安全、经济。建议至少覆盖：

- 任务效果：成功率、测试通过率、部分完成率、拒答正确率。
- 决策质量：计划合法率、工具选择准确率、参数 Schema 通过率、停止决策正确率。
- 可靠性：首次成功率、重试恢复率、超时率、重复失败率、恢复正确率。
- 安全性：越权调用、策略阻断、HITL 触发、重复副作用和敏感信息泄露。
- 效率：P50/P95/P99 延迟、Token、LLM/Tool Call 数和成本。
- 用户体验：追问次数、人工接管率、结果采纳率和人工修改率。

AgentCli 的 TraceMetrics 已统计工具成功率、测试通过率、Token、停止原因、Budget Stop 和 Memory 事件；还缺少延迟分位数、重试恢复率、人工接管率和重复副作用率。

代码：[TraceMetrics](src/my_agent/observability/trace_metrics.py#L17)、[协议评测指标](src/my_agent/evaluation/protocol_metrics.py#L137)

### 精选 22：Agent 项目的质量指标从 65% 提升到 90%，这个指标应该如何评定？

> 公司：小红书｜信源：X2｜原题编号：52
> AgentCli 状态：可以构造对照评测，但项目没有已验证的“65%→90%”结论。

首先必须定义 65% 和 90% 的分子、分母与样本范围。例如是任务完全成功率、测试通过率，还是 LLM Judge 分数；是 20 个内部样例还是冻结的独立测试集。没有统一口径，两个百分比没有可比性。

正确评估方式是：冻结任务集、顺序、模型、Tool Schema、预算和 Evaluator，对旧版与新版做逐任务配对；报告样本数、绝对提升、相对提升、置信区间和分类型结果。还要检查提升是否以更多 Token、延迟、人工介入或安全风险为代价。

AgentCli Benchmark 能记录 `resolved/failure_type/visible/hidden test/patch_apply` 等结果，OPD Evaluation 支持隔离不同 Arm。面试时只能引用实际跑过且可复现的结果，不能把目标值、局部样例或二次资料写成项目实测。

代码：[ManifestEvalResult](src/my_agent/evaluation/manifest_benchmark.py#L89)、[Evaluation Matrix](src/my_agent/evaluation/opd_evaluation.py#L172)

### 精选 23：基础模型服务卡死或不可用时，后端如何兜底？

> 公司：第四范式｜信源：N1｜原题编号：51
> AgentCli 状态：已有单 Provider 超时和有限重试，没有多 Provider Failover。

处理顺序应是：请求级 Deadline → 取消当前请求 → 根据错误类型决定重试 → Provider 熔断 → 切换备用模型或降级模型 → 返回部分结果/HITL。429 应尊重 `Retry-After`，5xx 和瞬时网络错误可重试，认证或请求格式错误应立即失败。

AgentCli 的 OpenAI-compatible Client 会对 429、5xx 和 URL Error 做有限次数的简单指数退避，并设置请求 Timeout。当前没有 jitter、总 Retry Budget、Provider 健康检查、熔断和模型故障转移。

切换模型时不能只替换 URL，还要检查 Tool Calling 格式、上下文窗口、Prompt 模板和安全能力是否兼容；否则“服务恢复”可能变成静默质量下降。降级结果必须在 Trace 和用户结果中显式标记。

代码：[LLM 请求重试](src/my_agent/llm/__init__.py#L149)、[ReAct LLM 失败终止](src/my_agent/react/agent.py#L693)

### 精选 24：如何设计一条真正有助于 Bad Case 定位的 Agent Trace？

> 公司：不适用，AgentCli 项目深挖题｜主要信源：当前项目实现
> AgentCli 状态：已实现本地 JSONL 事件 Trace，标准 Span 和集中检索不足。

Trace 应能回答五个问题：谁触发、当时看到了什么、做了什么决定、环境返回了什么、最终为何结束。建议事件统一包含：

```text
trace_id / run_id / event_id
span_id / parent_span_id
phase / component / attempt
model_or_tool_version
sanitized_input_hash / output_hash
started_at / elapsed_ms
status / error_code / retryable
budget_snapshot / side_effect_state
```

AgentCli 当前事件包含 `time/run_id/event/payload`，能够记录 LLM、Tool、Memory、审批和完成事件，但没有统一 `event_id/span_id`，LLM 阶段耗时、Schema Version、Trace 完整性校验和统一脱敏也不足。短期可先补齐事件 Schema 和阶段耗时，再考虑 OpenTelemetry、集中查询和 Dashboard。

代码：[TraceEvent](src/my_agent/schema.py#L131)、[TraceWriter](src/my_agent/observability/tracing.py#L11)、[Agent 完成事件](src/my_agent/observability/tracing.py#L114)

### 精选 25：多 Agent 场景中，如何把父 Agent、子 Agent 和 Tool Trace 串起来？

> 公司：不适用，AgentCli 项目深挖题｜主要信源：当前项目实现
> AgentCli 状态：已能记录子 Trace 路径，缺少标准父子 Span。

父 Agent 创建子任务时，应生成子 `span_id`，同时传递根 `trace_id`、父 `span_id`、任务 ID 和依赖 ID。子 Agent 的 LLM 与 Tool 事件继续挂在该 Span 下，Reviewer 再引用被审查步骤和产物，这样才能从最终失败反向定位到具体子任务和工具调用。

AgentCli 的父完成事件会保存 `child_trace_paths`，TraceMetrics 可以递归读取子 Trace；Plan/Team 状态也保存步骤结果和 Trace 路径。这满足离线追踪的基础需求，但路径引用不是稳定的分布式因果关系，跨进程、文件移动或并发排序时较弱。

建议增加 `trace_id/span_id/parent_span_id/task_id/attempt`，并让依赖边、Reviewer 结果和重试都引用原 Span，而不是只依赖本地文件路径。

代码：[父子 Trace 路径](src/my_agent/observability/tracing.py#L114)、[递归 Trace 汇总](src/my_agent/observability/trace_metrics.py#L147)、[Team 状态事件](src/my_agent/team/agent.py#L798)

### 精选 26：如何单独评估 Tool Calling 的可靠性？

> 公司：不适用，关联信源：N2、X1｜主要信源：AgentCli 评测实现
> AgentCli 状态：已有工具准确率、解析率和执行成功率，恢复指标不足。

Tool Calling 应拆成四段测量：

1. 选择：是否选对工具，是否出现未知或不必要调用。
2. 参数：JSON 是否可解析，Schema 和业务前置条件是否通过。
3. 执行：成功率、错误码、超时、阻断和延迟分位数。
4. 恢复：首次成功率、重试后成功率、无效重试率、重复副作用率。

离线集应包含相似工具、缺失参数、权限不足、超时和返回冲突等案例。在线指标还应按工具名、版本、错误类型和调用方分桶，避免总体成功率掩盖单个关键工具退化。

AgentCli 的 Protocol Metrics 支持参考 Tool 名称准确率与运行时解析率，TraceMetrics 统计执行成功和阻断；尚未统计参数字段级准确率、首次成功、恢复成功和重复副作用。

代码：[Tool Accuracy](src/my_agent/evaluation/protocol_metrics.py#L97)、[运行时解析评测](src/my_agent/evaluation/protocol_metrics.py#L137)、[Tool Trace 指标](src/my_agent/observability/trace_metrics.py#L163)

### 精选 27：Tool 超时后为什么不能直接重试？如何处理 `UNKNOWN` 状态？

> 公司：不适用，AgentCli 项目深挖题｜主要信源：当前 Tool 执行实现
> AgentCli 状态：超时会结构化返回，但未区分副作用是否已经提交。

超时只代表调用方没有及时拿到结果，不代表服务端一定没有执行。只读查询通常可以安全重试；创建订单、发送消息、写数据库或修改外部资源的调用可能已经成功，直接重试会产生重复副作用。

正确状态机是：

```text
PENDING → RUNNING → SUCCEEDED
                  → FAILED
                  → UNKNOWN
```

进入 `UNKNOWN` 后，Harness 应使用 `request_id/idempotency_key` 查询外部状态；确认未执行才能重试，确认已执行则补记结果，无法确认时请求人工处理或执行补偿动作。

AgentCli 当前 Tool Batch Timeout 返回 `retryable=True`，这对外部副作用工具过于粗糙。建议根据 Tool Risk 和幂等声明决定是否可重试，并为 MCP/EXTERNAL 工具增加状态查询与业务幂等契约。

代码：[Tool Batch Timeout](src/my_agent/tools/registry.py#L393)、[Tool Risk](src/my_agent/tools/spec.py#L12)、[MCP Tool 注册](src/my_agent/mcp/source.py#L32)

### 精选 28：如何设计降级策略，并证明降级没有造成更严重的问题？

> 公司：不适用，关联信源：N1、N2｜主要信源：当前失败处理实现
> AgentCli 状态：有局部 Fallback，缺少统一降级编排与效果评测。

降级要先定义不可破坏的语义。例如评测任务可以 Fail Closed，避免悄悄改变实验协议；交互查询可以退化为只读、部分结果或无长期 Memory；高风险写操作不能因为审批或主模型不可用就绕过安全策略。

推荐为每个组件定义有序路径：

```text
primary
→ bounded retry
→ compatible fallback
→ partial result / read-only mode
→ HITL or explicit failure
```

验证时使用同一批故障注入案例，比较降级成功率、答案质量损失、P95 延迟、额外成本、安全违规和重复副作用，并统计 `degraded=true` 后的最终 Outcome。没有显式标记的“静默降级”无法审计，也容易让线上指标看似成功、实际质量下降。

AgentCli 已有上下文压缩 Fallback、结构化 Tool 错误和 Memory 局部降级，但 Provider、Tool 与 Memory 的策略尚未统一。

代码：[上下文压缩重试](src/my_agent/react/agent.py#L365)、[Tool 错误边界](src/my_agent/tools/registry.py#L161)、[Legacy Memory 降级](src/my_agent/memory/evolver/runtime/legacy.py#L216)

### 精选 29：如何把一次 Bad Case 变成可复现、可防回归的测试？

> 公司：不适用，AgentCli 项目深挖题｜主要信源：Trace 与 Benchmark 实现
> AgentCli 状态：具备 Trace 和 Benchmark 证据，缺少自动回放与案例晋升流程。

先从 Trace 找到第一个异常事件，而不是只记录最终 `hidden_test_failed`。随后固定：用户输入、仓库提交、初始文件、模型与 Prompt 版本、Tool Schema、配置、随机参数、外部响应或 Mock，以及期望 Outcome。

回归案例应同时包含：

- 最小复现输入。
- 失败阶段和根因标签。
- 确定性验收器。
- 修复前必须失败、修复后必须通过的断言。
- 成本、权限和副作用上限。

AgentCli Benchmark 已保存测试结果、Patch、修改文件、Trace 和 `failure_type`，但当前分类更多描述最终表现，不能自动定位规划、检索、模型、工具、环境或 Memory 根因。建议增加统一 `FailureEnvelope`，修复验证通过后再把案例晋升到长期回归集或 Memory。

代码：[Benchmark 结果证据](src/my_agent/evaluation/manifest_benchmark.py#L89)、[Failure Type](src/my_agent/evaluation/manifest_benchmark.py#L1490)、[Benchmark Trace 结果](src/my_agent/observability/tracing.py#L37)

### 精选 30：生产环境中应该如何监控 Agent 的健康度并设置告警？

> 公司：不适用，AgentCli 项目深挖题｜主要信源：当前离线统计实现
> AgentCli 状态：有本地离线统计，没有集中式实时监控和告警。

建议围绕六类信号建立 SLI/SLO：

- Outcome：任务成功率、测试通过率、部分结果率。
- Latency：端到端和 LLM/Tool/Reviewer 分阶段 P50/P95/P99。
- Errors：按 `failure_stage/error_code/tool/provider` 分桶的错误率。
- Saturation：并发、队列长度、Token 与成本预算消耗。
- Safety：策略阻断、HITL 等待、越权和 `UNKNOWN` 副作用数量。
- Recovery：重试恢复率、熔断状态、降级率和恢复后成功率。

告警应基于时间窗口和错误预算消耗，而不是单次失败。例如关键 Tool 五分钟错误率超过阈值、`UNKNOWN` 副作用非零、P95 延迟持续超 SLO 或降级率突增。每个告警都应带 Trace 查询入口和受影响版本。

AgentCli 当前通过 JSONL Trace 和 `stats/trace_metrics` 做事后汇总，适合本地诊断；生产化需要补充标准指标导出、集中存储、Dashboard、告警和统一脱敏。

代码：[离线 Trace 汇总](src/my_agent/observability/trace_metrics.py#L112)、[Stats 汇总](src/my_agent/observability/stats.py#L95)、[Trace 写入](src/my_agent/observability/tracing.py#L11)

---

## 1. 先讲清楚当前实现边界

当前配置默认 `memory_enabled=True`，但 `memory_evolver_mode="off"`。因此默认运行状态是：

- 启用当前会话内的短期记忆和上下文压缩。
- 不启用四层长期经验的检索、选择、写入和维护。
- 只有显式选择 Legacy 或 Formal Evolver，长期经验链路才会参与任务执行。

代码：[默认配置](src/my_agent/config.py#L79)、[Disabled Evolver](src/my_agent/memory/evolver/runtime/disabled.py#L74)

### 1.1 总体架构

```text
当前任务
  ├─ 短期记忆：目标、对话、Tool Call、Tool Result、摘要
  │    └─ 超预算时执行 Map-Reduce 压缩或确定性降级摘要
  │
  └─ 长期经验：trajectory / tip / skill / tool
       ├─ 检索：Legacy 词法检索 / Formal Embedding 检索
       ├─ 选择：规则 Selector / LLM Selector
       ├─ 使用：作为独立 Memory System Message 注入 Prompt
       ├─ 写入：权威 Outcome 后提取、校验、原子持久化
       └─ 维护：lookup / merge / delete / finish
```

### 1.2 实现状态总览

| 能力 | 当前状态 | 结论 |
|---|---|---|
| 会话级短期记忆 | 已实现 | 有条目数、Token 上限和 FIFO 淘汰 |
| 上下文压缩 | 已实现但可增强 | 支持 Map-Reduce 和失败降级，缺少结构化关键事实保护 |
| 四层长期经验 | 已实现，默认关闭 | 四层是语义类型，不是缓存层级 |
| 长期经验检索 | 部分实现 | Legacy 为词法检索；Formal 为 Embedding Cosine；无正式混合检索和 ANN |
| 长期经验写入 | Formal 链路已实现 | 在权威 Outcome 后执行 Writer、Validator 和原子提交 |
| 并发与崩溃一致性 | 已实现 | FileLock、去重、CAS Revision、fsync、原子替换、提交后恢复 |
| 经验价值归因 | 部分实现 | Legacy 有 Outcome Attribution；Formal 尚未完全使用价值反馈闭环 |
| 经验维护 | 部分实现 | Formal 支持查找、同层合并、删除和结束，不支持 Promotion |
| 故障降级 | 部分实现 | 各链路策略不统一，Formal Embedding 失败通常直接终止 |
| Memory 对照评测 | 有基础框架 | 已有四个 Arm，但 `no_memory` 实际只关闭长期 Evolver |

---

## 2. 架构与运行链

### 问题 1：当前项目的 Memory 架构是怎样的？短期记忆和长期记忆有什么区别？

**实现状态：已实现，长期链路默认关闭。**

项目可以概括为“两级存储、四类长期经验”。

短期记忆保存当前 Session 中的用户目标、对话消息、Tool Call、Tool Result 和压缩摘要。它服务于当前任务的连续推理，不直接持久化为可跨任务复用的经验。写入入口见 [MemoryManager 短期内容写入](src/my_agent/memory/manager.py#L206)。

长期记忆持久化为 `ExperienceMemory`，分为：

- `trajectory`：完整任务解决轨迹。
- `tip`：短经验、警告或注意事项。
- `skill`：有前置条件和执行步骤的通用方法。
- `tool`：可复用命令、脚本或工具定义。

四层定义见 [ExperienceTier](src/my_agent/memory/experience/models.py#L22)，存储实现见 [ExperienceStore](src/my_agent/memory/experience/repository.py#L53)。这里的“四层”是经验的语义类型，不是按时间、速度或存储介质划分的 L1/L2/L3/L4 缓存。

**不足与建议：** 默认配置只提供会话级短期记忆。面试时不要直接说“AgentCli 默认会跨任务学习”，应说明长期 Evolver 需要显式开启。

### 问题 2：为什么把长期记忆分成 Trajectory、Tip、Skill、Tool 四层？

**实现状态：已实现，收益仍需消融实验验证。**

分层的核心原因是不同经验的复用粒度和约束不同：

| Tier | 复用粒度 | 典型用途 |
|---|---|---|
| Trajectory | 完整路径 | 参考相似任务如何端到端完成 |
| Tip | 单点提醒 | 避免重复踩坑、记住环境或接口约束 |
| Skill | 通用方法 | 在满足前置条件时复用一组步骤 |
| Tool | 可执行能力 | 复用命令、脚本或工具定义 |

每层都有强类型 Payload。例如 Skill 要求 `preconditions` 和非空 `steps`，Tool 至少要包含 `code` 或 `command`。代码见 [TrajectoryPayload](src/my_agent/memory/experience/models.py#L82)、[TipPayload](src/my_agent/memory/experience/models.py#L126)、[SkillPayload](src/my_agent/memory/experience/models.py#L154)、[ToolPayload](src/my_agent/memory/experience/models.py#L188)。

分层后可以独立设置每层 TopK、权重、Token Budget 和维护规则，避免短 Tip 与长 Trajectory 使用完全相同的策略。

**不足与建议：** 当前实现证明了“四层可运行”，但没有证明它一定优于相同内容和 Token 数量的扁平 Memory。需要增加 `flat_memory`、leave-one-tier-out 和等 Token 对照实验。

### 问题 3：短期记忆如何解决上下文窗口溢出？

**实现状态：已实现核心链路，关键事实保护不足。**

项目使用三层控制：

1. 短期存储设置 Token 和条目数硬上限，超限时按 FIFO 淘汰，见 [ShortTermMemory](src/my_agent/memory/short_term/store.py#L11) 和 [FIFO 淘汰](src/my_agent/memory/short_term/store.py#L130)。
2. 构造 Prompt 时先计算固定 Prompt、仓库上下文、Tool Schema、长期记忆和输出预留空间，再把剩余预算分给短期记忆，见 [AgentContextManager](src/my_agent/context/manager.py#L87)。
3. 达到动态阈值后，压缩较老的完整对话轮次，同时保留最近若干用户轮次。压缩入口见 [压缩触发与替换](src/my_agent/memory/manager.py#L541)，算法见 [Map-Reduce Compressor](src/my_agent/memory/short_term/compression.py#L32)。

压缩失败时会使用确定性文本摘要作为 Fallback，而不是直接阻断 Agent。

**不足与建议：** LLM 摘要可能丢失文件路径、工具参数、错误码、测试状态和未完成事项。建议把这些信息作为结构化 `critical_facts` 与自然语言摘要一起保存，并在压缩后做一致性校验。

### 问题 4：长期记忆什么时候写入？如何避免把所有运行过程都写进去？

**实现状态：Formal 链路已实现，证据绑定可增强。**

Formal 路径不是边执行边随意写 Memory，而是在任务结束且获得权威结果后写入：

```text
执行任务
→ 保存 Episode
→ Benchmark 产生 AuthoritativeTaskOutcome
→ Writer 提取可复用经验
→ Validator 校验
→ 原子写入 ExperienceStore
```

Coordinator 会检查 Outcome 完成状态、任务 ID、Task Group、Policy Identity 和 Repository Revision，见 [Formal Finalize](src/my_agent/memory/evolver/coordinator.py#L341)。

Writer Prompt 要求只提取由轨迹和结果支持的通用经验，禁止保存一次性答案、密钥和无证据结论，见 [Formal Writer](src/my_agent/memory/evolver/writing/formal.py#L147)。Validator 进一步校验：

- Tier 与 Payload Schema。
- 最低置信度、数量和长度限制。
- 重复 Proposal。
- 密钥、隐藏测试和 Ground Truth 等敏感信息。
- 危险 Tool 命令。

代码见 [ExperienceProposalValidator](src/my_agent/memory/evolver/writing/validation.py#L42)。

**不足与建议：** `writer_confidence` 仍主要是模型自评。建议为每条 Memory 增加 Evidence Span，强制引用对应 Tool Step、测试结果或 Outcome 字段，再由确定性校验器验证引用存在且结论一致。

### 问题 5：当前长期记忆如何检索和排序？

**实现状态：两套路径已实现，规模化检索不足。**

Legacy 模式采用分 Tier 的词法召回，根据字符片段候选、关键词覆盖率和时间衰减打分，见 [LexicalExperienceRetriever](src/my_agent/memory/experience/retrieval/lexical.py#L99) 和 [词法与时间衰减评分](src/my_agent/memory/experience/retrieval/lexical.py#L376)。规则 Selector 再综合：

```text
retrieval_score
× tier_weight
× attribution_value
× confidence
```

并应用每层数量上限、总条目数和 Token Budget，见 [Legacy Selection Score](src/my_agent/memory/evolver/selection/legacy.py#L172)。

Formal 模式默认使用 Embedding Cosine 召回，再由 LLM Selector 从冻结的 Candidate Snapshot 中选择：

- [Formal Retriever 配置](src/my_agent/memory/evolver/runtime/factory.py#L71)
- [EmbeddingRetriever](src/my_agent/memory/experience/retrieval/embedding.py#L74)
- [Retrieve Once + Select](src/my_agent/memory/evolver/coordinator.py#L253)

**不足与建议：** 当前没有正式的 BM25/词法与向量混合召回，没有 Cross-Encoder Reranker，Embedding 路径也不是 ANN 索引。经验规模扩大后，建议使用混合召回、RRF 融合、二阶段 Rerank 和向量索引。

### 问题 6：检索出的 Memory 如何进入 Agent Prompt？会不会挤占正常上下文？

**实现状态：已实现预算控制，使用效果缺少强归因。**

长期记忆被渲染为独立的 System Message，通常插入原始 System Prompt 之后，见 [Memory Context 注入](src/my_agent/context/manager.py#L223)。注入前有两层预算：

1. Candidate Selection 的 `max_items + token_budget`。
2. `AgentContextManager` 的整体 Prompt Budget。

Formal 模式在任务开始时只检索和筛选一次，并冻结 Candidate Snapshot。任务中途即使 Memory Store 改变，当前任务看到的候选也不会变化，有利于实验复现。

Plan 和 Team 的 Planner 也会读取长期记忆，见 [Plan Planner Memory](src/my_agent/plan/agent.py#L94) 和 [Team Planner Memory](src/my_agent/team/agent.py#L120)。子 ReAct 任务使用新的任务级短期记忆，但共享长期 ExperienceStore，见 [Child Memory Fork](src/my_agent/react/child_runner.py#L50)。

**不足与建议：** 当前主要是文本注入，模型可能忽略或错误使用 Memory。建议在 Action/Decision Trace 中记录 `used_memory_ids`、引用位置、影响的决策和后续 Outcome，形成在线使用证据。

---

## 3. 隔离、可靠性与治理

### 问题 7：项目如何实现不同项目和不同任务之间的 Memory 隔离？

**实现状态：项目级隔离已实现，租户权限模型不足。**

隔离由两个维度共同决定：

- `memory_dir`：物理存储目录。
- `memory_project_key`：逻辑可见性边界。

长期经验只允许 `PROJECT` 或 `GLOBAL` Scope。Project Memory 只有相同 `project_key` 的运行可见，Global Memory 对所有项目可见。代码见 [Scope 校验](src/my_agent/memory/experience/models.py#L250) 和 [可见性索引](src/my_agent/memory/experience/repository_index.py#L67)。默认 `project_key` 从仓库绝对路径派生，见 [Memory Factory](src/my_agent/memory/factory.py#L43)。

Benchmark 支持 `per_task`、`shared_stream` 和 `shared_by_group` 三种 Memory 目录策略。后两种模式会为同一 Stream 或 Group 设置稳定 Project Key，见 [Manifest Memory Modes](src/my_agent/evaluation/manifest_benchmark.py#L44)。

**不足与建议：** `project_key` 只是普通字符串，不是安全边界。生产多租户环境应增加 `tenant_id/user_id/project_id` 联合命名空间、鉴权、Global Memory 审批、来源白名单和审计日志。

### 问题 8：Memory Store 如何处理重复写入、并发写入和崩溃？

**实现状态：核心持久化能力已实现。**

`ExperienceStore` 已具备：

- 线程锁和跨进程 FileLock。
- 基于 Tier、Scope、Project、Fingerprint 的去重。
- Repository Revision。
- `expected_revision` 乐观并发控制。
- 临时文件写入、flush、fsync 和原子 replace。
- 写后重新加载并验证 Revision。
- Writer Post-Commit 状态恢复。

代码见 [原子批量追加和 CAS](src/my_agent/memory/experience/repository.py#L171)、[原子文件持久化](src/my_agent/memory/experience/repository.py#L469) 和 [Writer Post-Commit 恢复](src/my_agent/memory/evolver/writing/persistence.py#L126)。

运行启动时可以宽松加载并跳过坏行，写入和维护等可信边界则使用严格加载，见 [宽松与严格加载](src/my_agent/memory/experience/repository.py#L359)。

**不足与建议：** 当前 JSONL 更新需要重写完整文件。规模达到数万或数十万条后，写入、索引构建和 Embedding 扫描会成为瓶颈。届时应迁移到 SQLite/PostgreSQL，并配套向量索引和可恢复事务。

### 问题 9：系统如何判断一条 Memory 是有价值还是有害的？

**实现状态：Legacy 有相关性归因，Formal 价值闭环不完整。**

Legacy/离线路径会基于 Outcome 比较：

- Memory 被选中时的成功率和平均 Reward。
- 成为候选但未被选中时的成功率和平均 Reward。
- Candidate、Selected、Not Selected 样本数。
- 按 Task Type 分组后的 Reward 差异。

归因计算见 [Attribution 评分](src/my_agent/opd_data/legacy/attribution.py#L221)，结果会写回 `attribution_value`、`attribution_confidence`、`selected_count`、`success_when_selected` 和 `last_used` 等字段，见 [ExperienceMemory Attribution 字段](src/my_agent/memory/experience/models.py#L266)。Legacy Selector 会把价值和置信度纳入排序。

这仍然是相关性归因，不是严格因果归因，因为任务难度、其他 Memory 和模型策略都会同时影响选择与结果。

Formal Selector 的 Candidate Snapshot 主要包含内容、Tier、检索分数和 Token 数，没有把 Attribution Value 作为显式决策输入，见 [Candidate Snapshot](src/my_agent/memory/evolver/selection/service.py#L70)。

**不足与建议：** 增加 Memory 级 A/B、随机错配对照、最小样本量和置信区间；Formal Selector 应读取稳定的价值统计，同时保留探索比例，避免早期噪声让有价值经验永远失去曝光。

### 问题 10：系统如何实现遗忘、去重、合并和错误记忆清理？

**实现状态：Formal Maintenance 已实现部分治理动作。**

Formal Maintenance Agent 可以执行：

- `lookup`
- `merge`
- `delete`
- `finish`

工具定义见 [Formal Maintenance Tools](src/my_agent/memory/evolver/maintenance/formal/tools.py#L25)。修改先进入 Staged State，完整校验后再原子提交。Intent/Completion 记录可以识别“Repository 已提交，但完成记录未写入”的中间状态，见 [Maintenance Transaction](src/my_agent/memory/evolver/maintenance/formal/transaction.py#L46)。

Reducer 还会保护 Global、Manual、Protected 和其他 Project 的 Memory，见 [Maintenance 安全约束](src/my_agent/memory/evolver/maintenance/repository_reducer.py#L107)。

**不足与建议：** Formal Maintenance 仅支持删除和同层合并，不支持跨层 Promotion；Legacy 虽有 Promote 规则，但属于另一条链路。Memory 也缺少代码版本和适用版本范围，仓库重构后旧经验可能继续被召回。建议增加 `valid_from_revision`、`invalid_after_revision`、依赖指纹和过期原因。

### 问题 11：Memory 系统发生故障时，Agent 会怎样降级？

**实现状态：已有局部降级，但策略不统一。**

当前不同链路的处理方式是：

- 短期压缩 LLM 失败：使用确定性摘要，不阻断 Agent。
- Legacy Selector 失败：Fail Closed，注入空 Memory Context。
- Runtime 宽松加载遇到坏行：跳过坏数据并记录 Trace。
- 严格写入或维护发现坏数据：拒绝继续。
- Formal Embedding 检索失败：抛出 `EmbeddingRetrievalError`，通常导致 Formal 任务失败。

代码见 [压缩 Fallback](src/my_agent/memory/short_term/compression.py#L86)、[Legacy Selector 降级](src/my_agent/memory/evolver/runtime/legacy.py#L216) 和 [Formal Embedding 错误](src/my_agent/memory/experience/retrieval/embedding.py#L97)。

Formal 模式严格失败有利于训练数据与实验协议的完整性；交互或生产环境更适合配置化降级：

```text
Embedding 检索失败
→ 尝试词法检索
→ 仍失败则使用空长期记忆
→ Trace 标记 degraded=true、failure_stage、fallback_path
```

**不足与建议：** 统一错误分类、降级策略和可观测字段；评测模式保持 Fail Closed，交互模式优先可用，但必须显式暴露降级状态，不能悄悄改变实验语义。

---

## 4. 评测与训练

### 问题 12：如何证明 Memory 真的提升了 Agent，而不是增加了 Prompt 或碰巧命中答案？

**实现状态：已有四 Arm 框架，严谨对照仍需补充。**

当前项目提供四个核心评测 Arm：

- M0 + no memory
- M0 + memory
- Trained + memory
- Trained + no memory

各 Arm 使用相同任务顺序、Token Budget、工具集合和 Evaluator，并隔离 Repository、Ledger 与输出目录。代码见 [Evaluation Matrix](src/my_agent/evaluation/opd_evaluation.py#L172)、[固定评测环境](src/my_agent/evaluation/opd_evaluation.py#L279) 和 [评测隔离检查](src/my_agent/evaluation/opd_evaluation.py#L511)。

更有说服力的任务协议是 Teach→Transfer：

```text
Teach 1：学习项目规则
Teach 2：学习工具或修复模式
Transfer 1-3：在新任务中复用此前经验
```

Transfer 阶段应冻结 Writer 和 Maintenance，避免一边测试一边继续学习。还应增加：

- `shuffled_memory`：Memory 数量和 Token 相同，但内容随机错配。
- `flat_memory`：内容相同，但取消四层结构。
- Leave-one-tier-out：依次移除 Trajectory、Tip、Skill、Tool。
- 等 Token 空文本或普通上下文对照：排除“只是多给了 Prompt”的影响。
- 报告任务成功率、Token、延迟、Tool Call 数、Memory 增长量、负收益率和重复副作用。

需要特别说明：当前 `no_memory` Arm 设置的是 `memory_evolver_mode=off`，短期会话记忆和上下文压缩仍然存在。因此它准确的含义是 **no long-term experience memory**，不是完全没有任何 Memory。代码见 [评测运行环境](src/my_agent/evaluation/opd_evaluation.py#L293)。

### 问题 13：Memory Evolver 和 OPD 训练是什么关系？Memory 是否等于模型训练？

**实现状态：两条链路均有实现，但日常在线学习边界明确。**

两者不是一回事。

Memory Evolver 改变的是运行时外部状态：

```text
任务 → 检索经验 → 选择经验 → 执行 → 写经验 → 维护经验
```

OPD 改变的是模型策略，训练模型如何完成：

- Selection 决策。
- Action 决策。
- Writing 决策。
- Maintenance 决策。

Formal Coordinator 使用同一个 Policy 和 DecisionRecorder 保存这些角色的决策证据，见 [Formal Coordinator Policy Binding](src/my_agent/memory/evolver/coordinator.py#L84) 和 [Decision Recorder](src/my_agent/memory/evolver/coordinator.py#L101)。

准确表述是：

> Memory 提供可读写的外部经验库；OPD 训练模型如何检索、使用、生成和维护这套经验。只训练模型不会自动更新 Memory，只写 Memory 也不会改变模型参数。

Formal 模式要求完整 Task Metadata 和权威 Outcome。普通交互任务缺少这些字段时会跳过 Formal Evolver Session，见 [Interactive Formal Session Skip](src/my_agent/react/agent.py#L139)。所以当前 Formal Memory Evolver 更接近严格评测和训练采集链，而不是无约束的日常在线自学习系统。

---

## 5. 当前主要不足与演进优先级

### P0：正确性、证据和降级语义

1. 为 Writer Proposal 增加 Evidence Span，绑定 Tool Step、测试结果和权威 Outcome。
2. 统一 Memory 错误分类与降级策略，明确区分评测 Fail Closed 和生产可用性优先。
3. 在 Trace 中记录 `candidate_memory_ids`、`selected_memory_ids`、`used_memory_ids`、Fallback 路径和最终 Outcome。
4. 为摘要增加结构化关键事实，防止上下文压缩丢失约束、进度和失败信息。

### P1：评测与价值闭环

1. 实现 Teach→Transfer、`shuffled_memory`、`flat_memory` 和 leave-one-tier-out。
2. Transfer 阶段冻结 Writer/Maintenance，并统一任务顺序、Token、工具和模型配置。
3. 将 Formal Selector 与稳定的 Attribution 统计闭环，同时保留探索策略。
4. 将 `no_memory` 报告名称改为 `no_long_term_experience_memory`，避免实验含义被误读。

### P2：规模与治理

1. 从完整 JSONL 重写迁移到事务数据库和向量索引。
2. 增加租户、用户、项目联合命名空间和 Global Memory 审批。
3. 为 Memory 增加代码版本、依赖指纹和有效期。
4. 增加 Hybrid Retrieval、RRF、Reranker 和 ANN。
5. 补齐 Formal Maintenance 的跨层 Promotion。

---

## 6. 一页式面试速记

### 6.1 30 秒架构回答

AgentCli 的 Memory 分为会话级短期记忆和持久化长期经验。短期记忆保存当前任务消息与工具结果，受 Token 和条目上限约束，超预算时执行 Map-Reduce 压缩。长期经验分为 Trajectory、Tip、Skill、Tool 四类；Legacy 使用词法召回和规则选择，Formal 使用 Embedding 召回与 LLM Selector。Formal Writer 只在权威 Outcome 后提取经验，并经过 Schema、安全、置信度和 Repository Revision 校验后原子写入。默认配置只启用短期记忆，长期 Evolver 默认关闭。

### 6.2 30 秒可靠性回答

Memory Store 已有 FileLock、Fingerprint 去重、CAS Revision、fsync、原子替换和提交后恢复，持久化一致性基础较完整。主要不足是各链路降级策略不统一、Formal Embedding 失败通常终止、Memory 使用缺少在线强归因、摘要缺少结构化关键事实保护，以及 JSONL 在大规模下会成为性能瓶颈。

### 6.3 30 秒评测回答

不能只比较开关 Memory 后的成功率。应采用 Teach→Transfer，并固定任务顺序、模型、工具、Token Budget 和 Evaluator；增加 shuffled-memory 排除偶然命中、flat-memory 验证分层价值、leave-one-tier-out 判断各 Tier 贡献，Transfer 阶段冻结 Writer 与 Maintenance。同时报告成功率、Token、延迟、Tool Call 数、Memory 增长和负收益率。当前 `no_memory` 只关闭长期 Evolver，准确说法是 no long-term experience memory。

### 6.4 容易说错的四点

1. 不要把四层 Memory 说成四级缓存；它们是四种经验语义。
2. 不要说默认开启跨任务学习；默认 `memory_evolver_mode="off"`。
3. 不要把 Attribution 说成因果证明；当前主要是相关性统计。
4. 不要把 Memory Evolver 和 OPD 混为一谈；前者改变外部经验库，后者改变模型参数和策略。

---

## 7. 结合简历的定制高价值追问

本节的问题不是公开面经原题，而是面试官看到简历后很可能继续追问的项目题。简历中的项目名为 **CODINCLI**，以下答案按当前 AgentCli 仓库实现核对。

简历明确写到：分层上下文与 Token Budget、HITL 人工审批、Trajectory/Tip/Skill/Tool 四层经验记忆、Qwen3.5-4B BF16 LoRA SFT、两项 95% 协议指标，以及“尝试实现 OPD-Evolver”。回答时应保持三条边界：

1. 仓库中有代码和测试，只能证明实现了相应机制，不能自动证明线上效果。
2. 简历中的 95% 需要能够说明评测集、样本量、口径、模型版本和原始结果文件；解析成功不等于工具选择正确或任务成功。
3. OPD 当前可以详细讲方法和实现链路，但在没有训练 Checkpoint、评测矩阵结果和可复现实验记录时，应继续使用“尝试实现、完成方法链路”而不是“已经显著提升”。

### 7.1 HITL 与环境副作用

#### 简历追问 1：你在 AgentCli 中实现的 HITL 完整链路是什么？为什么不能让模型自己决定风险？

可以这样回答：

> 我的设计原则是“模型提出动作，Runtime 决定动作是否有资格执行，人对高风险动作保留最终授权”。模型擅长结合语义给建议，但不能同时充当动作发起者和权限裁判，否则 Prompt Injection、错误推理或上下文污染都可能绕过安全边界。

当前执行链路分为七步：

1. 根据工具名解析注册信息，并解析模型生成的参数。
2. 使用 JSON Schema 校验参数结构。
3. 在询问人工之前执行确定性的 Preflight，例如路径是否逃逸仓库、是否经过符号链接、是否访问保护文件、测试命令是否在白名单内。
4. 静态风险策略根据工具类型判定：READ 默认安全；EXECUTE 为高风险并要求审批；WRITE 等有副作用操作按配置选择询问、放行或拒绝；MCP 工具默认按外部中风险工具处理。
5. 如果需要审批，生成包含 `run_id`、`tool_call_id`、工具名、参数摘要和风险描述的请求。
6. 人工可以批准、批准本次会话中的同类工具、修改参数、拒绝或跳过。
7. 执行后写审计日志，并向 Trace 发出审批开始、审批完成或审计失败事件。

顺序上必须先做静态 Preflight，再请求人工。人工审批不应该覆盖“路径逃逸”或“危险 Shell 语法”这类硬约束；即使用户修改了参数，也会重新经过解析、Schema、Preflight 和风险策略，而不是直接执行。

当前风险判断仍以静态策略为主。代码虽然预留了 `RiskJudge` 接口，但默认装配没有注入真正的 LLM Judge，`NoopRiskJudge` 只返回静态结论。因此面试时不应说成“已经实现了 LLM 动态风险识别”，更准确的说法是“预留接口，当前生产边界由确定性策略控制”。

代码：[静态风险策略](src/my_agent/hitl/policy.py#L40)、[HITL 执行入口](src/my_agent/hitl/registry.py#L56)、[审批请求](src/my_agent/hitl/registry.py#L167)、[工具静态 Preflight](src/my_agent/tools/registry.py#L225)、[仓库路径与命令保护](src/my_agent/tools/hooks.py#L67)

#### 简历追问 2：批准、批准全部、修改、拒绝和跳过为什么要设计成不同状态？

这些状态对应不同的后续控制语义，不能只用一个布尔值：

- `APPROVED`：只批准当前调用，随后正常执行。
- `APPROVED_ALL`：缓存当前工具的批准状态，后续相同工具不再逐次询问。
- `MODIFIED`：人工改变参数。Runtime 会重新解析和校验新参数，防止修改后形成新的越权请求。
- `REJECTED`：当前操作没有授权，但模型可以选择更安全的替代方案，所以结果被标成阻塞且可重新决策。
- `SKIPPED`：表示用户明确不希望当前动作继续，返回结果会提示模型不要重复提交同一个调用，因此被标成不可重试。

当前实现的优点是拒绝和跳过具有不同语义，修改参数也没有绕过安全检查。主要不足是 `APPROVED_ALL` 的缓存粒度只有工具名，没有资源范围、参数类别、TTL 和风险版本。例如，同一次运行中批准一次 `write_file`，理论上会覆盖该工具后续不同文件的调用；`ApprovalScope.SERVER` 虽然存在于类型中，当前主链路实际使用的仍是 Tool Scope。

更稳妥的演进方式是把批准缓存键扩展为：

```text
(run_id, tool_name, resource_scope, normalized_argument_class, risk_policy_version)
```

同时设置 TTL；删除、外部发布、付费、权限变更等高风险动作强制逐次审批，不能进入 `approved_all`。拒绝后还应记录被拒绝调用的规范化签名，防止模型只微调无关参数反复询问。

代码：[审批状态定义](src/my_agent/hitl/types.py#L15)、[终端审批交互](src/my_agent/hitl/handler.py#L77)、[修改参数后的重新校验](src/my_agent/hitl/registry.py#L278)、[拒绝与跳过语义](src/my_agent/hitl/registry.py#L230)

#### 简历追问 3：无人值守环境没有人批准时怎么办？当前实现是否真正 Fail Closed？

当前命令行入口会判断 stdin 是否为交互终端：

- 交互环境使用 `TerminalHitlHandler`，显示风险和参数并等待输入。
- 非交互环境使用 `NonInteractiveHitlHandler`。只要 HITL 已启用，需要审批的调用会被拒绝，不会因为没有 TTY 就自动放行。

这部分是 Fail Closed 的。但还要诚实说明两个配置边界：

1. 如果配置直接关闭 HITL，`RepoTools` 会使用普通 `ToolRegistry`，写入和执行类工具不会经过人工审批；静态 Preflight 仍然存在，但它只负责硬安全约束，不等于人工授权。
2. `HitlToolRegistry` 发现 Handler 被动态设置为 Disabled 时，会执行工具并把审批状态记为 `disabled`。这适合显式关闭功能的本地场景，但不适合作为“审批服务故障”的默认处理。

生产系统应区分“管理员明确关闭审批”和“审批服务不可用”两个状态。后者对中高风险动作必须拒绝或进入 `PENDING_APPROVAL` 队列，不能降级成自动放行。还应为审批增加超时、取消、请求持久化和恢复机制；当前终端 Handler 是同步阻塞式交互，不是分布式审批中心。

代码：[默认 Handler 选择](src/my_agent/runtime/runner.py#L111)、[非交互拒绝](src/my_agent/hitl/handler.py#L143)、[Handler Disabled 分支](src/my_agent/hitl/registry.py#L70)、[RepoTools 的 Registry 装配](src/my_agent/tools/repo_tools.py#L60)

#### 简历追问 4：HITL 已经批准了一个有副作用的 Tool，但调用超时或进程崩溃，如何避免重复副作用？

HITL 解决的是“有没有授权”，不自动解决“动作到底执行到哪一步”。当前实现会记录审批和工具结果，本地文件工具也受仓库边界、符号链接、保护文件和内置忽略目录规则约束；但对于外部 MCP、发消息、创建工单、付费等副作用，尚没有统一的幂等、`UNKNOWN` 确认和补偿协议。

生产级方案可以这样设计：

1. 每个副作用调用生成稳定的 `idempotency_key`，例如 `run_id:tool_call_id`。
2. 执行前持久化 Intent，包括工具、规范化参数、目标资源和审批记录。
3. 返回成功后持久化 Result 与外部资源 ID。
4. 如果超时或连接中断，状态标为 `UNKNOWN`，不能直接当作失败重试。
5. 先用幂等键或外部资源 ID 查询执行结果；确认未执行后才能重试。
6. 无法确认时进入人工复核；多步骤副作用使用 Saga，为已成功步骤定义补偿动作。

因此，面试中的准确结论是：当前 HITL 已覆盖授权、参数修改、审计和本地静态保护，但尚未把外部副作用统一纳入事务式恢复。下一步优先级应高于继续增加更多 Prompt 风险规则。

代码：[审计记录字段](src/my_agent/hitl/audit.py#L11)、[执行并审计](src/my_agent/hitl/registry.py#L343)、[Tool 结构化结果](src/my_agent/tools/execution.py#L40)、[批次超时结果](src/my_agent/tools/registry.py#L393)

### 7.2 分层上下文管理

#### 简历追问 5：你说实现了“分层上下文管理与 Token 预算裁剪”，预算具体是如何计算的？

可以把 Prompt 分成五类预算：固定指令和当前输入、仓库上下文、工具 Schema、长期经验、短期会话。核心不是给每层写死一个 Token 数，而是先确定本次模型的安全 Prompt 上限，再动态分配剩余空间。

当前主要公式是：

```text
compression_trigger
  = context_window - response_reserve - compression_buffer

memory_budget
  = max(0, compression_trigger - fixed_tokens)

long_term_limit
  = min(profile_memory_limit, max(500, memory_budget * 3%), memory_budget)

short_term_allowed
  = max(0, compression_trigger - fixed_with_long_term_memory)
```

`fixed_tokens` 包括基础消息和当前纳入的 Tool Schema。长期经验注入后会重新估算 `fixed_with_memory_tokens`，再给短期记忆分配剩余空间。因此 `short_term_allowed` 是每次请求动态变化的，不是一个固定配置。

默认配置通过 `ContextProfile.resolve()` 推导 128K 窗口：响应预留约 10%，压缩缓冲约 5%，实际压缩触发点是 108,800 Token；仓库上下文预算约为窗口的 12%，Tool Schema 预算约为 8%。核心工具会优先完整保留，非核心工具按完整 Schema 省略，不会把一个 JSON Schema 截断一半。

最后还有硬保护：即使压缩和预算裁剪后仍达到触发阈值，也不会继续把超长 Prompt 发给模型，而是记录 `context.over_budget` 并抛出 `ContextOverBudgetError`。

代码：[动态 Context Profile](src/my_agent/context/profile.py#L44)、[压缩触发公式](src/my_agent/context/profile.py#L215)、[长期记忆预算](src/my_agent/context/profile.py#L158)、[消息预算计算](src/my_agent/context/manager.py#L45)、[Tool Schema 预算](src/my_agent/context/tool_budget.py#L40)

#### 简历追问 6：上下文压缩如何保证不会丢掉用户约束、文件修改和失败信息？

当前短期记忆压缩使用按用户 Turn 分组的 Map-Reduce：旧消息先分块，每个 Map Prompt 明确要求保留用户目标、限制、工具结果、修改文件、测试结果、错误、未解决问题和技术决策；多个摘要再去重合并。最近若干用户 Turn 不参与压缩，避免刚发生的工具调用和上下文被过早概括。

如果摘要 LLM 调用失败，系统会退化为确定性的文本拼接摘要，并在 `fallback` 字段中留下证据；旧消息只有在得到非空摘要后才会被替换。压缩前后 Token、Map 数量、是否 Reduce、是否 Fallback 都会进入 Trace。

但这仍然不能等价于“保证关键信息不丢失”。当前保护主要依赖摘要 Prompt，没有独立的结构化关键事实表，也没有自动验证摘要是否覆盖所有约束。更稳妥的方案是把上下文拆成两条路径：

- 自然语言摘要用于模型理解。
- `CriticalFacts` 用结构化字段保存目标、硬约束、已完成步骤、失败步骤、外部资源 ID、审批决定、修改文件和测试状态。

压缩后运行确定性 Coverage Check，验证关键字段仍存在，并为每个事实保存来源 Entry ID。这样即使摘要表达发生漂移，恢复和安全判断仍然基于结构化事实，而不是再次相信摘要模型。

代码：[Map-Reduce 压缩 Prompt](src/my_agent/memory/short_term/compression.py#L12)、[压缩与 Fallback](src/my_agent/memory/short_term/compression.py#L48)、[保留最近 Turn](src/my_agent/memory/short_term/store.py#L61)、[压缩替换和 Trace](src/my_agent/memory/manager.py#L541)

#### 简历追问 7：Context、Agent State、短期 Memory 和长期 Memory 有什么区别？为什么不能放在一个消息列表里？

四者的职责不同：

- **Context** 是本轮真正发送给模型的派生视图，强调“此刻推理需要什么”。
- **Agent State** 是任务状态机事实，例如目标、计划、步数、工具历史、停止原因和最终答案，强调“任务执行到了哪里”。
- **短期 Memory** 是会话内消息和工具结果，强调时间顺序和最近交互。
- **长期 Memory** 是跨任务复用的 Trajectory、Tip、Skill、Tool 经验，强调检索、归因、治理和持久化。

如果全部混在一个消息列表里，会出现三个问题：恢复任务必须重放大量无关文本；摘要错误可能同时破坏状态和审计；跨任务经验会不断挤占当前对话。

当前 AgentCli 在 Prompt 构建时先注入长期经验，再拼接预算内的短期消息；Formal Evolver 每个任务只做一次候选检索和选择，任务中使用固定的选中上下文，避免每轮检索导致实验状态漂移。子任务 `fork_for_task()` 会创建空的短期 Store，但共享同一个 Experience Store，从而实现任务内上下文隔离和项目级经验共享。

不足是主 Agent State 还没有统一成为跨进程持久化 Checkpoint。因此可以说当前已经分离了逻辑层次，但还没有完整实现“Durable State + Derived Prompt + Append-only Trace”的恢复架构。

代码：[Context 组装](src/my_agent/context/manager.py#L87)、[长期经验注入位置](src/my_agent/context/manager.py#L223)、[短期与长期 Memory Manager](src/my_agent/memory/manager.py#L50)、[任务级 Memory Fork](src/my_agent/memory/manager.py#L485)、[Formal 每任务一次选择](src/my_agent/memory/evolver/runtime/formal.py#L120)

#### 简历追问 8：你如何证明 Prompt 变短后效果没有下降？

不能只展示压缩前后 Token 数。完整评测至少应包含四层指标：

1. **成本层**：输入 Token、压缩比例、摘要调用次数、延迟和费用。
2. **信息保持层**：用户硬约束召回率、修改文件召回率、测试状态召回率、未解决错误召回率。
3. **行为层**：重复 Tool Call、错误文件修改、违反约束、无信息增益循环和 Tool 参数错误。
4. **任务层**：测试通过率、最终任务成功率、人工接管率和恢复成功率。

实验应固定模型、温度、工具、任务顺序和最大步数，比较：不压缩、仅最近窗口、当前 Map-Reduce、结构化事实加摘要。长任务要按 Prompt 长度分桶，因为短任务几乎不会触发压缩，混在一起会掩盖差异。

当前代码已经记录 `before_tokens`、`after_tokens`、`fallback`、Memory 命中数、Tool Schema 占用和是否 Over Budget，可以作为成本和触发证据；但还没有自动计算“关键事实保持率”。所以简历中的“在保证关键信息保留的前提下”最好在面试中解释为设计目标，并补充你实际使用的抽样核验或离线评测，不能只用 Prompt 变短来证明。

代码：[压缩 Trace](src/my_agent/context/manager.py#L131)、[Prompt 准备 Trace](src/my_agent/context/manager.py#L178)、[Over Budget Guard](src/my_agent/context/manager.py#L166)

### 7.3 SFT 与工具协议对齐

#### 简历追问 9：你的 SFT 数据是怎么从 Agent 轨迹构造出来的？只使用成功轨迹有什么问题？

当前 `traces_to_sft()` 会遍历 Trace JSONL，先要求该 Trace 中存在 `benchmark_result.status == "passed"`，再关联 `tool.started` 和 `tool.completed`。只有成功完成的非 `finish` 工具调用会被写成监督样本；样本输入包含任务、计划和最近六条工具历史，输出包含工具名、参数和原因。

这样做的优点是标签来自真实 Runtime 协议，并且过滤掉最终任务失败的 Trace，能快速得到格式较干净的工具调用数据。但它有三类偏差：

1. **成功轨迹不等于每一步都最优**：通过测试的轨迹中仍可能包含多余读取、偶然成功或低效调用。
2. **成功样本偏差**：模型看不到参数错误、未知工具、审批拒绝、超时和恢复决策，容易学会“总要调用一个工具”，却不会在不确定时修复参数或停止。
3. **状态截断**：固定保留最近六条历史可能缺失更早的关键约束，长链任务尤其明显。

改进时应把数据分成：高质量成功调用、失败后成功修复、正确拒绝/停止、HITL 修改、工具不可用时降级，以及高相似度错误调用的对比数据。每条样本还应保存 Trace ID、任务 ID、模型版本、Tool Registry 版本和自动验证结果，避免把“最终通过”直接当成每一步的金标签。

代码：[Trace 转 SFT](src/my_agent/data/builders.py#L316)、[成功 Trace 判定](src/my_agent/data/builders.py#L426)、[SFT Sample 校验](src/my_agent/data/sft_samples.py#L1)

#### 简历追问 10：当前 Train/Validation 切分有什么数据泄漏风险？你会怎么改？

当前 Alpaca 导出器把所有行汇总后用固定随机种子打乱，再按比例切分。这个过程可复现，但切分单位是“样本行”。同一条 Trace 中连续的多个工具调用、同一仓库的近似任务或同一个 Bug 的不同步骤，可能同时出现在训练集和验证集，导致验证指标虚高。

更合理的切分单位应该是 Group，而不是 Row：

```text
group_id = repository_id + task_id
```

至少要保证同一个 `trace_file` 全部进入同一个 Split；更严格时按仓库切分，验证模型是否能迁移到未见过的代码结构。还应使用规范化输入/输出 Hash 做近重复去重，并单独保留：

- IID Validation：同分布新任务。
- OOD Tool Validation：未见参数组合、更多工具或不同 Schema。
- Long-horizon Validation：更长历史和多步恢复。
- Safety Validation：危险参数、路径逃逸、审批拒绝和未知状态。

因此，面试时可以主动指出：固定 Seed 解决的是可复现性，不解决数据独立性；若 95% 指标来自行级随机切分，需要重新用任务级或仓库级 Split 复核。

代码：[Alpaca 导出与随机切分](src/my_agent/data/converters.py#L75)、[Shuffle 和 Split](src/my_agent/data/converters.py#L142)、[来源统计](src/my_agent/data/converters.py#L247)

#### 简历追问 11：为什么选择 Qwen3.5-4B、BF16 和 LoRA？这些训练参数是如何确定的？

当前训练脚本固定 Qwen3.5-4B 及具体 Revision，并使用 `qwen3_5_nothink` 模板。LoRA 配置为 Rank 16、Alpha 32、Dropout 0，只训练 `q_proj/k_proj/v_proj/o_proj`；默认 Micro Batch 为 1、梯度累积 16、Gradient Checkpointing 开启、学习率 `2e-5`、1 个 Epoch、Cutoff Length 8192，精度使用 BF16。

选择逻辑可以这样解释：

- 4B 模型在单卡实验中比更大模型更容易迭代，同时具备原生工具调用模板。
- LoRA 只更新低秩 Adapter，显著降低可训练参数、梯度和优化器状态开销。
- BF16 与 FP16 占用相近，但指数范围接近 FP32，通常更不容易因为数值范围产生 Overflow；前提是 GPU 支持 BF16。
- Gradient Accumulation 提升有效 Batch，但不会把单步激活显存乘以 16；Gradient Checkpointing 用计算换激活显存。

这些参数应称为当前实现的默认合同，不应说成已经证明最优。实际选择还需要学习率、Rank、Cutoff、Epoch 的小规模 Sweep，并同时观察协议指标、任务成功率、过拟合和峰值显存。

一个重要实现细节是 SFT 和 OPD 共用 Adapter 结构合同。SFT 训练完成后会保存训练 Manifest；注册为 M0 时再次校验 Base Model、Revision、模板、LoRA 配置 Hash 和 Adapter Artifact Hash，避免后续 OPD 在不兼容的初始化上训练。

代码：[SFT 默认训练参数](scripts/train_llamafactory_lora.sh#L6)、[LoRA 合同校验](scripts/train_llamafactory_lora.sh#L89)、[训练命令](scripts/train_llamafactory_lora.sh#L101)、[SFT 注册为 OPD M0](src/my_agent/training/sft_registration.py#L53)

#### 简历追问 12：简历中的“结构化输出有效率 95%”和“Runtime 工具调用解析成功率 95%”分别是什么？

这两个指标必须分开：

- `json_valid_rate`：模型原始输出能否被严格解析成顶层 JSON Object。它只证明语法和顶层类型有效。
- `runtime_tool_call_parse_rate`：对 Tool Call 任务，当前 Runtime 的 `parse_tool_calls()` 能否从输出中得到至少一个工具调用。解析器支持标准 JSON/Tool Call，也兼容 Qwen3.5 XML 风格。

当前评测还提供 `field_hit_rate`、`tool_accuracy`、`file_mention_rate` 和 ROUGE-L。它们分别回答字段是否齐全、工具名是否和参考答案一致、是否提到目标文件以及文本相似度。

95% 的解析率不等于 95% 的工具调用正确率，更不等于 95% 的任务成功率。一个调用可以被成功解析，但工具名选错、参数不合法、权限不足或执行后结果不满足任务。面试时应明确给出：

```text
样本量 N
Tool Call 子集大小
Train/Validation/Test 切分方式
Base 与 SFT 使用的模型 Revision、模板和解码参数
95% 对应的精确指标名
失败样本分类
置信区间或至少分子/分母
```

如果暂时拿不出原始结果文件，最稳妥的表述是“在自建离线协议评测集上，简历记录的严格 JSON 有效率和 Runtime 解析率均约为 95%；该指标只衡量协议对齐，端到端任务效果需要另行评测”。

代码：[SFT 协议指标定义](src/my_agent/evaluation/protocol_metrics.py#L18)、[单样本评分](src/my_agent/evaluation/protocol_metrics.py#L132)、[聚合口径](src/my_agent/evaluation/protocol_metrics.py#L153)、[Runtime Tool Call 解析](src/my_agent/evaluation/protocol_metrics.py#L236)、[Qwen3.5 Tool Call 解析](src/my_agent/policy/transformers_policy.py#L602)

### 7.4 OPD-Evolver 与四角色联合训练

#### 简历追问 13：OPD 和普通 SFT 的本质区别是什么？为什么 Teacher 和 Student 使用同一段 Completion？

普通 SFT 学习一个固定目标序列：给定输入 (x)，最大化人工或轨迹标签 (y) 的概率。当前 OPD 实现则是 On-Policy Distillation：

- Student 只看到部署时可见的 Public View。
- Teacher 使用同一个当前模型，但在 Public View 后追加 Privileged Hindsight。
- Completion 不是由 Teacher 另写的答案，而是当前 Student 在线采样出的同一段 Token Prefix。
- 在这段相同 Completion 的每个位置，最小化 `KL(p_teacher || q_student)`。

使用同一 Completion 的原因是要比较“面对相同已经发生的决策前缀，有无事后证据时模型对下一 Token 的分布差异”。如果 Teacher 重新生成另一条答案，就无法逐 Token 对齐，训练会退化成另一种离线蒸馏或 SFT，也更容易把不可部署的事后答案直接泄漏给 Student。

Collator 会分别拼接 Student Prompt 和 Teacher Prompt，但在二者后面追加完全相同的 Student Completion，并使用同一个 `assistant_loss_mask`。Trainer 用同一个共享模型做两次 Forward：Teacher 分支在 `no_grad` 下运行，Student 分支保留梯度；只有 Student 分支通过共享 LoRA Adapter 更新。

为降低显存，损失函数不一次物化完整的 `[B, S, V]` Logits，而是保存 Completion Hidden State，再按 Vocabulary Chunk 投影并计算精确的 Full-Vocabulary KL。它不是 PPO、DPO 或 GRPO，因为没有单独的 Reward Policy Gradient 或偏好对。

代码：[Public/Teacher Prompt 构造](src/my_agent/training/opd_rollout.py#L305)、[Student/Teacher 同 Completion 对齐](src/my_agent/training/opd_collator.py#L25)、[双分支 Forward](src/my_agent/training/opd_trainer.py#L482)、[Chunked Full-Vocabulary KL](src/my_agent/training/opd_loss.py#L95)

#### 简历追问 14：Selection、Action、Writing、Maintenance 四类任务分别看什么数据？是四个模型还是一个模型？

四类任务对应经验记忆生命周期中的四种决策，但使用同一个 Base Model 和同一个共享 LoRA Adapter，通过 Role Prompt 和类型化视图区分任务：

| 角色 | Student 可见的 Public View | Teacher 额外看到的 Hindsight | 学习目标 |
|---|---|---|---|
| Selection | 当前任务、候选 Memory、Token Budget | 每个候选的后验价值证据 | 选择真正有用且预算内的经验 |
| Action | 任务、工具、部署可见的消息前缀 | 正价值 Memory 与成功轨迹 | 在看不到私有答案的前提下改进 Tool/Action 决策 |
| Writing | 任务、完整轨迹、Reward、已选 Memory ID | 写入 Memory 的后验价值 | 只沉淀可复用且未来有效的经验 |
| Maintenance | Repository Snapshot、历史 Outcome、维护工具 | 错误、低价值和冗余诊断 | 合并、删除或保留 Memory，控制长期膨胀 |

最重要的安全边界是 Action Leakage。快速运行轨迹可以作为证据，但不能直接把已选 Memory 内容或官方答案放进 Action Student Prompt。代码会检查 Forbidden Marker、选中 Memory ID 和内容是否泄漏；Teacher 可以看到 Hindsight，但梯度只通过 Student 分支更新。

运行时四种决策有先后顺序，但训练时不是“先训练 Selection，再训练 Action”。`RoleSampler` 按角色权重从四个 Bucket 中混合采样，联合更新同一个 Adapter，同时在 Checkpoint Manifest 中分别记录各角色样本数、Token 数、KL 和梯度统计。这样能共享底层工具协议与领域表示，同时保留角色级可观测性。

代码：[四类 Public/Hindsight Schema](src/my_agent/training/role_views.py#L476)、[Action Leakage Guard](src/my_agent/training/opd_rollout.py#L346)、[四角色 Dataset 校验](src/my_agent/training/opd_dataset.py#L40)、[Role-Balanced Sampler](src/my_agent/training/opd_dataset.py#L158)、[共享 Adapter 配置](src/my_agent/training/opd_trainer.py#L34)

#### 简历追问 15：你所说的 OPD-Evolver“快慢双循环”具体是什么？和 M0→D0→M1→D1→M2 有什么关系？

这里最好主动区分三种时间尺度，避免把概念混在一起：

1. **任务内快速循环**：任务开始时进行候选召回与 Selection，Agent 执行 Action；权威 Evaluator 给出 Outcome 后，Writer 决定是否写入经验。这一层每个任务都会发生。
2. **Memory Repository 慢维护**：不是每个任务都全库整理，而是按 Cadence 触发。当前默认每 30 个任务执行一次 Maintenance，对长期 Memory 做查询、合并、删除和完成确认，降低重复和错误积累。
3. **策略训练外循环**：冻结 M0 执行任务并采集 D0，用 D0 训练得到 M1；再由 M1 重新在线采集 D1，用 D1 训练得到 M2。主实验不把 D0 Replay 到第二轮，确保每轮数据来自当前策略。

因此，快慢循环不仅是“一个 Agent 跑得快、另一个 Agent 跑得慢”。核心是：运行时以任务为单位生成选择、执行和写入证据；周期性维护外部经验库；离线再将当前 Checkpoint 产生的证据转成 Learner Dataset，更新同一个策略 Adapter。

当前实现还做了两个重要约束：每个任务只允许一次 Formal Selection；Writer 只接受最终化的权威 Outcome，并检查 Repository Revision，防止基于过期快照写入。Recollection 又验证 Dataset/Checkpoint Identity、Hash、隔离路径和 `dataset_consumption`，避免把不同轮次的数据混用。

代码：[任务开始与一次 Selection](src/my_agent/memory/evolver/coordinator.py#L253)、[Outcome 后写入](src/my_agent/memory/evolver/coordinator.py#L341)、[Maintenance Cadence](src/my_agent/memory/evolver/coordinator.py#L446)、[两轮 Recollection](src/my_agent/training/recollection.py#L125)、[单轮 Dataset 构造](src/my_agent/training/collection_round.py#L53)

#### 简历追问 16：如何证明 OPD 真正提升了 Memory 使用效果，而不是只让四种输出更容易解析？

需要把“协议能力”和“任务收益”分开评测。

第一层是角色协议评测：

- Selection：决策成功率、Schema 有效率、未知候选标签率。
- Action：决策成功率、Runtime Tool Call 解析率、未知工具率。
- Writing：JSON Array 率、Validator 接受率、非空 Proposal 率。
- Maintenance：Tool Call 解析率、未知 Memory ID 率。

这些指标能证明模型学会了四类接口，但不能证明 Memory 有用。

第二层是固定 Held-out Protocol 的四臂对照：

| 对照 | 回答的问题 |
|---|---|
| A：M0 + No Memory | 初始策略在无长期经验时的基线 |
| B：M0 + Memory | 不训练策略时，Memory 本身带来多少收益 |
| C：Trained + Memory | OPD 与 Memory 共同作用后的结果 |
| D：Trained + No Memory | 策略参数本身是否提升，是否只是脱离 Memory 也变强 |

其中 `C-B` 更接近 OPD 在 Memory 场景中的增量，`C-D` 反映训练后对 Memory 的使用价值，`B-A` 反映原始 Memory 机制的价值，`D-A` 则是脱离 Memory 的策略变化。各 Arm 必须固定任务顺序、最大步数、Token Budget、工具、Evaluator 和解码参数，并隔离 Repository、Ledger 和输出路径。

第三层才是 Memory 因果评测：增加 Teach→Transfer、`shuffled_memory`、等 Token 的 Flat Memory、Leave-One-Tier-Out，以及 No Attribution、Similarity Only、No Writing、No Maintenance 等消融。报告任务成功率之外，还要看负收益率、错误 Memory 使用率、Token/延迟、Memory 增长、重复率和 Bootstrap 置信区间。

当前仓库已经实现角色协议指标、评测矩阵、隔离和 Readiness Check，但代码存在不等于实验已经完成。简历使用“尝试使用 OPD 提升 Memory 使用效果”是合理边界；只有拿出 M0/M1/M2 Checkpoint、Dataset Manifest、各 Arm 原始结果和统计报告后，才适合改成确定的提升结论。

代码：[四角色协议阈值](src/my_agent/evaluation/formal_role_protocol.py#L26)、[四角色协议评分](src/my_agent/evaluation/formal_role_protocol.py#L42)、[Held-out 四臂矩阵](src/my_agent/evaluation/opd_evaluation.py#L172)、[评测隔离校验](src/my_agent/evaluation/opd_evaluation.py#L511)、[数值复现 Readiness](src/my_agent/evaluation/opd_evaluation.py#L352)

### 7.5 面试前建议准备的证据

为了让以上回答不仅“讲得通”，还可以被继续追问验证，建议提前准备四组最小证据：

1. **HITL Demo**：一次 READ 自动放行、一次 WRITE 修改参数、一次 EXECUTE 拒绝，以及对应审批事件和 Audit JSONL。
2. **Context Demo**：同一长任务压缩前后的 Token、关键事实核对、最终测试结果和 `memory.compacted` Trace。
3. **SFT Eval**：Base/SFT 使用同一测试集的 `metrics_summary.json`、样本分组方式、分子分母、至少 10 条失败 Case，以及实际训练 Manifest。
4. **OPD Evidence**：M0/D0/M1/D1/M2 的 Identity/Hash 链、四角色样本分布、Role KL、评测矩阵原始结果和消融结果。若还没有完整训练产物，就准备方法 Smoke Test，并明确它只证明链路正确。

这四组材料能把回答从“知道概念”提升为“能够解释实现、指标和证据边界”。
