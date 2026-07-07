# HebutGuide 项目七个技术亮点

本文结合项目代码，对 HebutGuide 招生咨询 Agent 的七个核心亮点做简要总结。重点说明每个能力在当前项目中的落点、运行逻辑和实现方案。

## 1. 端到端意图识别

项目中的意图识别入口是 `core/intent_recognizer.py` 的 `IntentRecognizer.recognize()`，主链路由 `api/main.py` 在构建 Orchestrator 请求时调用，也会被 `AgentOrchestrator` 在缺少预识别结果时兜底调用。

实现逻辑是“三路融合”：

- LLM 语义识别：`_llm_recognize()` 通过招生咨询 Few-shot、最近对话历史和意图定义，让模型输出 `intent/confidence/reasoning`。
- 向量相似度识别：`_embedding_recognize()` 用模板样例做相似度匹配。当前代码优先尝试客户端 embedding 能力，不可用时退化为本地字符 n-gram 哈希向量，保证链路可运行。
- 关键词模式识别：`_pattern_recognize()` 用招生场景关键词做低延迟兜底，例如“分数线、位次、稳不稳”映射到 `score_risk`。

融合方案由 `_vote()` 完成：LLM 为主权重，向量和关键词为辅助；如果使用兼容接口导致 embedding 不可用，则自动调整权重。识别后还会通过 `_extract_entities()` 提取省份、科类、分数、位次、专业等实体，并用 `_urgency()` 判断紧急度。这样 `/chat` 不只是分类文本，而是拿到“意图 + 实体 + 紧急度”，后续可直接服务于工具调用和 Agent 路由。

## 2. MCP 工具调用框架 + RAG 知识库

工具框架在 `mcp/tool_manager.py`，知识库在 `mcp/knowledge_base.py`，风险评估工具在 `mcp/admissions_tools.py`。`api/main.py` 的 `_create_tool_manager()` 将 `knowledge_search` 和 `risk_assessment` 两个工具注册到 `MCPToolManager`。

MCP 工具调用方案包括：

- 统一注册：每个工具用 `Tool` 描述名称、schema、handler、缓存时间、超时、fallback 和是否支持 rerank。
- 统一调用：`MCPToolManager.call()` 执行缓存检查、熔断检查、参数校验、超时控制、异常 fallback、统计更新。
- 检索优化：`search_with_rewrite()` 先用 LLM 将查询改写为多个子查询，再并行调用检索工具，合并去重后用 `_rerank()` 重排。
- 稳定性保障：工具层内置 TTL cache、CircuitBreaker、fallback result，避免工具异常直接打断对话。

RAG 知识库方案是：`KnowledgeBase` 使用 ChromaDB collection `hebut_admissions_kb` 存储招生知识文档，导入时按 500 字左右切片，查询时用 ChromaDB 语义检索返回标题、片段、相似度和 chunk 信息。`api/main.py` 的 `_build_knowledge_context()` 会按意图判断是否需要检索，检索后把结果格式化为 `[知识库检索结果]` 注入 Agent 背景。

结构化风险评估和 RAG 分工明确：RAG 负责章程、专业、校园生活等文本事实；`risk_assessment` 负责分数、位次、专业历史数据的规则计算。`_build_risk_context()` 会把工具返回的 `status/risk_level/history/reason/suggestion` 转成 `[MCP 录取风险评估结果]`，并明确约束 Agent 不得编造录取结论。

## 3. 三级记忆管理

记忆模块在 `memory/conversation_memory.py`，由 `api/main.py` 的 `/chat` 链路读取和写入。核心入口是 `MemoryManager.get_context()` 和 `MemoryManager.add_message()`。

三级记忆分别是：

- 工作记忆：当前会话最近消息存 Redis list，key 为 `wm:{user_id}:{conv_id}`，TTL 为 24 小时。
- 情景记忆：压缩后的历史对话摘要存 ChromaDB collection `episodic`，读取时按当前 query 做语义检索。
- 用户画像：从对话中提炼出的省份、科类、分数、位次、目标专业、偏好等信息存 ChromaDB collection `user_profile`。

运行方案是：`/chat` 收到业务咨询后先调用 `get_context()`，拿到最近对话、相关历史、用户画像和摘要，再通过 `MemoryContext.to_prompt_text()` 变成 Agent 背景。回复完成后，`add_message()` 写入用户消息和助手消息；如果消息包含明确报考信息，`api/main.py` 会异步触发 `update_profile()`，用 LLM 提炼并合并用户画像。

为避免上下文无限增长，`add_message()` 在 Redis 消息数达到 `COMPRESS_AT` 后触发 `_compress()`：旧消息被 LLM 总结，摘要写回 Redis，旧对话摘要写入 ChromaDB，工作记忆只保留最近 5 条。这样项目同时保留短期上下文、跨会话历史和长期报考画像。

## 4. 多 Agent 路由与编排

多 Agent 编排位于 `agents/agent_orchestrator.py`。项目定义了 `GeneralAdmissionsAgent`、`PolicyAgent`、`RiskAgent`、`PlanningAgent` 四类实际处理 Agent，并保留 `ESCALATION` 类型作为升级标记。

主流程在 `AgentOrchestrator.run()`：

1. 如果请求没有预置意图，则调用 `IntentRecognizer` 识别。
2. `_collaboration_targets()` 判断是否为多维问题，例如同时涉及分数、专业选择、就业、宿舍时触发并行协作。
3. 单 Agent 场景由 `_route()` 根据意图和紧急度选 Agent 类型。
4. `_execute()` 选择同类中 `routing_score()` 最高的 Agent 执行；失败时降级到 General。
5. 根据内容、紧急度或 escalation 意图标记 `escalated`。

项目的路由方案不是简单 if/else，而是“意图路由 + 性能路由 + 降级路由”。`AgentStats.routing_score()` 综合成功率、平均延迟和 Monitor 写入的 `monitor_penalty`，后续如果同类 Agent 横向扩展，Orchestrator 可以自动选择表现更好的实例。

并行协作也有边界控制：默认只路由给一个主 Agent，只有用户明确问多个维度时才 `run_parallel()`。这避免了招生咨询回答变得冗长重复，同时保留复杂问题拆分处理的能力。

## 5. 动态 Skills 加载与 Prompt 注入

Skills 加载器在 `core/skill_loader.py`，技能文档位于 `skills/*/SKILL.md`。`api/main.py` 启动时创建 `SkillManager`，读取 `HEBUTGUIDE_SKILLS_DIR`，并暴露 `/skills` 和 `/skills/reload` 用于查看与热加载。

实现方案是：

- 发现文件：`SkillManager._discover_files()` 优先扫描标准 `SKILL.md`，也支持 `.md/.txt/.json`。
- 解析元数据：Markdown 顶部 front matter 提供 `name/description/keywords/agents/enabled`。
- 请求匹配：`Skill.matches()` 同时判断 Agent 类型和用户消息关键词。
- Prompt 注入：`prompt_for()` 把命中的 Skill 格式化为规则块，并按 `max_prompt_chars` 控制总长度。

注入发生在 Orchestrator 内部：`_execute()` 调用 `_with_skills()`，针对当前 Agent 类型生成专属 `skills_context`，再由 `BaseAgent._call_llm()` 拼到 system prompt 后面。因此同一个用户问题在 RiskAgent、PolicyAgent、PlanningAgent 中可以拿到不同业务规则，避免所有规则混在一个大 prompt 里。

当前内置 Skills 覆盖通用接待、招生政策、分数位次风险、专业规划、官方升级。它们把招生业务口径从代码里抽出来，便于运营或业务人员调整规则，也能通过 `/skills/reload` 无重启生效。

## 6. Monitor 闭环监控与自动降权

监控模块在 `monitor/performance_monitor.py`，由 `api/main.py` 在服务启动时创建并 `start()`。它周期性读取 `AgentOrchestrator.get_stats()` 和 `MCPToolManager.get_stats()`，不需要在业务代码里额外埋点。

监控逻辑包括：

- 实时采集：Agent 统计包括 total、success_rate、avg_ms、monitor_penalty、routing_score；工具统计包括 success_rate、avg_latency_ms、consecutive_fails、circuit_state。
- 阈值告警：成功率低、延迟高会生成 `Alert`，可选通过 webhook 发送。
- 异常检测：`AnomalyDetector` 使用滑动窗口 Z-score 识别指标突变。
- 优化建议：连续失败、Agent 成功率偏低会生成可操作 `Suggestion`。
- Prometheus：配置端口后可暴露 Gauge、Histogram、Counter。

闭环体现在 `_collect()` 的路由反馈：Monitor 根据成功率和延迟计算 `_routing_penalty()`，再调用 `orchestrator.update_routing_penalties()` 写回每个 Agent 的 `monitor_penalty`。Orchestrator 后续 `_best_agent()` 选路由时会使用新的 `routing_score()`，从而自动降低表现差的 Agent 权重。

当前项目每类 Agent 只有单实例，但这套设计已经为多实例扩展预留了自动降权机制：单实例时体现为监控解释和告警，多实例时就能直接变成在线流量调度。

## 7. 端到端 Agent 评测

评测框架在 `evaluation/evaluator.py`，API 入口是 `api/main.py` 的 `/eval/run`。评测器启动时接入真实 Orchestrator 和 Recognizer，因此评测的不是孤立函数，而是接近线上 `/chat` 的完整 Agent 链路。

评测方案分四层：

- 意图识别评测：`IntentEvaluator.evaluate()` 用标注用例计算 accuracy、macro-F1 和逐类 precision/recall/F1。
- 对话质量评测：`EndToEndEvaluator._evaluate_dialog_case()` 构造单轮或多轮对话，请 Orchestrator 生成真实回复。
- LLM-as-Judge：`LLMJudge.judge()` 从相关性、准确性、完整性、有用性四个维度打分，并对“假设工具返回”“XXX/YYY”等伪造数据做硬性扣分。
- 回归检测：`_detect_regressions()` 与历史 baseline 比较，发现超过 5% 的退化；`_save_baseline()` 将报告保存为后续基线。

一个关键细节是评测上下文复用主链路能力。`api/main.py` 在创建 `EndToEndEvaluator` 时传入 `eval_context_builder`，内部同样调用意图识别、知识库检索和风险工具上下文构建。这样评测不仅检查模型回答本身，也检查“意图识别、RAG、MCP、路由、Prompt 约束”组合后的端到端效果。

## 总体链路

项目整体链路可以概括为：

```text
用户请求
  -> FastAPI /chat
  -> MemoryManager 读取三级记忆
  -> IntentRecognizer 识别意图、实体和紧急度
  -> MCPToolManager 按需调用 RAG 检索和风险评估工具
  -> AgentOrchestrator 选择单 Agent 或多 Agent 协作
  -> SkillManager 按 Agent 和关键词注入业务规则
  -> Agent 基于记忆、知识库、MCP 结果和 Skills 生成回复
  -> MemoryManager 写入对话并异步更新画像
  -> PerformanceMonitor 采集指标并反馈路由降权
  -> EndToEndEvaluator 用测试集和 LLM Judge 做回归评测
```

这七个亮点不是彼此独立的模块，而是在 `/chat` 主链路中串成闭环：识别决定路由，路由决定 Agent，工具和 RAG 提供事实依据，记忆提供个性化上下文，Skills 注入业务口径，Monitor 反馈在线表现，Evaluator 检查整体质量。
