# HebutGuide Skills 文档

HebutGuide 启动时会从 `HEBUTGUIDE_SKILLS_DIR` 读取 Skills，并在匹配用户请求时注入到对应 Agent 的 system prompt。Skills 适合维护招生咨询口径、信息澄清流程、录取风险边界、专业规划建议和官方升级规则。

系统主链路：

```text
用户请求
  -> api/main.py
  -> MemoryManager 读取上下文
       - Redis 工作记忆
       - ChromaDB 情景记忆 episodic
       - ChromaDB 用户画像 user_profile
  -> IntentRecognizer 三路融合识别意图
  -> MCPToolManager 检索 knowledge_base 并构建知识上下文
  -> AgentOrchestrator 路由到专业 Agent
  -> SkillManager 按 Agent 类型和关键词筛选 Skills
  -> Agent 基于记忆 + 知识库 + Skills 调用 LLM 生成回复
  -> MemoryManager 写入记忆并异步更新用户画像
  -> PerformanceMonitor 采集指标并反馈路由降权
```

当前内置五类 Skills：

```text
skills/general_admissions/SKILL.md    # 通用招生咨询：接待、澄清、学校概况、校园生活
skills/policy_admissions/SKILL.md     # 招生政策：章程、录取规则、调剂、退档、收费
skills/score_risk/SKILL.md            # 分数位次：历年数据、冲稳保、风险边界
skills/planning_admissions/SKILL.md   # 专业规划：专业对比、就业升学、志愿搭配
skills/official_escalation/SKILL.md   # 官方确认：招生办、政策未发布、人工升级
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 分数位次风险分析规范
description: 适用于 RiskAgent 的历年录取数据解释、位次判断和冲稳保分析规范
keywords: 分数,位次,录取线,最低分,稳不稳,能报吗,冲稳保
agents: risk
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，可填 `general`、`policy`、`risk`、`planning`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把政策、风险、规划、校园生活规则混在一个文件里。
- 必须包含“适用场景”“处理流程”“必须追问的信息”“升级条件”“禁止事项”等稳定章节。
- 对考生隐私和敏感信息保持克制，不要求身份证号、准考证号、完整手机号等非必要信息。
- 对录取概率、招生计划、收费、转专业等无法保证的事项使用保守措辞，例如“倾向于”“需要结合位次”“以当年官方发布为准”。
- 对政策未发布、数据缺失、特殊类型招生、投诉和最终录取结论等场景要明确建议联系招生办公室或省级考试院确认。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
