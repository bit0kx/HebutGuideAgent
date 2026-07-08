"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAdmissionsAgent

并行协作：
  - 默认只路由到一个主 Agent；只有明确多维问题才并行协作
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低或涉及官方口径 → 建议联系招生办公室或查看官方招生网
  路由到 escalation 时，最终会降级到 GeneralAdmissionsAgent，同时 result.escalated=True。
  这是合理的，因为当前系统没有人工坐席或工单系统，只做“升级标记 + 官方渠道提示”。
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_client import create_llm_client

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL = "general"     # 兜底、学校概况、校园生活、简单问答
    POLICY = "policy"       # 招生章程、录取规则、调剂、退档、转专业、收费
    RISK = "risk"           # 分数、位次、历年录取线、冲稳保分析
    PLANNING = "planning"   # 专业介绍、专业对比、就业升学、志愿搭配
    ESCALATION = "escalation"

@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    skills_context: str = ""     # 来自 SkillManager 的动态业务规则
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    def __init__(self, client: Any, model: str):
        self._client = client
        self._model  = model
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        system_prompt = self.system_prompt
        if req.skills_context:
            system_prompt = f"{system_prompt}\n\n{_clean(req.skills_context)}"

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens(req),
            system=system_prompt,
            messages=messages,
        )
        return resp.content[0].text

    def _max_tokens(self, req: Request) -> int:
        """简单介绍类问题限制输出长度，减少延迟和过度展开。"""
        msg = (req.message or "").strip().lower()
        brief_markers = ["简单", "简要", "简短", "大概", "概况", "概述"]
        deep_markers = ["详细", "具体", "就业", "升学", "课程", "录取", "分数", "位次", "报考建议", "怎么选", "对比", "规划"]
        if any(marker in msg for marker in brief_markers) and not any(marker in msg for marker in deep_markers):
            return 512
        return 900

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级为官方人工确认。"""
        keywords = [
            "联系招生办", "招生办公室", "官方招生网", "人工确认",
            "无法确认", "以学校公布为准", "以招生章程为准",
            "escalate", "specialist", "无法处理",
        ]
        return any(kw in content for kw in keywords)


class GeneralAdmissionsAgent(BaseAgent):
    agent_type = AgentType.GENERAL
    system_prompt = (
        "你是 河北工业大学招生咨询助手，服务对象是有意向报考该校的考生和家长。"
        "你主要负责学校概况、办学层次、校区位置、校园生活、宿舍食堂、社团活动、通用接待和问题澄清。"
        "优先依据知识库、记忆上下文和 MCP 工具结果；没有依据时必须说明信息边界，不要用常识补全成官方事实。"
        "涉及学院清单、专业清单、校区安排、收费、招生计划等事实表时，只有知识库或工具明确列出时才逐项列举；否则说明当前知识库未提供完整清单，并引导查看本科招生网。"
        "涉及具体年份、省份、科类、批次、专业、分数、位次时要明确条件。"
        "不要承诺一定录取，不要编造官方数据。信息不足时，主动追问省份、科类、分数、位次、目标专业等关键信息。"
        "涉及最终录取、收费、招生计划等官方口径时，提醒以学校招生网、当年招生章程和省级考试院公布信息为准。" \
        "最后可以加一句推荐加入学长学姐答疑QQ群: 925926323（新生入学咨询群）获取更多校园信息。"
    )


class PolicyAgent(BaseAgent):
    agent_type = AgentType.POLICY
    system_prompt = (
        "你是河北工业大学招生政策 Agent。用户没有明确学校时，默认目标学校是河北工业大学。"
        "专注回答招生章程、录取规则、专业调剂、转专业、体检限制、投档和退档风险、学费住宿费、招生计划等问题。"
        "优先依据知识库、记忆上下文和 MCP 工具结果；没有依据时必须说明信息边界，不得把往年政策或经验说成当年官方规定。"
        "必须强调政策具有年份差异，具体以当年本科招生章程、学校招生网和省级考试院文件为准。"
        "遇到用户缺少省份、科类、批次或年份时，应先提示补充条件，再给出一般性解释。"
    )


class RiskAgent(BaseAgent):
    agent_type = AgentType.RISK
    system_prompt = (
        "你是河北工业大学录取分数与风险分析 Agent。用户没有明确其他学校时，只评估河北工业大学。"
        "专注处理分数、位次、历年录取线、招生计划、冲稳保判断。"
        "优先依据知识库、记忆上下文和 MCP 工具结果；没有依据时必须说明信息边界。"
        "你只能基于用户提供的省份、科类、分数、位次、目标专业，以及知识库或 MCP 工具返回的历年数据进行解释。"
        "如果工具结果为信息不足、无数据或未给出具体年份数据，不得编造分数、位次、年份或用 XXX/YYY、假设数据占位。"
        "如果 MCP 录取风险评估状态不是 ok，只能追问缺失条件或说明当前无法自动判断，不能输出录取趋势和冲稳保结论。"
        "工具给出历年数据时，必须优先引用工具中的年份、最低分、最低位次和风险判断。"
        "MCP 上下文中的 facts 是原始 xlsx 事实，derived_claims 是工具计算出的均值、匹配年份和风险结论；"
        "回答分数位次时只能引用这两类字段，不得自行计算新的均值、范围或结论。"
        "当用户询问“搭配什么稳妥专业、备选专业、保底专业、冲稳保组合”时，只能列举 MCP 工具返回的 alternative_majors 或上下文明确给出的候选；"
        "如果 MCP 未返回候选专业，不得自行根据常识列出专业名称、年份、最低分或最低位次，只能说明当前工具数据不足并建议查询本科招生网。"
        "不要说“请稍等，我将查询”或“假设工具返回”；如果上下文没有实际工具结果，就直接说明缺少信息或无法自动判断。"
        "不得承诺一定录取，只能给出倾向性判断和依据，例如冲、稳、保、风险较高、需要结合位次进一步确认。"
        "如果用户只给分数未给省份、科类或位次，应优先追问这些信息，不要泛泛推荐其他学校。"
    )


class PlanningAgent(BaseAgent):
    agent_type = AgentType.PLANNING
    system_prompt = (
        "你是河北工业大学报考规划 Agent。用户没有明确学校时，默认目标学校是河北工业大学。"
        "专注回答专业介绍、专业对比、就业升学、专业搭配、冲稳保组合建议和长期发展路径。"
        "优先依据知识库、记忆上下文和 MCP 工具结果；没有依据时必须说明信息边界，不得把通用规划建议包装成学校官方事实。"
        "涉及学院清单、专业清单、校区安排、收费、招生计划等事实表时，只有知识库或工具明确列出时才逐项列举；否则说明当前知识库未提供完整清单，并引导查看本科招生网。"
        "必须先判断用户是在要“简单介绍”还是“深度规划”。"
        "当用户说“简单介绍、简要介绍、简单说一下、概况、了解一下”时，只输出学院或专业的核心概况，"
        "控制在 3-5 句话或 3 个短要点内；不要展开专业逐项详解、社会服务、就业升学、录取风险、志愿梯度或报考建议，除非用户明确追问。"
        "如果用户上一轮已说明对象，本轮只说“简单介绍一下”等模糊追问，应沿用上一轮对象，不要重新反问哪个专业。"
        "当用户要求专业选择、对比、志愿搭配或深度规划时，再把兴趣、能力要求、课程差异、就业方向、升学路径、录取风险和志愿梯度结合起来。"
        "专业介绍必须优先基于知识库中该专业的专门资料；如果知识库只提供通用框架或没有给出该专业事实，"
        "不得把通用课程、就业方向、培养特色、保研就业数据包装成河北工业大学官方信息。"
        "可以用“通常来说”说明学科大方向，但必须明确这是通用参考，不代表学校当年培养方案或官方统计。"
        "回答具体专业时，应区分“知识库已确认的信息”和“需以招生网、学院官网、当年培养方案确认的信息”。"
        "不要反问“哪所大学”，除非用户明确表示要比较其他学校；不要承诺薪资、就业率或一定录取。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAdmissionsAgent
    """

    # 意图值 → Agent 类型的静态映射（路由表）。
    # 使用字符串是为了兼容当前旧版 IntentCategory，以及后续改造后的招生意图。
    _INTENT_ROUTING: Dict[str, AgentType] = {
        "school_info": AgentType.GENERAL,
        "school_overview": AgentType.GENERAL,
        "major_info": AgentType.PLANNING,
        "admission_policy": AgentType.POLICY,
        "score_risk": AgentType.RISK,
        "tuition": AgentType.POLICY,
        "tuition_scholarship": AgentType.POLICY,
        "campus_life": AgentType.GENERAL,
        "career": AgentType.PLANNING,
        "career_prospect": AgentType.PLANNING,
        "comparison": AgentType.PLANNING,
        "escalation": AgentType.ESCALATION,
        # 兼容外部调用方可能传入的泛化意图，统一收敛到招生咨询语义。
        "query": AgentType.GENERAL,
        "request": AgentType.GENERAL,
        "policy": AgentType.POLICY,
        "contact": AgentType.ESCALATION,
        "complaint": AgentType.ESCALATION,
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
    ):
        client = create_llm_client(api_key=api_key, base_url=base_url, model=model)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL:   [GeneralAdmissionsAgent(client, model)],
            AgentType.POLICY:    [PolicyAgent(client, model)],
            AgentType.RISK:      [RiskAgent(client, model)],
            AgentType.PLANNING:  [PlanningAgent(client, model)],
        }

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency

        # 复杂问题自动并行协作，但默认只选择一个主 Agent，避免回答冗长重复。
        collaboration = self._collaboration_targets(req)
        # 如果是多个问题，需要多个 Agent 并行处理，就跳过 三层路由决策_route()
        if len(collaboration) > 1:
            return await self.run_parallel(req, collaboration)

        # 2. 路由：选择 Agent 类型
        agent_type = self._route(req.intent, req.urgency)

        # 3. 执行（含降级）
        response = await self._execute(req, agent_type)

        # 4. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or self._intent_value(req.intent) == "escalation":
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处可记录招生咨询请求、提示联系招生办公室或招生网官方渠道。

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂招生问题（如同时涉及分数风险、专业选择和就业前景）。
        """
        t0 = time.monotonic()
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                parts.append(f"[{r.agent_type.value}]\n{r.content}")

        combined = "\n\n".join(parts) if parts else "抱歉，所有 Agent 均处理失败。"
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        intent_value = self._intent_value(intent)
        if intent_value in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent_value]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里仅在用户明确提出多维需求时才补充协作 Agent，
        例如"河南理科 580 分想报计算机，宿舍和就业怎么样"才需要多个 Agent 协作。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        intent_target = self._INTENT_ROUTING.get(self._intent_value(req.intent))
        if intent_target and intent_target != AgentType.ESCALATION:
            targets.append(intent_target)

        dimensions = [
            (AgentType.RISK, ["稳不稳", "能不能报", "能报吗", "录取概率", "录取风险", "位次", "分数线", "录取线", "最低分", "冲稳保"]),
            (AgentType.POLICY, ["转专业", "调剂", "退档", "体检", "招生章程", "录取规则", "招生计划", "学费", "住宿费", "收费"]),
            (AgentType.PLANNING, ["怎么选", "对比", "区别", "搭配", "组合", "专业推荐", "就业", "升学", "考研", "保研", "课程", "学什么"]),
            (AgentType.GENERAL, ["宿舍", "食堂", "校区", "社团", "校园生活", "学校优势", "在哪个城市"]),
        ]
        matched = [
            agent_type
            for agent_type, keywords in dimensions
            if any(keyword in msg for keyword in keywords)
        ]

        # 至少命中两个维度，才认为用户在问复合问题；否则交给主路由即可。
        if len(set(matched)) >= 2:
            targets.extend(matched)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAdmissionsAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        scoped_req = self._with_skills(req, agent.agent_type)
        response = await agent.handle(scoped_req)

        # 专属 Agent 失败时降级到 GeneralAdmissionsAgent
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAdmissionsAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                fallback_req = self._with_skills(req, fallback.agent_type)
                response = await fallback.handle(fallback_req)

        return response

    def _with_skills(self, req: Request, agent_type: AgentType) -> Request:
        """
        按当前 Agent 类型和用户关键词筛选 Skills，并创建请求副本。

        并行协作时每个 Agent 会拿到自己的 Skill 上下文，避免共享 Request 被并发修改。
        """
        if self._skill_manager is None:
            return req
        try:
            skills_context = self._skill_manager.prompt_for(req.message, agent_type.value)
        except Exception as ex:
            logger.warning(f"SkillManager 构建上下文失败: {ex}")
            return req
        if not skills_context:
            return req
        return replace(req, skills_context=skills_context)

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 risk_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)

    @staticmethod
    def _intent_value(intent: Optional[IntentCategory]) -> str:
        if intent is None:
            return ""
        return str(getattr(intent, "value", intent))
