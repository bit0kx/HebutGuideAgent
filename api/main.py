"""
HebutGuide 招生咨询系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   HebutGuide  v2.0     ║
   ║   招生咨询 Agent    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_recognizer   = None


def _normalize_anthropic_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    }
    base_url = _normalize_anthropic_base_url(os.getenv("ANTHROPIC_BASE_URL", ""))
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def _create_tool_manager(
    cfg: Dict[str, Any],
    *,
    chroma_host_default: str = "chromadb",
    chroma_port_default: str = "8000",
    chroma_path_default: str = "/app/data/chroma",
):
    """创建并注册招生咨询 MCP 工具，供 /chat 和 CLI 复用。"""
    from mcp.admissions_tools import risk_assessment_fallback, risk_assessment_handler
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool

    tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", chroma_host_default),
        chroma_port=int(os.getenv("CHROMA_PORT", chroma_port_default)),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", chroma_path_default),
    )
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或联系招生办公室、查看本科招生网确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))

    tool_manager.register(Tool(
        name="risk_assessment",
        description="根据省份、科类、分数、位次、目标专业和历年录取数据评估报考风险",
        handler=risk_assessment_handler,
        schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "province": {"type": "string"},
                "subject_type": {"type": "string"},
                "score": {"type": "integer"},
                "rank": {"type": "integer"},
                "major": {"type": "string"},
            },
        },
        cache_ttl=3600.0,
        timeout_s=5.0,
        fallback=risk_assessment_fallback,
    ))
    return tool_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _recognizer

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator
    from core.intent_recognizer import IntentRecognizer
    from core.skill_loader import SkillManager
    from evaluation.evaluator import EndToEndEvaluator
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    _recognizer = recognizer

    # Skill 管理器：按 Agent 类型和关键词动态注入招生咨询规则
    skills_dir = os.getenv("HEBUTGUIDE_SKILLS_DIR") or str(pathlib.Path(_ROOT) / "skills")
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("HEBUTGUIDE_SKILLS_MAX_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = _create_tool_manager(cfg)

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    async def eval_context_builder(
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        intent_result = await recognizer.recognize(message, history=history)
        knowledge_text, _ = await _build_knowledge_context(message, history=history, intent=intent_result.intent)
        risk_text, _ = await _build_risk_context(message, history=history, intent=intent_result.intent)
        return "\n\n".join(part for part in [knowledge_text, risk_text] if part)

    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
        context_builder=eval_context_builder,
    )

    logger.info("HebutGuide 已就绪")
    yield

    await _monitor.stop()
    logger.info("HebutGuide 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HebutGuide 招生咨询 Agent",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {
        "status": "ok",
        "agents": _orchestrator.get_stats(),
        "skills": _skill_manager.summary() if _skill_manager else None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _orchestrator is None or _memory is None or _recognizer is None:
        raise HTTPException(503, "服务未就绪")

    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    if _is_pure_greeting(req.message):
        response = "你好！我是河北工业大学招生咨询助手。你可以问我招生政策、专业、分数位次、学费校区等问题。"
        await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
        await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, response)
        return ChatResponse(
            conv_id=conv_id,
            response=response,
            intent="greeting",
            agent_type="general",
            escalated=False,
            latency_ms=0.0,
            knowledge_used=False,
        )

    # 1-3. 读取记忆 → API 层意图识别 → 按意图构建知识库/风险工具上下文
    orch_req, knowledge_used = await _build_orchestrator_request(
        message=req.message,
        user_id=req.user_id,
        conv_id=conv_id,
        memory=_memory,
        recognizer=_recognizer,
        tool_manager=_tool_manager,
    )

    # 4. 执行
    result = await _orchestrator.run(orch_req)

    # 5. 写入记忆
    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

    # 6. 异步更新用户画像（不阻塞响应）。仅业务咨询更新，避免寒暄污染长期画像。
    if _should_update_profile(req.message):
        asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

    return ChatResponse(
        conv_id=conv_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        agent_type=result.agent_type.value,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=knowledge_used,
    )


async def _build_orchestrator_request(
    *,
    message: str,
    user_id: str,
    conv_id: str,
    memory,
    recognizer,
    tool_manager,
):
    """共享的请求上下文构建链路，供 /chat 和 CLI 复用。"""
    from agents.agent_orchestrator import Request as OrcReq

    mem_ctx = await memory.get_context(user_id, conv_id, query=message)
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    intent_result = await recognizer.recognize(message, history=history)
    knowledge_text, knowledge_used = await _build_knowledge_context(
        message,
        history=history,
        intent=intent_result.intent,
        tool_manager=tool_manager,
    )
    risk_text, risk_used = await _build_risk_context(
        message,
        history=history,
        intent=intent_result.intent,
        tool_manager=tool_manager,
    )

    context_parts = []
    if _should_use_memory_context(message):
        context_parts.append(mem_ctx.to_prompt_text())
    if knowledge_text:
        context_parts.append(knowledge_text)
    if risk_text:
        context_parts.append(risk_text)
    full_context = "\n\n".join(part for part in context_parts if part)

    return OrcReq(
        message=message,
        user_id=user_id,
        conv_id=conv_id,
        context=full_context,
        history=history,
        intent=intent_result.intent,
        urgency=intent_result.urgency,
    ), knowledge_used or risk_used


async def _build_knowledge_context(
    message: str,
    top_k: int = 3,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    intent=None,
    tool_manager=None,
) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    这里复用 MCPToolManager 的查询改写、并行召回、重排、fallback 能力。
    """
    manager = tool_manager or _tool_manager
    if manager is None:
        return "", False
    query = _build_knowledge_query(message, history)
    if not _should_use_knowledge(query, intent=intent):
        return "", False
    try:
        simple_intro = _is_simple_intro_query(message)
        effective_top_k = min(top_k, 2) if simple_intro else top_k
        snippet_chars = 360 if simple_intro else 600
        result = await manager.search_with_rewrite("knowledge_search", query, top_k=effective_top_k)
        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        used = False
        for i, item in enumerate(result.data[:effective_top_k], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            if not content:
                continue
            used = True
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:snippet_chars]}")

        if not used:
            return "", False
        if simple_intro:
            parts.append("用户要求简单介绍：请只做简短概况，控制在 3-5 句话或 3 个要点内；不要展开报考建议、就业升学、录取风险、志愿梯度，除非用户继续追问。")
        parts.append("请优先依据以上知识库内容回答；如果知识库内容不足，应说明信息边界，并引导用户查看河北工业大学本科招生网或补充关键报考条件。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


async def _build_risk_context(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    intent=None,
    tool_manager=None,
) -> tuple[str, bool]:
    """为分数/位次类问题构建 MCP 风险评估上下文。"""
    manager = tool_manager or _tool_manager
    if manager is None or not _should_use_risk_assessment(message, intent=intent):
        return "", False
    try:
        risk_query = _build_risk_query(message, history)
        result = await manager.call(
            "risk_assessment",
            {"message": risk_query},
            use_cache=True,
        )
        if not result.success or not isinstance(result.data, dict):
            return "", False

        data = result.data
        parts = ["[MCP 录取风险评估结果]"]
        status = data.get("status", "")
        parts.append(f"状态: {status}")
        parts.append(f"风险判断: {data.get('risk_level', '未知')}")

        if status == "need_more_info":
            missing = "、".join(str(x) for x in data.get("missing", []))
            parts.append(f"缺少信息: {missing}")
            parts.append(str(data.get("message", "请补充省份、科类、分数、位次和目标专业。")))
            parts.append("MCP 状态不是 ok 时，只能追问缺失信息，不得生成历年分数、位次或风险结论。")
        elif status == "ok":
            score = data.get("user_score")
            score_text = f"{score} 分，" if score not in (None, "") else ""
            parts.append(
                "考生条件: "
                f"{data.get('province')} {data.get('subject_type')} "
                f"{score_text}位次 {data.get('user_rank')}，"
                f"目标专业 {data.get('major')}"
            )
            parts.append(f"依据: {data.get('reason', '')}")
            parts.append(f"建议: {data.get('suggestion', '')}")
            history = data.get("history", [])
            if isinstance(history, list) and history:
                parts.append("历年数据:")
                for item in history:
                    parts.append(
                        f"- {item.get('year')}: 最低分 {item.get('min_score')}，"
                        f"最低位次 {item.get('min_rank')}，计划数 {item.get('plan')}"
                    )
            parts.append(str(data.get("disclaimer", "")))
        else:
            parts.append(str(data.get("message", "未获得可用风险评估结果。")))
            parts.append("MCP 状态不是 ok 时，只能说明无法自动判断，不得生成历年分数、位次或风险结论。")

        parts.append("请基于以上 MCP 结果回答；不得承诺一定录取，应说明该判断仅供参考。")
        return "\n".join(part for part in parts if part), True
    except Exception as ex:
        logger.warning(f"构建录取风险上下文失败: {ex}")
        return "", False


def _build_risk_query(message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """合并最近用户补充信息，避免多轮咨询时工具只看到半句话。"""
    user_messages: List[str] = []
    for item in history or []:
        if str(item.get("role", "")).lower() != "user":
            continue
        content = str(item.get("content", "")).strip()
        if content:
            user_messages.append(content)

    current = (message or "").strip()
    if current:
        user_messages.append(current)

    return "\n".join(user_messages[-4:])


def _build_knowledge_query(message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """为知识库检索补全多轮追问主题，例如“简单介绍一下”沿用上一轮的材料学院。"""
    current = (message or "").strip()
    if not current:
        return ""
    if not _is_vague_followup(current):
        return current

    for item in reversed(history or []):
        if str(item.get("role", "")).lower() != "user":
            continue
        previous = str(item.get("content", "")).strip()
        if previous and previous != current and not _is_vague_followup(previous):
            return f"{previous}\n{current}"
    return current


def _is_simple_intro_query(message: str) -> bool:
    """识别只需要概况的轻量问题，避免 PlanningAgent 展开成长篇规划。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    brief_markers = ["简单", "简要", "简短", "大概", "概况", "概述", "了解一下", "介绍一下"]
    deep_markers = ["详细", "具体", "就业", "升学", "课程", "录取", "分数", "位次", "报考建议", "怎么选", "对比", "规划"]
    return any(marker in msg for marker in brief_markers) and not any(marker in msg for marker in deep_markers)


def _is_vague_followup(message: str) -> bool:
    """识别缺少主题、需要依赖历史的问题。"""
    msg = (message or "").strip().lower()
    vague_set = {
        "简单介绍一下", "简要介绍一下", "简单说一下", "简要说一下",
        "介绍一下", "讲一下", "说一下", "简单介绍", "简单点",
    }
    if msg in vague_set:
        return True
    return _is_simple_intro_query(msg) and not any(
        keyword in msg
        for keyword in ["学院", "专业", "学校", "校区", "材料", "计算机", "电气", "机械", "化工", "人工智能", "软件"]
    )


def _intent_value(intent) -> str:
    if intent is None:
        return ""
    return str(getattr(intent, "value", intent))


def _should_use_knowledge(message: str, intent=None) -> bool:
    """跳过纯寒暄，业务类问题才检索知识库，避免无关 RAG 干扰回复。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    intent_value = _intent_value(intent)
    if intent_value == "greeting":
        return False
    if intent_value in {
        "school_info",
        "major_info",
        "admission_policy",
        "score_risk",
        "tuition",
        "campus_life",
        "career",
        "comparison",
        "escalation",
    }:
        return True
    admissions_keywords = [
        "河北工业大学", "河工大", "hebut", "学校", "大学", "校区", "专业", "招生", "报考",
        "志愿", "录取", "分数", "位次", "排名", "分数线", "录取线", "最低分", "最低位次",
        "招生计划", "招生章程", "录取规则", "调剂", "退档", "转专业", "体检", "批次",
        "学费", "住宿费", "奖学金", "助学金", "宿舍", "食堂", "社团", "就业", "升学",
        "考研", "保研", "中外合作", "招生办", "联系电话", "本科招生网",
    ]
    return any(kw in msg for kw in admissions_keywords) or len(msg) >= 8


def _is_pure_greeting(message: str) -> bool:
    """纯寒暄不进入 RAG/记忆/Agent，避免旧画像导致过度发挥。"""
    msg = (message or "").strip().lower()
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "老师好", "老师您好", "早上好", "晚上好"}
    return msg in greetings


def _should_use_memory_context(message: str) -> bool:
    """只有用户提出实际咨询时才注入长期记忆和用户画像。"""
    msg = (message or "").strip()
    if not msg or _is_pure_greeting(msg):
        return False
    if _should_use_knowledge(msg) or _should_use_risk_assessment(msg):
        return True
    return len(msg) >= 12


def _should_update_profile(message: str) -> bool:
    """只有包含明确报考信息的问题才提炼画像，避免把模型回答反写成用户偏好。"""
    msg = (message or "").strip()
    if not msg or _is_pure_greeting(msg):
        return False
    profile_keywords = [
        "河北", "天津", "河南", "山东", "山西", "物理", "历史", "理科", "文科",
        "分", "位次", "排名", "专业", "学费", "校区", "调剂", "中外合作",
        "计算机", "电气", "机械", "材料", "电子信息", "土木", "化工", "环境",
    ]
    return any(kw in msg for kw in profile_keywords)


def _should_use_risk_assessment(message: str, intent=None) -> bool:
    """只有明显涉及分数、位次、报考风险的问题才调用结构化风险评估工具。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    if _intent_value(intent) == "score_risk":
        return True
    risk_keywords = [
        "分", "位次", "排名", "能报", "稳吗", "稳不稳", "冲", "稳", "保",
        "录取线", "分数线", "最低分", "最低位次", "报考风险",
    ]
    return any(kw in msg for kw in risk_keywords)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看 SkillManager 当前加载的招生咨询 Skills。"""
    if _skill_manager is None:
        raise HTTPException(503, "SkillManager 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """热加载 skills 目录中的招生咨询规则。"""
    if _skill_manager is None:
        raise HTTPException(503, "SkillManager 未初始化")
    _skill_manager.reload()
    return _skill_manager.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    知识库检索调试接口：直接按用户输入查询知识库，便于验证导入文档是否命中。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.call(
        "knowledge_search",
        {"query": query, "top_k": top_k},
        use_cache=False,
    )
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


class RiskAssessmentInput(BaseModel):
    """录取风险评估请求。可只传 message，也可传结构化字段。"""
    message: Optional[str] = None
    province: Optional[str] = None
    subject_type: Optional[str] = None
    score: Optional[int] = None
    rank: Optional[int] = None
    major: Optional[str] = None


@app.post("/admissions/risk", tags=["招生工具"])
async def admissions_risk(body: RiskAssessmentInput):
    """调用 MCP risk_assessment 工具，返回结构化录取风险判断。"""
    if _tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    params = body.model_dump(exclude_none=True)
    result = await _tool_manager.call("risk_assessment", params, use_cache=True)
    if not result.success:
        raise HTTPException(500, result.error or "录取风险评估失败")
    return result.data


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "2026 年本科招生章程补充说明", "content": "录取规则、调剂政策、体检要求等以当年本科招生章程为准..."},
        {"title": "计算机科学与技术专业介绍", "content": "培养目标、核心课程、就业方向和报考建议..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([{"title": d.title, "content": d.content} for d in body.documents])
    cleared = _tool_manager.clear_cache("knowledge_search") if _tool_manager else 0
    return {
        "message": f"成功导入 {count} 个文档片段",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
        "cache_cleared": cleared,
    }


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    cleared = _tool_manager.clear_cache("knowledge_search") if _tool_manager else 0
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
        "cache_cleared": cleared,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("HebutGuide CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator
    from core.intent_recognizer import IntentRecognizer
    from core.skill_loader import SkillManager
    from memory.conversation_memory import MemoryManager, MsgRole

    cfg = _anthropic_cfg()
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    skills_dir = os.getenv("HEBUTGUIDE_SKILLS_DIR") or str(pathlib.Path(_ROOT) / "skills")
    skills = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("HEBUTGUIDE_SKILLS_MAX_CHARS", "5000")),
    )
    skills.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skills,
    )
    tool_manager = _create_tool_manager(
        cfg,
        chroma_host_default="localhost",
        chroma_port_default="8000",
        chroma_path_default="/tmp/chroma",
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        if _is_pure_greeting(msg):
            response = "你好！我是河北工业大学招生咨询助手。你可以问我招生政策、专业、分数位次、学费校区等问题。"
            await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
            await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, response)
            print(f"\nHebutGuide [general]: {response}\n")
            continue

        req, _ = await _build_orchestrator_request(
            message=msg,
            user_id=user_id,
            conv_id=conv_id,
            memory=mem,
            recognizer=recognizer,
            tool_manager=tool_manager,
        )
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)
        if _should_update_profile(msg):
            asyncio.create_task(mem.update_profile(user_id, conv_id))

        print(f"\nHebutGuide [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
