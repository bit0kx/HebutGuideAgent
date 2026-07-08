"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from core.llm_client import create_llm_client

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    SCHOOL_INFO = "school_info"              # 学校概况、校区、办学特色
    MAJOR_INFO = "major_info"                # 专业介绍、课程、适合人群
    ADMISSION_POLICY = "admission_policy"    # 招生章程、录取规则、调剂、转专业
    SCORE_RISK = "score_risk"                # 分数、位次、冲稳保风险
    TUITION = "tuition"                      # 学费、住宿费、奖助学金
    CAMPUS_LIFE = "campus_life"              # 宿舍、食堂、社团、校园生活
    CAREER = "career"                        # 就业、升学、行业前景
    COMPARISON = "comparison"                # 专业/方向/校区对比
    GREETING = "greeting"                    # 问候
    ESCALATION = "escalation"                # 联系招生办、人工确认、投诉
    OTHER = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.SCHOOL_INFO: [
        "学校在哪个城市？", "这个大学怎么样？", "学校有什么优势学科？",
    ],
    IntentCategory.MAJOR_INFO: [
        "计算机专业学什么？", "人工智能专业适合我吗？", "这个专业课程难不难？",
    ],
    IntentCategory.ADMISSION_POLICY: [
        "转专业政策是什么？", "服从调剂会被退档吗？", "学校录取规则是什么？",
    ],
    IntentCategory.SCORE_RISK: [
        "我河北省理科580分能报吗？", "位次三万报这个专业稳不稳？", "去年最低录取分是多少？",
    ],
    IntentCategory.TUITION: [
        "学费多少？", "住宿费贵吗？", "奖学金怎么评？",
    ],
    IntentCategory.CAMPUS_LIFE: [
        "宿舍条件怎么样？", "有几个校区？", "食堂和社团怎么样？",
    ],
    IntentCategory.CAREER: [
        "这个专业就业怎么样？", "毕业后能去哪些单位？", "考研和保研情况如何？",
    ],
    IntentCategory.COMPARISON: [
        "计算机和软件工程怎么选？", "两个专业有什么区别？", "哪个专业更适合就业？",
    ],
    IntentCategory.GREETING: ["你好", "老师您好", "您好，我想咨询一下"],
    IntentCategory.ESCALATION: ["招生办电话是多少？", "我要联系老师", "这个需要人工确认吗？"],
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻", "马上截止", "最后一天"],
    UrgencyLevel.HIGH: ["今天", "马上", "尽快", "hurry", "now", "志愿截止", "填报截止"],
    UrgencyLevel.MEDIUM: ["这周", "soon", "快点", "最近", "填志愿"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        self.client    = create_llm_client(api_key=api_key, base_url=base_url, model=model)
        self.model     = model
        self.threshold = confidence_threshold
        # 始终启用第二路向量相似度策略。
        # 如果当前客户端没有 embeddings.create（第三方兼容 API 常见），
        # _embed_text() 会退化为稳定的本地字符 n-gram 哈希向量。
        self._embedding_enabled = True

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """

        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence = self._vote(llm, emb, pat)
        entities = await self._extract_entities(message, history=history)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是大学招生咨询意图分析专家。根据示例判断考生或家长的咨询意图，返回 JSON。

示例:
{examples}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

意图定义:
- school_info: 学校概况、校区、地理位置、优势学科、学校特色
- major_info: 专业介绍、核心课程、培养方向、适合人群、学习难度
- admission_policy: 招生章程、录取规则、调剂、退档、转专业、体检限制
- score_risk: 分数、位次、历年录取线、招生计划、冲稳保风险
- tuition: 学费、住宿费、奖学金、助学金
- campus_life: 宿舍、食堂、社团、校园生活、校区生活条件
- career: 就业、升学、考研、保研、出国、行业前景
- comparison: 专业之间、方向之间、校区之间的对比和选择
- greeting: 问候
- escalation: 联系招生办、找老师、投诉、需要官方人工确认
- other: 以上都不符合

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        if any(kw in msg for kw in ["对比", "比较", "区别", "怎么选", "哪个好", "更适合", "还是"]):
            return {"intent": IntentCategory.COMPARISON, "confidence": 0.8}
        if any(kw in msg for kw in ["招生办", "联系电话", "人工", "官方确认", "联系老师"]):
            return {"intent": IntentCategory.ESCALATION, "confidence": 0.8}

        patterns = {
            IntentCategory.ESCALATION: ["招生办", "联系电话", "电话", "老师", "人工", "投诉", "官方确认", "联系"],
            IntentCategory.SCORE_RISK: ["分数", "位次", "排名", "录取线", "分数线", "最低分", "平均分", "能报", "稳不稳", "冲", "稳", "保"],
            IntentCategory.ADMISSION_POLICY: ["招生章程", "录取规则", "调剂", "退档", "转专业", "体检", "投档", "批次", "招生计划"],
            IntentCategory.MAJOR_INFO: ["专业", "课程", "培养", "学什么", "人工智能", "计算机", "软件工程", "法学", "会计"],
            IntentCategory.COMPARISON: ["对比", "比较", "区别", "怎么选", "哪个好", "更适合", "还是"],
            IntentCategory.TUITION: ["学费", "住宿费", "收费", "奖学金", "助学金", "中外合作", "费用"],
            IntentCategory.CAMPUS_LIFE: ["宿舍", "食堂", "校区", "社团", "校园", "生活", "环境", "交通"],
            IntentCategory.CAREER: ["就业", "升学", "考研", "保研", "出国", "薪资", "前景", "毕业", "单位"],
            IntentCategory.SCHOOL_INFO: ["学校", "大学", "在哪", "城市", "位置", "优势", "特色", "排名", "学科"],
            IntentCategory.GREETING: ["你好", "您好", "嗨", "hello", "hi", "老师好"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float]:
        """加权投票，返回最终意图和融合置信度。"""
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], float(emb.get("confidence", 0.0))
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], float(pat.get("confidence", 0.0))
            return IntentCategory.OTHER, 0.0

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        confidence = float(scores[best])
        if confidence < self.threshold:
            return IntentCategory.OTHER, confidence
        return best, confidence

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    async def _extract_entities(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, List[str]]:
        """用 LLM 从当前消息和最近历史中提取结构化实体。"""
        message = self._clean_text(message)
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )
        prompt = f"""从大学招生咨询消息中提取实体，返回 JSON（字段值为列表，没有则为空列表）:
{ctx}
消息: "{message}"
格式: {{
  "school": [],
  "major": [],
  "province": [],
  "subject_type": [],
  "score": [],
  "rank": [],
  "year": [],
  "batch": [],
  "campus": [],
  "preference": []
}}"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {
                "school": [],
                "major": [],
                "province": [],
                "subject_type": [],
                "score": [],
                "rank": [],
                "year": [],
                "batch": [],
                "campus": [],
                "preference": [],
            }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    # 不用模型，把文本拆成 n 字符片段，用 md5 hash 投影成 256 维向量。
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.SCORE_RISK and any(kw in msg for kw in ["截止", "今天", "马上", "最后"]):
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{self._clean_text(m.get('role', ''))}:{self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )
        return self._clean_text(f"{history_text}\n{message}")[:500]

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
