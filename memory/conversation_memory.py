"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 上下文构建时三级记忆融合，按重要性 + 时效性排序
  - 工作记忆超过阈值时自动压缩（LLM 摘要），防止 context 爆炸
  - 对话摘要和用户画像提取通过 LLM API 完成
  - 情景记忆和用户画像的向量化由 ChromaDB 默认 embedding 模型完成
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis
from core.llm_client import create_llm_client

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 考生画像：省份、科类、分数、位次、专业偏好
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[考生画像]\n{json.dumps(self.user_profile, ensure_ascii=False)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
    ):
        self._client = create_llm_client(api_key=api_key, base_url=base_url, model=model)
        self._model  = model

        self._redis = redis.from_url(redis_url, decode_responses=True)

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        try:
            chroma = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection("episodic")
        # 用户画像：存储提炼出的偏好和实体
        self._profile  = chroma.get_or_create_collection("user_profile")

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        # 追加到 Redis 列表（左推，最新在前）
        self._redis.lpush(key, json.dumps({
            "role":      msg.role.value,
            "content":   msg.content,
            "ts":        msg.timestamp.isoformat(),
            "metadata":  msg.metadata,
        }))
        self._redis.expire(key, 86400)  # 24h TTL

        # 超过压缩阈值时触发压缩
        if self._redis.llen(key) >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼考生报考画像，更新用户画像。
        用 LLM 提炼省份、科类、分数、位次、专业偏好等，然后存入 ChromaDB。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        prompt = f"""从以下大学招生咨询对话中提炼考生报考画像，返回 JSON。

重点提取：
- province: 考生省份，例如河北、天津、河南
- subject_type: 科类或选科，例如物理类、历史类、理科、文科、综合改革
- score: 高考分数，只保留数字字符串
- rank: 高考位次，只保留数字字符串
- target_school: 目标学校，当前项目默认是河北工业大学
- target_majors: 目标专业列表，例如计算机科学与技术、软件工程
- preferences: 报考偏好，例如就业好、城市交通方便、考研、保研、宿舍好、低学费
- risk_preference: 风险偏好，只能是 稳妥 / 冲刺 / 均衡 / 保底 / 未知
- concerns: 关注点，例如转专业、宿舍、奖学金、就业、考研率、校区、学费
- budget_sensitive: 是否明显关注费用，true/false
- accepts_adjustment: 是否接受专业调剂，true/false/null
- accepts_sino_foreign: 是否接受中外合作或高学费项目，true/false/null

没有提到的信息使用空字符串、空列表、false 或 null，不要编造。

对话:
{text}

返回格式:
{{
  "province": "",
  "subject_type": "",
  "score": "",
  "rank": "",
  "target_school": "河北工业大学",
  "target_majors": [],
  "preferences": [],
  "risk_preference": "未知",
  "concerns": [],
  "budget_sensitive": false,
  "accepts_adjustment": null,
  "accepts_sino_foreign": null
}}"""
        prompt = self._safe_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=512, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            extracted_profile = json.loads(raw[s:e])
            old_profile = await self._get_profile(user_id)
            profile_data = self._merge_profile(old_profile, extracted_profile)

            doc_id = f"{user_id}_profile"
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))

            try:
                self._profile.delete(ids=[doc_id])
            except Exception:
                pass

            # 直接传 documents，让 ChromaDB 内置模型生成 embedding（不依赖 Voyage API）
            self._profile.add(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat()}],
            )
            logger.info(f"用户画像已更新: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(user_id, query or (recent[-1].content if recent else ""))

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过）
        summary = self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 旧消息存入情景记忆（ChromaDB）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条
        keep        = messages[-5:]

        # LLM 摘要
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(resp.content[0].text).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        # 存摘要到 Redis
        skey = self._summary_key(user_id, conv_id)
        old_summary = self._redis.get(skey) or ""
        new_summary = self._safe_text(f"{old_summary}\n{summary}").strip()
        self._redis.setex(skey, 86400, new_summary)

        # 旧消息存入情景记忆
        await self._store_episodic(user_id, conv_id, text, summary)

        # 重置工作记忆为最近 5 条
        key = self._wm_key(user_id, conv_id)
        self._redis.delete(key)
        for m in reversed(keep):
            self._redis.lpush(key, json.dumps({
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            }))
        self._redis.expire(key, 86400)
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """语义检索情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            # 直接传 query_texts，ChromaDB 内置模型自动生成向量做匹配
            results = self._episodic.query(
                query_texts=[query_text],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
            )
            docs = results["documents"][0] if results["documents"] else []
            return [self._safe_text(doc) for doc in docs if isinstance(doc, str) and doc.strip()]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """将压缩后的对话片段存入情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            self._episodic.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:500])}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取考生画像（取最新一条）。"""
        try:
            results = self._profile.get(where={"user_id": user_id})
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            if not docs:
                return {}
            latest_idx = 0
            latest_ts = ""
            for idx, meta in enumerate(metas or []):
                ts = str((meta or {}).get("ts", ""))
                if ts >= latest_ts:
                    latest_ts = ts
                    latest_idx = idx
            return json.loads(docs[latest_idx])
        except Exception as ex:
            logger.warning(f"获取考生画像失败: {ex}")
        return {}

    @classmethod
    def _merge_profile(cls, old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """合并考生画像：保留旧信息，只用明确的新信息覆盖或追加。"""
        merged = cls._default_profile()
        if isinstance(old, dict):
            merged.update(cls._normalize_profile(old))
        new_profile = cls._normalize_profile(new if isinstance(new, dict) else {})

        scalar_keys = ["province", "subject_type", "score", "rank", "target_school", "risk_preference"]
        for key in scalar_keys:
            value = new_profile.get(key)
            if value not in (None, "", [], {}):
                merged[key] = value

        list_keys = ["target_majors", "preferences", "concerns"]
        for key in list_keys:
            merged[key] = cls._merge_unique_list(merged.get(key), new_profile.get(key))

        if new_profile.get("budget_sensitive") is True:
            merged["budget_sensitive"] = True
        for key in ["accepts_adjustment", "accepts_sino_foreign"]:
            if new_profile.get(key) is not None:
                merged[key] = new_profile[key]

        merged["updated_at"] = datetime.now().isoformat()
        return merged

    @staticmethod
    def _default_profile() -> Dict[str, Any]:
        return {
            "province": "",
            "subject_type": "",
            "score": "",
            "rank": "",
            "target_school": "河北工业大学",
            "target_majors": [],
            "preferences": [],
            "risk_preference": "未知",
            "concerns": [],
            "budget_sensitive": False,
            "accepts_adjustment": None,
            "accepts_sino_foreign": None,
            "updated_at": "",
        }

    @classmethod
    def _normalize_profile(cls, profile: Dict[str, Any]) -> Dict[str, Any]:
        normalized = cls._default_profile()
        for key in normalized:
            if key in profile:
                normalized[key] = profile[key]

        for key in ["province", "subject_type", "score", "rank", "target_school", "risk_preference"]:
            normalized[key] = cls._safe_text(normalized.get(key)).strip()

        if not normalized["target_school"]:
            normalized["target_school"] = "河北工业大学"
        if normalized["risk_preference"] not in {"稳妥", "冲刺", "均衡", "保底", "未知"}:
            normalized["risk_preference"] = "未知"

        for key in ["target_majors", "preferences", "concerns"]:
            normalized[key] = cls._as_clean_list(normalized.get(key))

        for key in ["budget_sensitive", "accepts_adjustment", "accepts_sino_foreign"]:
            normalized[key] = cls._normalize_bool_or_none(normalized.get(key))

        return normalized

    @classmethod
    def _as_clean_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            items = [value]
        return list(dict.fromkeys(
            cls._safe_text(item).strip()
            for item in items
            if cls._safe_text(item).strip()
        ))

    @classmethod
    def _merge_unique_list(cls, old: Any, new: Any) -> List[str]:
        return list(dict.fromkeys(cls._as_clean_list(old) + cls._as_clean_list(new)))

    @staticmethod
    def _normalize_bool_or_none(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "yes", "1", "是", "接受", "愿意"}:
                return True
            if text in {"false", "no", "0", "否", "不接受", "不愿意"}:
                return False
            if text in {"", "null", "none", "未知"}:
                return None
        return bool(value)

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
