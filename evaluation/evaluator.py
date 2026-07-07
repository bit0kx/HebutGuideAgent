"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from core.intent_recognizer import IntentCategory, IntentRecognizer
from core.llm_client import create_llm_client

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]


class ContextBuilder(Protocol):
    async def __call__(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        ...


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是一个大学招生咨询质量评估专家。请对以下招生 Agent 响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对考生或家长的问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确，是否优先依据知识库、MCP 工具结果或官方口径，是否避免编造录取结论（0=明显错误，1=完全可靠）
- completeness: 是否覆盖招生咨询所需关键条件，例如省份、科类/选科、分数、位次、专业、年份、政策边界（0=严重缺失，1=完整充分）
- helpfulness: 用户能否据此采取下一步行动，例如补充位次、查看招生章程、查询招生网、配置冲稳保方案（0=毫无帮助，1=非常有帮助）

额外要求：
- 对录取概率、分数位次、招生计划等问题，不能因为回答谨慎就扣分；只要说明依据和不确定性，应视为更可靠。
- 如果信息不足，Agent 主动追问省份、科类、分数、位次或目标专业，应视为有帮助。
- 如果 Agent 承诺“一定录取”、编造数据、混用不同省份数据，应降低 accuracy 分数。
- 如果响应出现“假设工具返回”“假设数据”“XXX/YYY”等占位或伪造检索结果，accuracy 必须不高于 0.2，completeness 必须不高于 0.4。

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    def __init__(self, client: Any, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        violation = self._hard_violation(response)
        if violation:
            return QualityScores(
                relevance=0.8,
                accuracy=0.0,
                completeness=0.2,
                helpfulness=0.2,
                error=violation,
            )

        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
            )
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=str(ex),
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @staticmethod
    def _hard_violation(response: str) -> str:
        text = response or ""
        forbidden = [
            "假设工具返回",
            "假设工具",
            "假设数据",
            "假设返回",
            "假设我们有以下数据",
            "（假设",
            "(假设",
            "XXX",
            "YYY",
        ]
        for marker in forbidden:
            if marker in text:
                return f"响应包含伪造或占位数据标记: {marker}"
        return ""


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            history = self._context_to_history(case.context)
            result = await self._recognizer.recognize(case.message, history=history)
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "history": history or [],
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }

    @staticmethod
    def _context_to_history(context: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, str]]]:
        """
        将 IntentTestCase.context 转成 IntentRecognizer 可用的多轮 history。

        支持几种常见写法：
        - {"history": [{"role": "user", "content": "..."}]}
        - {"recent_messages": [{"role": "assistant", "content": "..."}]}
        - {"turns": ["第一轮用户消息", "第二轮用户消息"]}
        """
        if not context:
            return None

        raw_history = (
            context.get("history")
            or context.get("recent_messages")
            or context.get("messages")
        )
        history: List[Dict[str, str]] = []

        if isinstance(raw_history, list):
            for item in raw_history:
                if isinstance(item, dict):
                    role = str(item.get("role") or "user").strip().lower()
                    content = str(item.get("content") or item.get("message") or "").strip()
                else:
                    role = "user"
                    content = str(item).strip()
                if role not in {"user", "assistant", "system"}:
                    role = "user"
                if content:
                    history.append({"role": role, "content": content})

        turns = context.get("turns")
        if isinstance(turns, list):
            for item in turns:
                content = str(item).strip()
                if content:
                    history.append({"role": "user", "content": content})

        return history or None


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        client = create_llm_client(api_key=api_key, base_url=base_url, model=model)

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(client, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()
        self._context_builder = context_builder

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": []
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        # 3. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 4. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 5. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
        )
        self._history.append(report)
        self._save_baseline(report)
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            history_context = self._history_context(history)
            domain_context = await self._build_domain_context(
                question,
                history=history[-6:] if history else None,
            )
            context = "\n\n".join(part for part in [history_context, domain_context] if part)
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_req)
            actual_answer = orch_result.response

            scores = await self._judge.judge(question, actual_answer, context=context or None)
            passed = scores.overall >= self.PASS_THRESHOLD

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                },
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}",
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type.value,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                },
            ))

        return results

    async def _build_domain_context(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """为评测对话补充与 /chat 主链路一致的领域上下文。"""
        if self._context_builder is None:
            return ""
        try:
            return await self._context_builder(question, history=history)
        except Exception as ex:
            logger.warning(f"构建评测领域上下文失败: {ex}")
            return ""

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：补充招生场景 Few-shot，重点覆盖分数风险、专业咨询、录取政策、校园生活等类别")
        if scores.get("relevance", 1.0) < 0.75:
            recs.append("相关性偏低：检查招生 Agent system_prompt，确保回答聚焦于考生问题和目标学校信息")
        if scores.get("accuracy", 1.0) < 0.75:
            recs.append("准确性偏低：检查 RAG 知识库、risk_assessment 工具数据和回答中的官方口径约束，避免编造录取结论")
        if scores.get("completeness", 1.0) < 0.75:
            recs.append("完整性偏低：分数位次类问题应覆盖省份、科类、分数、位次、专业、年份和不确定性说明")
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append("有用性偏低：回答应给出下一步行动，例如补充位次、查询招生网、联系招生办或配置冲稳保方案")
        if not recs:
            recs.append("招生咨询评测指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("这个学校在哪个城市？", "school_info"),
    IntentTestCase("河北工业大学是 211 吗？", "school_info"),
    IntentTestCase("计算机专业学什么？", "major_info"),
    IntentTestCase("我河南理科580分能报吗？", "score_risk"),
    IntentTestCase("我河北物理类620分，位次9000，报计算机稳吗？", "score_risk"),
    IntentTestCase("转专业政策是什么？", "admission_policy"),
    IntentTestCase("服从调剂会被退档吗？", "admission_policy"),
    IntentTestCase("宿舍条件怎么样？", "campus_life"),
    IntentTestCase("学费和住宿费是多少？", "tuition"),
    IntentTestCase("这个专业就业前景如何？", "career"),
    IntentTestCase("软件工程和人工智能怎么选？", "comparison"),
    IntentTestCase("你好，我想咨询报考", "admission_policy"),
    IntentTestCase(
        message="那这个呢？",
        expected_intent="score_risk",
        context={
            "history": [
                {"role": "user", "content": "我是河北物理类620分，位次9000，想报计算机科学与技术，稳不稳？"},
                {"role": "assistant", "content": "需要结合历年最低位次和当年计划判断。"},
                {"role": "user", "content": "如果换成软件工程"},
            ],
        },
    ),
    IntentTestCase(
        message="哪个更适合我？",
        expected_intent="comparison",
        context={
            "history": [
                {"role": "user", "content": "我比较看重就业，也想考研。"},
                {"role": "assistant", "content": "可以比较软件工程和人工智能的课程、就业和升学路径。"},
                {"role": "user", "content": "软件工程和人工智能这两个方向"},
            ],
        },
    ),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {"question": "河北工业大学的优势专业有哪些？"},
    {"question": "河北工业大学计算机科学与技术专业主要学什么？"},
    {"question": "河北工业大学转专业政策是什么？"},
    {"question": "天津考生631分、位次6375，报计算机科学与技术稳不稳？"},
    {"turns": ["我是河南理科580分", "想报计算机", "稳不稳？"]},
    {"turns": ["我是河北物理类620分，位次9000", "想报计算机科学与技术", "需要搭配什么稳妥专业吗？"]},
    {"turns": ["我比较看重就业", "软件工程和人工智能怎么选？"]},
]

