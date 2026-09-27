"""多轮问诊会话：证据取证（可信度区分）-> 红旗检测 -> 动态复辨。

ConsultSession 是不依赖具体LLM实现的会话状态机：
- 每轮先做确定性红旗检测（安全门禁，LLM之前）；
- LLM负责提取（症状原词/自诊/病史/舌脉/疑点），不默认用户输入全真；
- EvidenceEngine 做可信度分级、矛盾检测与质询生成，澄清后按最新陈述解决；
- 每轮用**当前可信证据**重新辨证（动态复辨），证型变化时明确报告；
- ready_for_analysis 或用户要求时输出最终结果（同样只基于可信证据）。
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Optional

from . import reasoning
from .evidence import EvidenceEngine
from .models import (
    ChatStepResult,
    ChatTurn,
    ConsultationResult,
    EvidenceItem,
    PatientCase,
)

#: 单次问诊最大对话轮数（防止无限追问）
MAX_TURNS = 12

#: 用户明确要求分析的表述（LLM未置ready时由引擎兜底强制出报告）
_ANALYSIS_RE = re.compile(r"(分析|辨证|辩证|什么证|给个结论|直接看看|帮我看看)")

LlmFn = Callable[[list[Any], str], Awaitable[ChatTurn]]


def _core_syndrome_name(name: str) -> str:
    """证型核心名（去括号别名）用于变化比较。"""
    name = (name or "").replace("（", "(").replace("）", ")")
    return name.split("(")[0].strip()


#: 只对冲突类信息发起质询：前后矛盾、数值与表述冲突。
#: 自诊核实/脉象来源等仍会被引擎分级与排除，但不弹"需核实"，
#: 由助手reply以常规追问方式自然覆盖。
_CONFLICT_KINDS = {"contradiction", "numeric_conflict"}


class ConsultSession:
    """多轮问诊会话状态机。"""

    def __init__(self, llm_fn: Optional[LlmFn] = None,
                 case: Optional[PatientCase] = None,
                 constitution: str = "",
                 profile_summary: str = "") -> None:
        self.case = case or PatientCase()
        self.constitution = constitution            # 用户体质倾向（画像）
        self.history: list[Any] = []
        self.evidence_texts: list[str] = list(self.case.symptoms)
        if self.case.free_text:
            self.evidence_texts.append(self.case.free_text)
        # 画像注入LLM上下文：个性化问诊与建议
        if profile_summary:
            from pydantic_ai.messages import (
                ModelRequest,
                ModelResponse,
                TextPart,
                UserPromptPart,
            )
            self.history = [
                ModelRequest(parts=[UserPromptPart(content="[用户健康画像] " + profile_summary)]),
                ModelResponse(parts=[TextPart(content="已了解该用户画像，问诊与建议将结合体质背景进行。")]),
            ]
        self.llm_fn = llm_fn
        self.turns = 0
        self.has_final = False  # 已出过辨证报告（之后转入咨询模式）
        self.engine = EvidenceEngine()
        # 初始症状视为用户自述症状（high），舌脉按可信度规则分级
        initial = [
            EvidenceItem(text=s, kind="symptom", turn=0) for s in self.case.symptoms
        ]
        if self.case.tongue:
            initial.append(EvidenceItem(text=self.case.tongue, kind="tongue", turn=0))
        if self.case.pulse:
            initial.append(
                EvidenceItem(text=self.case.pulse, kind="pulse",
                             source="doctor" if self.case.pulse.startswith("医生") else "user",
                             turn=0)
            )
        if initial:
            self.engine.add_items(initial)
        self._last_primary = ""

    # ------------------------------------------------------------ 内部工具

    def _credible_case(self) -> PatientCase:
        """由当前可信证据构建用于辨证的PatientCase。"""
        case = self.case.model_copy()
        case.symptoms = self.engine.credible_symptoms()
        case.tongue = self.engine.credible_tongue()
        case.pulse = self.engine.credible_pulse()
        return case

    def _items_from_turn(self, turn: ChatTurn) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for s in turn.new_symptoms:
            items.append(EvidenceItem(text=s, kind="symptom", turn=self.turns))
        for s in turn.self_diagnoses:
            items.append(EvidenceItem(text=s, kind="self_diagnosis", turn=self.turns))
        for s in turn.history_claims:
            items.append(EvidenceItem(text=s, kind="history", turn=self.turns))
        if turn.tongue:
            items.append(EvidenceItem(text=turn.tongue, kind="tongue", turn=self.turns))
        if turn.pulse:
            items.append(EvidenceItem(
                text=turn.pulse, kind="pulse",
                source="doctor" if turn.pulse_from_doctor else "user",
                turn=self.turns,
            ))
        return items

    def _emergency_result(self, advice: str) -> ChatStepResult:
        final = ConsultationResult(
            is_emergency=True,
            emergency_advice=advice,
            symptoms_canonical=reasoning.normalize_symptoms(self.engine.credible_symptoms()),
        )
        return ChatStepResult(
            reply=f"⚠️ {advice}",
            is_emergency=True,
            emergency_advice=advice,
            final=final,
            turns=self.turns,
        )

    def _reassess(self) -> tuple[ConsultationResult, str]:
        """动态复辨：基于可信证据重新辨证，返回（评估, 证型变化说明）。"""
        case = self._credible_case()
        assessment = reasoning.consult(case, constitution=self.constitution)
        change = ""
        current = _core_syndrome_name(assessment.primary_syndrome)
        if assessment.primary_syndrome and self._last_primary and current != self._last_primary:
            change = f"{self._last_primary} → {current}（依据信息有更新，请以最新判断为准）"
        if current:
            self._last_primary = current
        return assessment, change

    # ------------------------------------------------------------ 主流程

    async def step(self, user_text: str, force_analysis: bool = False) -> ChatStepResult:
        """处理一轮用户输入。"""
        self.turns += 1
        user_text = (user_text or "").strip()
        if user_text:
            self.evidence_texts.append(user_text)

        # 1) 安全门禁：确定性红旗检测（先于LLM）
        flags = reasoning.check_red_flags(self.evidence_texts)
        emergency = next((f for f in flags if f.level == "emergency"), None)
        if emergency is not None:
            return self._emergency_result(f"{emergency.name}：{emergency.advice}")

        # 2) 无LLM时退化为离线追问
        if self.llm_fn is None:
            followups = reasoning.missing_dimensions(
                self.evidence_texts, sex=self.case.sex, age=self.case.age
            )
            return ChatStepResult(
                reply=(followups[0].question if followups else "请补充更多症状信息。"),
                turns=self.turns,
            )

        # 轮数上限只在首次报告前强制触发；报告后转入咨询模式，不重复强制
        if self.turns > MAX_TURNS and not self.has_final:
            force_analysis = True

        turn: ChatTurn = await self.llm_fn(self.history, user_text)

        # 3) 人口学信息合并（客观信息，直接采信）
        if turn.sex and not self.case.sex:
            self.case.sex = turn.sex
        if turn.age is not None and self.case.age is None:
            self.case.age = turn.age
        if turn.pregnant is not None:
            self.case.pregnant = self.case.pregnant or turn.pregnant

        # 4) 证据取证：可信度分级、矛盾检测、质询生成（仅保留冲突类质询）
        challenges = [c for c in self.engine.add_items(self._items_from_turn(turn))
                      if c.kind in _CONFLICT_KINDS]
        challenges += self.engine.check_temperature(user_text, self.turns)

        # 5) LLM判断的急症（对确定性规则的补充兜底）
        if turn.emergency:
            advice = turn.reply or "症状可能为急症，请立即就医。"
            return self._emergency_result(advice)

        # 5.5) 用户明确要求分析且已有基本证据：即使LLM未置位也强制出报告
        if not force_analysis and not turn.ready_for_analysis and user_text:
            if _ANALYSIS_RE.search(user_text) and len(self.engine.credible_symptoms()) >= 2:
                force_analysis = True

        # 6) 动态复辨：每轮基于可信证据重新辨证
        assessment, change = self._reassess()

        excluded = self.engine.excluded_notes()
        resolutions = self.engine.resolution_notes()

        if turn.ready_for_analysis or force_analysis:
            self.has_final = True
            return ChatStepResult(
                reply=turn.reply,
                final=assessment,
                assessment=assessment,
                challenges=challenges,
                syndrome_change=change,
                excluded_evidence=excluded + resolutions,
                turns=self.turns,
            )

        return ChatStepResult(
            reply=turn.reply,
            challenges=challenges,
            assessment=assessment,
            syndrome_change=change,
            excluded_evidence=excluded + resolutions,
            turns=self.turns,
        )


# ---------------------------------------------------------------- 生产用llm_fn

def make_agent_llm_fn(model: Optional[object] = None) -> LlmFn:
    """用pydantic-ai chat agent 构造 llm_fn（自动使用真实LLM）。"""
    from .agents import build_chat_agent

    agent = build_chat_agent(model)

    async def fn(h: list[Any], user_text: str) -> ChatTurn:
        result = await agent.run(user_text, message_history=h)
        h.clear()
        h.extend(result.all_messages())
        return result.output

    return fn
