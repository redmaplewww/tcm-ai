"""pydantic 输出模型：Agent的结构化输出与推理引擎的结果类型。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 知识查询

class KBSource(BaseModel):
    """知识来源引用（文件+章节）。"""
    file: str = Field(description="知识库文件相对路径")
    section: str = Field(default="", description="章节标题")
    score: float = Field(default=0.0, description="检索相关度得分")


class QueryAnswer(BaseModel):
    """知识查询Agent的输出。"""
    question: str = Field(description="用户问题")
    answer: str = Field(description="面向用户的中文回答，必须依据知识库内容组织")
    key_points: list[str] = Field(default_factory=list, description="要点列表（3-6条）")
    sources: list[KBSource] = Field(default_factory=list, description="引用的知识库来源")
    disclaimer: str = Field(
        default="本内容为中医药知识科普参考，不构成诊疗建议；用药与治疗请咨询执业中医师。",
        description="免责声明",
    )


# ---------------------------------------------------------------- 问诊推理

class RedFlag(BaseModel):
    """急症红旗警示。"""
    id: str
    level: Literal["emergency", "urgent", "see_doctor"] = Field(
        description="emergency=立即120急诊; urgent=24小时内就医; see_doctor=先明确诊断再调理"
    )
    name: str
    matched: list[str] = Field(default_factory=list, description="命中的触发关键词")
    advice: str = ""
    note: str = ""


class SyndromeCandidate(BaseModel):
    """辨证候选证型（带打分）。"""
    name: str = Field(description="证型名称，如'风寒束表证'")
    system: str = Field(default="", description="辨证体系，如'脏腑辨证'")
    organ: str = Field(default="", description="病位脏腑，如'肝'")
    score: float = Field(description="匹配得分")
    confidence: float = Field(description="置信度0-1")
    matched_symptoms: list[str] = Field(default_factory=list, description="命中的关键症状")
    missing_key_symptoms: list[str] = Field(default_factory=list, description="未确认的关键症状")
    tongue: str = ""
    pulse: str = ""
    treatment: str = Field(default="", description="治法")
    formulas: list[str] = Field(default_factory=list, description="代表方剂")
    rationale: str = Field(default="", description="辨证依据说明")


class FollowUpQuestion(BaseModel):
    """按十问歌维度生成的追问。"""
    dimension: str = Field(description="问诊维度，如'寒热''汗''二便'")
    question: str = Field(description="具体追问内容")


class PatientCase(BaseModel):
    """患者输入信息（问诊推理的输入）。"""
    symptoms: list[str] = Field(default_factory=list, description="症状列表（可为口语表述）")
    free_text: str = Field(default="", description="病情自由描述")
    tongue: str = Field(default="", description="舌象描述，如'舌红苔黄腻'；未知为空")
    pulse: str = Field(default="", description="脉象描述，如'脉弦滑'；未知为空")
    sex: Optional[Literal["男", "女"]] = None
    age: Optional[int] = None
    pregnant: bool = Field(default=False, description="是否怀孕")


class ConsultationResult(BaseModel):
    """问诊Agent的输出：红旗检测 -> 辨证 -> 建议 -> 追问。"""
    red_flags: list[RedFlag] = Field(default_factory=list)
    is_emergency: bool = Field(default=False, description="是否命中emergency级红旗")
    emergency_advice: str = Field(default="", description="急症就医指引")
    symptoms_canonical: list[str] = Field(default_factory=list, description="规范化后的症状术语")
    syndromes: list[SyndromeCandidate] = Field(default_factory=list, description="候选证型（按得分降序，最多3个）")
    primary_syndrome: str = Field(default="", description="最可能的证型名称")
    treatment_principle: str = Field(default="", description="治法")
    formula_suggestions: list[str] = Field(default_factory=list, description="代表方剂参考（须医师辨证）")
    otc_suggestions: list[str] = Field(default_factory=list, description="常用中成药参考")
    care_advice: list[str] = Field(default_factory=list, description="调护建议")
    lifestyle_advice: list[str] = Field(default_factory=list, description="按用户体质的生活习惯建议")
    followup_questions: list[FollowUpQuestion] = Field(default_factory=list, description="按十问歌生成的追问")
    special_population_notes: list[str] = Field(default_factory=list, description="特殊人群提示（孕妇/小儿/老人）")
    disclaimer: str = Field(
        default="以上为中医知识库推理参考，不构成诊断与处方建议；请前往正规医疗机构由执业中医师四诊合参、辨证施治。",
    )


# ---------------------------------------------------------------- 多轮问诊

class EvidenceItem(BaseModel):
    """一条用户提供的证据（症状/自诊/病史/舌象/脉象），带可信度。"""
    text: str = Field(description="用户原词")
    kind: Literal["symptom", "self_diagnosis", "history", "tongue", "pulse", "temperature"] = "symptom"
    reliability: Literal["high", "medium", "low", "disputed"] = "high"
    dimension: str = Field(default="", description="所属问诊维度（汗/二便/睡眠/食欲/舌象等），用于矛盾检测")
    reason: str = Field(default="", description="降级或存疑原因")
    source: Literal["user", "doctor"] = "user"
    turn: int = 0


class EvidenceChallenge(BaseModel):
    """对用户的质询：矛盾澄清、自诊核实、数值确认等。"""
    kind: Literal["contradiction", "self_diagnosis", "pulse_unverified",
                  "numeric_conflict", "llm_doubt"] = "contradiction"
    dimension: str = ""
    question: str
    involved: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    """问诊对话中LLM每一轮的结构化输出（信息收集阶段）。"""
    reply: str = Field(description="对用户说的话：追问1-2个最关键的问题，或说明下一步")
    new_symptoms: list[str] = Field(default_factory=list, description="本轮用户直接描述的症状（保留原词，不含自诊术语）")
    self_diagnoses: list[str] = Field(default_factory=list, description="用户自称的病证名（如'肾虚''上火'），不可直接采信")
    history_claims: list[str] = Field(default_factory=list, description="既往病史/手术/在用药物等陈述（原词）")
    tongue: str = Field(default="", description="用户提供的舌象（未提供则留空，勿覆盖已有信息）")
    pulse: str = Field(default="", description="用户提供的脉象（未提供则留空）")
    pulse_from_doctor: bool = Field(default=False, description="脉象是否为医师面诊转述")
    sex: Optional[Literal["男", "女"]] = None
    age: Optional[int] = None
    pregnant: Optional[bool] = None
    doubts: list[str] = Field(default_factory=list, description="仅记录真实的前后矛盾或数值冲突（每条一句话）；普通缺失信息不要写入")
    emergency: bool = Field(default=False, description="用户描述出现急症迹象时为true，reply中须给出就医指引")
    ready_for_analysis: bool = Field(default=False, description="信息收集完成（或用户要求分析）时为true")


class ChatStepResult(BaseModel):
    """多轮问诊每一步会话状态机的输出。"""
    reply: str = ""
    is_emergency: bool = False
    emergency_advice: str = ""
    final: Optional[ConsultationResult] = None
    challenges: list[EvidenceChallenge] = Field(default_factory=list, description="本轮需要用户核实的问题")
    assessment: Optional[ConsultationResult] = Field(default=None, description="基于当前可信证据的即时辨证")
    syndrome_change: str = Field(default="", description="动态复辨的证型变化说明")
    excluded_evidence: list[str] = Field(default_factory=list, description="因存疑未参与辨证的信息及原因")
    turns: int = 0


# ---------------------------------------------------------------- 配伍安全

class HerbSafetyResult(BaseModel):
    """中药配伍安全性检查结果。"""
    herbs: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list, description="配伍禁忌（十八反/十九畏）")
    pregnancy_warnings: list[str] = Field(default_factory=list)
    toxicity_notes: list[str] = Field(default_factory=list)
    is_safe: bool = True
    advice: str = ""


# ---------------------------------------------------------------- 用户画像

class SyndromeHistoryItem(BaseModel):
    date: Optional[float] = None
    syndrome: str = ""
    treatment: str = ""


class UserProfile(BaseModel):
    """用户健康画像（从问诊历史推导）。"""
    username: str
    sessions: int = 0
    first_seen: float = 0
    last_active: float = 0
    dominant_constitution: str = Field(default="", description="主导体质倾向，如'阴虚质'")
    constitution_scores: dict[str, float] = Field(default_factory=dict, description="体质占比")
    constitution_confident: bool = Field(default=False, description="体质结论是否达到置信阈值（占比≥0.4）")
    top_symptoms: list[str] = Field(default_factory=list, description="高频症状")
    chronic_symptoms: list[str] = Field(default_factory=list, description="跨会话反复出现的症状")
    syndrome_history: list[SyndromeHistoryItem] = Field(default_factory=list, description="历次辨证结论")
    lifestyle_advice: list[str] = Field(default_factory=list, description="按体质的生活习惯建议")
