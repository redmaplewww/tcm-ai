"""pydantic-ai Agent 定义：知识查询Agent与问诊推理Agent。

- build_query_agent : 查询类问题（理论/中药/方剂/穴位/术语），输出QueryAnswer。
- build_consult_agent: 病情咨询，先红旗检测后辨证，输出ConsultationResult。

两个Agent共享确定性推理引擎与知识库工具；模型可通过环境变量TCM_MODEL
配置（如 openai:gpt-4o-mini）。测试时可传入TestModel离线运行。
"""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Agent, RunContext

from . import llm, tools
from .config import default_model
from .models import ChatTurn, ConsultationResult, QueryAnswer

QUERY_SYSTEM_PROMPT = """\
你是"中医知识查询助手"，基于一套分级中医知识库回答问题。严格遵守：

1. 回答必须依据检索到的知识库内容（search_knowledge_base）与数据查询工具的结果组织；
   工具明确说"未收录/未找到"时，如实告知用户知识库未收录，禁止编造。
2. 涉及具体中药药性、方剂组成、穴位定位、证型标准时，必须调用对应查询工具核实。
3. 区分两类表述：中医传统理论解释（"中医认为…"）与现代研究证据（"现代研究显示…"），
   不将理论表述当作科学事实陈述。
4. 用户描述出现急症迹象（剧烈胸痛、偏瘫、大出血、高热神昏等）时，优先建议立即就医。
5. 回答用简体中文，条理清晰；结尾附一句：内容供参考，诊疗请咨询执业中医师。
"""

CONSULT_SYSTEM_PROMPT = """\
你是"中医辨证参考助手"，辅助用户理解自身症状的中医辨证思路。严格遵守流程：

1. 第一步必须调用 check_red_flag_risk 检查用户描述。若有 emergency 级红旗，
   直接在结果中给出就医指引（is_emergency=true），不再输出方药建议。
2. 用 run_syndrome_reasoning 做确定性辨证（将用户症状**原词**列成列表传入，
   不要自行合并或改写术语，引擎会自动做同义词/口语规范化；可附tongue/pulse）。
   辨证结论以该工具返回为准，不得自行编造证型或方剂。
3. 综合工具结果输出 ConsultationResult：
   - 候选证型按置信度排序，说明匹配与缺失的关键症状；
   - 方剂/中成药仅作为"知识参考"呈现，必须注明须经执业中医师辨证后使用；
   - followup_questions 填入十问歌中尚未覆盖维度的追问（可用ask_followup_questions）。
4. 永远不给出最终处方、不给出确切剂量疗程；孕妇、小儿(≤6岁)、高龄(≥70岁)
   必须在 special_population_notes 中提醒就医。
5. 语气专业平和，避免绝对化与恐吓式表述。
"""

CHAT_SYSTEM_PROMPT = """\
你是"中医问诊收集助手"，任务是通过**多轮文字对话**收集辨证所需信息，由确定性
引擎给出最终辨证。核心原则：**用户的陈述不可默认全部正确**，你要做的是取证而非
采信。严格遵守：

1. 分类提取（每轮从用户话中区分）：
   - new_symptoms：用户**直接描述的身体症状原词**（"吃不下饭"就写"吃不下饭"）；
   - self_diagnoses：用户自称的病证名（"我肾虚""上火了""湿气重""气血不足"）——
     这些是判断不是症状，绝不能放进new_symptoms；
   - history_claims：既往病史、手术、在用药物，以及**长期存在的老毛病/体质
     问题**（如"平时一动就出汗好多年"）——要与本次新发症状分开记录，不能混入
     new_symptoms。
2. 只对**冲突信息**当场质询（一次问清）：
   - 前后矛盾（先说无汗后说出汗、先便秘后腹泻）：指出不一致，请用户以最新情况澄清；
   - 数值与表述冲突（"低烧"却说39℃）：请用户确认具体数值。
   **用户说到的症状必须如实记入new_symptoms**（即使可疑、矛盾或重复——可信度
   与矛盾由系统处理），不要因为信息存疑就略过不记，否则系统无法检测冲突。
   其他情况（自诊术语、未量体温、维度缺失等）**不要标注核实/质询**，按常规追问
   自然进行即可（如"具体哪里不舒服？""方便量个体温吗？"）。
3. 脉象：只有医师当面诊脉转述的才可信（pulse_from_doctor=true）；用户自查或
   模糊感觉到的脉象不要采信（系统会自动排除，无需专门追问）。
4. 每轮reply只问1-2个当前最重要的问题，优先级：主诉细节→寒热→汗→头身→二便→
   饮食口味→睡眠→情志→舌象→（女性）月经带下；已问过的不重复问。
5. 用户描述出现急症迹象（剧烈胸痛、口角歪斜偏瘫、大出血、高热神昏、呼吸困难、
   剧烈腹痛等）时：emergency=true，reply立即给出就医指引，不再常规问诊。
6. 主要症状明确且维度基本覆盖，或用户明确要求"直接分析/辨证"时：
   ready_for_analysis=true。
7. 可用工具：check_red_flag_risk（每轮必查）、ask_followup_questions（查看还缺
   哪些维度）、check_herb_safety（药物同用咨询）。
8. 你不诊断、不开方、不给剂量；语气亲切专业，避免术语轰炸，不要每轮都强调
   "核实/存疑"。
"""


def _default_model_arg():
    """默认模型参数：优先真实LLM（本机GLM凭据），否则退回TCM_MODEL配置。"""
    model = llm.build_model()
    return model if model is not None else default_model()


def _make_agent(model, output_type, system_prompt: str) -> Agent:
    """兼容不同pydantic-ai版本的Agent构建（output_type/result_type）。"""
    if model is None:
        model = _default_model_arg()
    try:
        return Agent(model, output_type=output_type, system_prompt=system_prompt)
    except TypeError:  # pragma: no cover - 旧版本兼容
        return Agent(model, result_type=output_type, system_prompt=system_prompt)


def build_query_agent(model: Optional[object] = None) -> Agent[None, QueryAnswer]:
    """构建知识查询Agent。model为None时自动选择（真实LLM优先）。"""
    agent: Agent[None, QueryAnswer] = _make_agent(model, QueryAnswer, QUERY_SYSTEM_PROMPT)

    @agent.tool
    def search_knowledge(ctx: RunContext[None], query: str, k: int = 5) -> str:
        """在中医分级知识库中检索最相关的章节内容（理论/诊断/中药/方剂/针灸/临床/科学证据）。

        Args:
            query: 检索问题或关键词，如"阴虚的症状表现"。
            k: 返回条数，默认5。
        """
        return tools.search_knowledge_base(query, k)

    @agent.tool
    def lookup_term(ctx: RunContext[None], term: str) -> str:
        """查询中医术语的标准定义（如：阴虚、辨证论治、藏象）。"""
        return tools.get_glossary_term(term)

    @agent.tool
    def lookup_herb(ctx: RunContext[None], name: str) -> str:
        """查询中药的性味归经、功效主治、用量与禁忌（如：黄芪、麻黄）。"""
        return tools.get_herb_info(name)

    @agent.tool
    def lookup_formula(ctx: RunContext[None], name: str) -> str:
        """查询方剂或中成药的组成、功用、主治证型与禁忌（如：六味地黄丸）。"""
        return tools.get_formula_info(name)

    @agent.tool
    def lookup_acupoint(ctx: RunContext[None], name: str) -> str:
        """查询穴位的定位、归经、主治与孕妇禁忌（如：足三里、合谷）。"""
        return tools.get_acupoint_info(name)

    @agent.tool
    def lookup_syndrome(ctx: RunContext[None], name: str) -> str:
        """查询证型的关键症状、舌脉表现、治法与代表方（如：心脾两虚证）。"""
        return tools.get_syndrome_info(name)

    @agent.tool
    def herb_safety(ctx: RunContext[None], herbs: list[str], pregnant: bool = False) -> str:
        """检查中药配伍安全性：十八反/十九畏禁忌、妊娠禁忌、毒性标注。

        Args:
            herbs: 药名列表，如["甘草","甘遂"]。
            pregnant: 用户是否怀孕。
        """
        return tools.check_herb_safety(herbs, pregnant)

    return agent


def build_consult_agent(model: Optional[object] = None) -> Agent[None, ConsultationResult]:
    """构建问诊推理Agent。model为None时自动选择（真实LLM优先）。"""
    agent: Agent[None, ConsultationResult] = _make_agent(model, ConsultationResult, CONSULT_SYSTEM_PROMPT)

    @agent.tool
    def detect_red_flags(ctx: RunContext[None], text: str) -> str:
        """红旗急症检测：判断病情描述是否命中需立即/尽快就医的规则。

        Args:
            text: 用户的全部病情描述（症状串联文本）。
        """
        return tools.check_red_flag_risk(text)

    @agent.tool
    def reason_syndromes(
        ctx: RunContext[None],
        symptoms: list[str],
        tongue: str = "",
        pulse: str = "",
    ) -> str:
        """确定性中医辨证：输入症状列表（口语亦可），返回候选证型打分、治法与代表方。

        Args:
            symptoms: 症状列表，如["恶寒","发热","无汗"]。
            tongue: 舌象描述（可选），如"舌淡红苔薄白"。
            pulse: 脉象描述（可选），如"脉浮紧"。
        """
        return tools.run_syndrome_reasoning(symptoms, tongue, pulse)

    @agent.tool
    def followups(ctx: RunContext[None], symptoms: list[str], sex: str = "", age: int = 0) -> str:
        """按十问歌生成尚未覆盖问诊维度的追问问题。"""
        return tools.ask_followup_questions(symptoms, sex, age)

    @agent.tool
    def search_knowledge(ctx: RunContext[None], query: str, k: int = 5) -> str:
        """在中医分级知识库中检索相关章节（调护建议、病症知识等）。"""
        return tools.search_knowledge_base(query, k)

    return agent


def build_chat_agent(model: Optional[object] = None) -> Agent[None, ChatTurn]:
    """构建多轮问诊对话Agent（信息收集阶段，每轮输出一个ChatTurn）。"""
    agent: Agent[None, ChatTurn] = _make_agent(model, ChatTurn, CHAT_SYSTEM_PROMPT)

    @agent.tool
    def detect_red_flags_chat(ctx: RunContext[None], text: str) -> str:
        """红旗急症检测：判断用户本轮描述是否命中需立即就医的急症规则。"""
        return tools.check_red_flag_risk(text)

    @agent.tool
    def remaining_followups(ctx: RunContext[None], symptoms: list[str],
                            sex: str = "", age: int = 0) -> str:
        """查看十问歌中还缺哪些问诊维度（避免重复提问）。"""
        return tools.ask_followup_questions(symptoms, sex, age)

    @agent.tool
    def herb_safety_chat(ctx: RunContext[None], herbs: list[str],
                         pregnant: bool = False) -> str:
        """检查中药同用安全性（十八反/十九畏/妊娠禁忌）。"""
        return tools.check_herb_safety(herbs, pregnant)

    return agent
