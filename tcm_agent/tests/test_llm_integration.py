"""真实LLM集成测试：GLM（Anthropic兼容端点）驱动两个pydantic-ai Agent。

- 凭据不可用时自动跳过（环境变量TCM_LLM_KEY或本机ZCode CLI登录凭据）。
- 运行：python -m pytest -m llm -v   （默认pytest运行会排除本组测试）
- 验证目标：LLM能否在工具接地（知识库/推理引擎）下正确完成问答与辨证。
"""

import asyncio

import pytest

from tcm_agent import llm
from tcm_agent.agents import build_consult_agent, build_query_agent
from tcm_agent.cli import _case_prompt
from tcm_agent.models import ConsultationResult, PatientCase

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not llm.llm_available(), reason="无LLM凭据（TCM_LLM_KEY或ZCode CLI登录）"),
]


def _run(agent, prompt):
    return asyncio.run(agent.run(prompt)).output


# ---------------------------------------------------------------- 知识问答

def test_llm_query_yin_xu():
    out = _run(build_query_agent(), "什么是阴虚？有哪些典型表现？")
    assert "阴虚" in out.answer and len(out.answer) > 50, f"回答异常: {out.answer[:100]}"
    assert out.disclaimer


def test_llm_query_herb():
    out = _run(build_query_agent(), "麻黄的功效和使用禁忌是什么？")
    blob = out.answer + "；".join(out.key_points)
    assert "发汗" in blob, f"未提及发汗功效: {blob[:200]}"


def test_llm_query_wind_cold_vs_heat():
    out = _run(build_query_agent(), "风寒感冒和风热感冒怎么区别？各用什么方向的中成药？")
    blob = out.answer + "；".join(out.key_points)
    assert "风寒" in blob and "风热" in blob, f"未覆盖两种证型: {blob[:200]}"


# ---------------------------------------------------------------- 病情推理

def test_llm_consult_wind_cold():
    case = PatientCase(
        symptoms=["怕冷", "发烧", "无汗", "头痛", "身痛", "清鼻涕"],
        tongue="舌淡红苔薄白", pulse="脉浮紧", sex="男", age=30,
    )
    out = _run(build_consult_agent(), _case_prompt(case))
    assert isinstance(out, ConsultationResult)
    assert not out.is_emergency
    family = ("风寒", "表寒", "太阳伤寒")
    assert out.primary_syndrome and any(k in out.primary_syndrome for k in family), \
        f"证型异常: {out.primary_syndrome}; 症状规范化: {out.symptoms_canonical}"
    assert out.disclaimer and "执业" in out.disclaimer


def test_llm_consult_emergency():
    case = PatientCase(
        symptoms=["胸口像压了石头", "胸痛放射到左臂", "出冷汗"], sex="男", age=58,
    )
    out = _run(build_consult_agent(), _case_prompt(case))
    assert out.is_emergency, f"急症红旗未触发: red_flags={out.red_flags}"
    assert out.emergency_advice
    assert not out.formula_suggestions, "急症不应给方剂"


def test_llm_consult_heart_spleen():
    case = PatientCase(
        symptoms=["心慌", "睡不着", "多梦", "健忘", "疲乏", "吃不下饭"],
        tongue="舌淡", sex="女", age=35,
    )
    out = _run(build_consult_agent(), _case_prompt(case))
    assert not out.is_emergency
    assert "心脾两虚" in out.primary_syndrome, \
        f"应为心脾两虚，实为: {out.primary_syndrome}; 规范化: {out.symptoms_canonical}"


# ---------------------------------------------------------------- 多轮问诊

def test_llm_chat_multiturn_wind_cold():
    """真实LLM多轮问诊：两轮收集后要求分析，应得出风寒类证型。"""
    from tcm_agent.chat_session import ConsultSession, make_agent_llm_fn

    session = ConsultSession(make_agent_llm_fn())
    r1 = asyncio.run(session.step("我这两天怕冷、有点发烧、身上没有汗"))
    assert r1.reply and not r1.is_emergency and r1.final is None
    assert session.engine.credible_symptoms(), "LLM未提取症状"

    r2 = asyncio.run(session.step(
        "还鼻塞流清鼻涕，舌头是淡红的苔薄白。可以帮我分析一下了"
    ))
    assert r2.final is not None, f"第二轮未给出结果: {r2.reply}"
    assert any(k in r2.final.primary_syndrome for k in ("风寒", "表寒", "太阳伤寒")), \
        f"证型异常: {r2.final.primary_syndrome}; 症状: {session.case.symptoms}; 舌: {session.case.tongue}"


def test_llm_chat_emergency():
    """多轮问诊中用户描述急症：应立即给出就医指引。"""
    from tcm_agent.chat_session import ConsultSession, make_agent_llm_fn

    session = ConsultSession(make_agent_llm_fn())
    r = asyncio.run(session.step("我爸突然胸口剧痛，疼到左胳膊，满头冷汗"))
    assert r.is_emergency and r.emergency_advice
    assert r.final is not None and r.final.is_emergency
    assert not r.final.formula_suggestions


# ---------------------------------------------------------------- 配伍安全

def test_llm_query_herb_combination():
    """LLM问答应调用配伍安全工具并给出正确结论。"""
    out = _run(build_query_agent(), "甘草和甘遂可以一起用吗？")
    blob = out.answer + "；".join(out.key_points)
    assert ("十八反" in blob) or ("不能" in blob) or ("不宜" in blob) or ("禁止" in blob), \
        f"未给出禁忌结论: {blob[:300]}"


# ---------------------------------------------------------------- 动态复辨与取证

def test_llm_chat_dynamic_reshabing():
    """真实LLM动态复辨：补充脾虚症状后证型从心血虚更新为心脾两虚。"""
    from tcm_agent.chat_session import ConsultSession, make_agent_llm_fn

    session = ConsultSession(make_agent_llm_fn())
    r1 = asyncio.run(session.step("我最近心慌、睡不着觉、多梦、还老忘事"))
    assert r1.assessment is not None and r1.final is None, f"首轮即出终稿: {r1.reply[:100]}"
    first = r1.assessment.primary_syndrome
    assert first, f"首轮未辨证: 症状={session.engine.credible_symptoms()}"

    r2 = asyncio.run(session.step(
        "而且最近吃不下饭，浑身疲乏没力气。请帮我分析一下"
    ))
    assert r2.final is not None, f"次轮未出结果: {r2.reply[:200]}"
    assert "心脾两虚" in r2.final.primary_syndrome, \
        f"应为心脾两虚: {r2.final.primary_syndrome}; 症状: {session.engine.credible_symptoms()}"
    # 证型发生动态变化或首轮本就是心脾两虚
    assert ("心脾两虚" in first) or (r2.syndrome_change != ""), \
        f"未体现动态变化: first={first}, change={r2.syndrome_change}"


def test_llm_chat_self_diagnosis_quiet():
    """自诊术语被静默排除（不弹质询，仅冲突才质询）。"""
    from tcm_agent.chat_session import ConsultSession, make_agent_llm_fn

    session = ConsultSession(make_agent_llm_fn())
    r = asyncio.run(session.step("我觉得我就是肾虚，网上对照的，最近腰有点酸"))
    # 仅冲突类质询；自诊不产生质询噪音
    assert all(c.kind in ("contradiction", "numeric_conflict") for c in r.challenges), \
        f"存在非冲突质询: {[c.kind for c in r.challenges]}"
    # "肾虚"这个词本身不应出现在规范化症状中
    assert "肾虚" not in r.assessment.symptoms_canonical, \
        f"自诊术语泄漏进辨证: {r.assessment.symptoms_canonical}"
