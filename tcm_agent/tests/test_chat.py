"""多轮问诊会话状态机测试（离线，llm_fn用脚本注入）。"""

import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from tcm_agent.agents import build_chat_agent
from tcm_agent.chat_session import ConsultSession
from tcm_agent.models import ChatTurn, PatientCase


def _scripted_llm(turns: list[ChatTurn], fail_on_extra: bool = True):
    """构造脚本化llm_fn：按序弹出预设ChatTurn。"""
    queue = list(turns)

    async def fn(history, user_text):
        if not queue:
            if fail_on_extra:
                raise AssertionError(f"LLM被意外多调用一次，输入: {user_text}")
            return ChatTurn(reply="请继续描述", ready_for_analysis=True)
        return queue.pop(0)

    return fn


def test_session_accumulates_and_finalizes():
    """三轮对话收集风寒表证信息后辨证。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="发烧吗？怕冷还是怕热？", new_symptoms=["怕冷", "无汗"]),
        ChatTurn(reply="鼻涕什么颜色？", new_symptoms=["发烧", "清鼻涕"]),
        ChatTurn(reply="好的，下面给出辨证参考。", new_symptoms=["头身疼痛"],
                 tongue="舌淡红苔薄白", ready_for_analysis=True),
    ]))
    r1 = asyncio.run(session.step("我最近怕冷，没有汗"))
    assert "发烧" in r1.reply and r1.final is None
    assert session.engine.credible_symptoms() == ["怕冷", "无汗"]

    r2 = asyncio.run(session.step("有点发烧，流清鼻涕"))
    assert r2.final is None
    assert "清鼻涕" in session.engine.credible_symptoms()

    r3 = asyncio.run(session.step("还浑身酸痛，舌头淡红苔薄白，帮我分析下"))
    assert r3.final is not None
    assert any(k in r3.final.primary_syndrome for k in ("风寒", "表寒", "太阳伤寒"))
    assert session.engine.credible_tongue() == "舌淡红苔薄白"
    assert session.engine.credible_symptoms() == ["怕冷", "无汗", "发烧", "清鼻涕", "头身疼痛"]


def test_session_emergency_short_circuits_before_llm():
    """急症红旗在LLM之前拦截（llm_fn被调用即失败）。"""
    session = ConsultSession(_scripted_llm([]))
    r = asyncio.run(session.step("胸口像压了石头，胸痛放射到左臂，出冷汗"))
    assert r.is_emergency and r.final is not None and r.final.is_emergency
    assert r.emergency_advice
    assert not r.final.formula_suggestions


def test_session_llm_reported_emergency():
    """LLM判断的急症同样短路（确定性规则未覆盖的表述）。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="您的情况需要立即就医，请拨打120！", emergency=True,
                 new_symptoms=["突发剧烈头痛伴喷射状呕吐"]),
    ]))
    r = asyncio.run(session.step("突然剧烈头痛还喷射状呕吐，脖子僵硬"))
    assert r.is_emergency
    assert "就医" in r.reply or "120" in r.reply


def test_session_force_analysis():
    """/诊断 强制出结果。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="好的，马上分析。", new_symptoms=["心慌", "睡不着", "疲乏", "吃不下饭"],
                 ready_for_analysis=False),
    ]))
    r = asyncio.run(session.step("", force_analysis=True))
    assert r.final is not None
    assert "心脾两虚" in r.final.primary_syndrome


def test_session_patient_info_merge():
    """性别/年龄/怀孕信息合并。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="了解了。", new_symptoms=["疲乏"], sex="女", age=28, pregnant=True),
        ChatTurn(reply="分析如下。", ready_for_analysis=True),
    ]))
    asyncio.run(session.step("女，28岁，怀孕两个月，最近很累"))
    r = asyncio.run(session.step("请分析"))
    assert session.case.sex == "女" and session.case.age == 28 and session.case.pregnant
    assert r.final is not None
    assert any("妊娠" in n or "孕妇" in n or "孕期" in n
               for n in r.final.special_population_notes)


def test_session_max_turns_force():
    """超过最大轮数自动强制辨证。"""
    turns = [ChatTurn(reply=f"再问第{i}个问题") for i in range(15)]
    turns[-1] = ChatTurn(reply="信息足够，给出参考。", ready_for_analysis=True)
    session = ConsultSession(_scripted_llm(turns))
    final = None
    for i in range(15):
        r = asyncio.run(session.step("补充一点信息"))
        if r.final is not None:
            final = r.final
            break
    assert final is not None, "超过轮数上限仍未给出结果"


def test_session_offline_without_llm():
    """无llm_fn时退化为追问提示（不出最终结果）。"""
    session = ConsultSession(case=PatientCase(symptoms=["头痛"]))
    r = asyncio.run(session.step(""))
    assert r.reply and r.final is None and not r.is_emergency


def test_session_continues_after_final():
    """报告后不结束会话：可继续咨询，补充新信息可更新报告。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="分析如下。", new_symptoms=["心悸", "失眠", "食少", "疲乏"],
                 ready_for_analysis=True),
        ChatTurn(reply="归脾汤相关问题解答……"),
        ChatTurn(reply="结合新补充的信息更新分析。", new_symptoms=["健忘", "多梦"],
                 ready_for_analysis=True),
    ]))
    r1 = asyncio.run(session.step("心慌睡不着、吃不下饭"))
    assert r1.final is not None and session.has_final is True
    assert "心脾两虚" in r1.final.primary_syndrome

    # 报告后继续咨询：不出新报告，会话保持
    r2 = asyncio.run(session.step("归脾汤能长期吃吗"))
    assert r2.final is None and r2.reply and not r2.is_emergency

    # 补充重要症状并要求重新分析：报告更新
    r3 = asyncio.run(session.step("对了，我还健忘多梦，重新分析一下"))
    assert r3.final is not None
    assert "心脾两虚" in r3.final.primary_syndrome
    assert set(["健忘", "多梦"]) <= set(session.engine.credible_symptoms())


def test_explicit_analysis_request_forces_final():
    """用户明确要求分析但LLM没置ready：引擎兜底强制出报告。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="还想再问几个问题……",  # LLM没置ready
                 new_symptoms=["心悸", "失眠", "食少", "疲乏"], ready_for_analysis=False),
    ]))
    r = asyncio.run(session.step("心慌睡不着吃不下饭，请帮我分析一下"))
    assert r.final is not None, "明确分析请求未兜底出报告"
    assert "心脾两虚" in r.final.primary_syndrome

    # 证据不足时（仅1条症状）不强制，继续收集
    session2 = ConsultSession(_scripted_llm([
        ChatTurn(reply="先补充些信息", new_symptoms=["头痛"], ready_for_analysis=False),
    ]))
    r2 = asyncio.run(session2.step("帮我分析下"))
    assert r2.final is None


def test_chat_agent_wiring():
    """chat agent构建与工具注册（TestModel）。"""
    agent = build_chat_agent(TestModel())
    ts = getattr(agent, "_function_toolset", None)
    tools = getattr(ts, "tools", {})
    assert isinstance(tools, dict) and len(tools) >= 3, f"工具不足: {list(tools)}"
    result = asyncio.run(agent.run("我最近怕冷"))
    assert hasattr(result.output, "reply")


# ---------------------------------------------------------------- 动态复辨与取证

def test_dynamic_reshabing_syndrome_change():
    """补充脾虚症状后，证型从心血虚动态更新为心脾两虚。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="胃口怎么样？", new_symptoms=["心悸", "失眠", "多梦", "健忘"]),
        ChatTurn(reply="明白了，正在分析。", new_symptoms=["食少", "神疲乏力"],
                 ready_for_analysis=True),
    ]))
    r1 = asyncio.run(session.step("最近心慌、睡不着、多梦、健忘"))
    assert r1.assessment is not None and r1.final is None
    first = r1.assessment.primary_syndrome
    assert "心血虚" in first, f"首轮应为心血虚: {first}"

    r2 = asyncio.run(session.step("最近吃不下饭，浑身没劲"))
    assert r2.final is not None
    assert "心脾两虚" in r2.final.primary_syndrome
    assert r2.syndrome_change and "心血虚" in r2.syndrome_change and "心脾两虚" in r2.syndrome_change


def test_session_contradiction_challenge_flow():
    """矛盾→质询→排除；澄清→替换并解决。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="出汗情况？", new_symptoms=["恶寒", "发热", "无汗"]),
        ChatTurn(reply="（质询被引擎生成）", new_symptoms=["自汗"]),
        ChatTurn(reply="好的。", new_symptoms=["盗汗"], ready_for_analysis=True),
    ]))
    r1 = asyncio.run(session.step("怕冷发烧，没有汗"))
    assert not r1.challenges
    r2 = asyncio.run(session.step("对了其实我出汗挺多"))
    assert any(c.kind == "contradiction" for c in r2.challenges), r2.challenges
    # 矛盾双方都不参与即时辨证
    assert "无汗" not in r2.assessment.symptoms_canonical
    assert "自汗" not in r2.assessment.symptoms_canonical
    assert any("无汗" in n for n in r2.excluded_evidence)

    r3 = asyncio.run(session.step("准确说是晚上睡着出汗，白天不出汗"))
    assert not any(c.kind == "contradiction" for c in r3.challenges)
    assert "盗汗" in r3.final.symptoms_canonical
    assert "无汗" not in r3.final.symptoms_canonical


def test_session_self_diagnosis_excluded_no_noise():
    """自诊术语不参与辨证、静默排除（不弹质询，由reply常规追问）。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="具体哪里不舒服？", new_symptoms=["腰膝酸软"],
                 self_diagnoses=["肾虚"]),
    ]))
    r = asyncio.run(session.step("我觉得我肾虚，腰有点酸"))
    # 仅冲突类才质询；自诊不产生质询
    assert all(c.kind in ("contradiction", "numeric_conflict") for c in r.challenges), \
        f"自诊不应触发质询: {[c.kind for c in r.challenges]}"
    assert "肾虚" not in r.assessment.symptoms_canonical
    assert any("肾虚" in n for n in r.excluded_evidence)
    assert "腰膝酸软" in r.assessment.symptoms_canonical


def test_session_pulse_low_reliability_excluded():
    """普通用户自述脉象不参与辨证，静默排除（不弹质询）。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="脉象怎么知道的？", new_symptoms=["失眠", "心悸"],
                 pulse="脉弦细", pulse_from_doctor=False),
    ]))
    r = asyncio.run(session.step("睡不好心慌，我自己摸脉好像弦细"))
    assert all(c.kind in ("contradiction", "numeric_conflict") for c in r.challenges), \
        f"脉象不应触发质询: {[c.kind for c in r.challenges]}"
    assert session._credible_case().pulse == ""
    assert any("脉" in n for n in r.excluded_evidence)


def test_session_pulse_from_doctor_used():
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="收到。", new_symptoms=["失眠", "心悸"],
                 pulse="脉细弱", pulse_from_doctor=True),
    ]))
    asyncio.run(session.step("上周中医面诊说脉细弱"))
    assert session._credible_case().pulse == "脉细弱"


def test_session_temperature_numeric_conflict():
    """低烧与实测39℃冲突：质询并按实测修正。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="请确认体温。", new_symptoms=["低热"]),
    ]))
    r = asyncio.run(session.step("我有点低烧，刚才量了39度"))
    assert any(c.kind == "numeric_conflict" for c in r.challenges)
    joined = "、".join(r.assessment.symptoms_canonical)
    assert "低热" not in joined and "发热" in joined


def test_assessment_every_turn():
    """每轮都有即时辨证评估。"""
    session = ConsultSession(_scripted_llm([
        ChatTurn(reply="还有别的症状吗？", new_symptoms=["恶寒", "发热", "无汗"]),
        ChatTurn(reply="继续描述。", new_symptoms=["流清涕"]),
    ]))
    r1 = asyncio.run(session.step("怕冷发烧无汗"))
    assert r1.assessment is not None and "表寒" in r1.assessment.primary_syndrome
    r2 = asyncio.run(session.step("还流清鼻涕"))
    assert r2.assessment is not None and r2.syndrome_change == ""
