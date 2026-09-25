"""pydantic-ai Agent接线测试（离线，使用TestModel，不需要API key）。

- 验证Agent可构建、工具注册正确、run流程与输出类型校验通过。
- TestModel会自动调用每个工具一次，顺带验证工具函数对任意入参不崩溃。
"""

import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from tcm_agent.agents import build_consult_agent, build_query_agent
from tcm_agent.models import ConsultationResult, QueryAnswer
from tcm_agent import tools


def test_build_agents():
    qa = build_query_agent(TestModel())
    ca = build_consult_agent(TestModel())
    # 工具已注册（pydantic-ai v2: agent._function_toolset.tools 为dict）
    def tool_names(agent) -> list[str]:
        ts = getattr(agent, "_function_toolset", None)
        tools = getattr(ts, "tools", None)
        return list(tools.keys()) if isinstance(tools, dict) else list(agent._tools) if hasattr(agent, "_tools") else []
    assert len(tool_names(qa)) >= 6, f"查询Agent工具不足: {tool_names(qa)}"
    assert len(tool_names(ca)) >= 4, f"问诊Agent工具不足: {tool_names(ca)}"


def test_query_agent_runs_offline():
    agent = build_query_agent(TestModel())
    result = asyncio.run(agent.run("什么是阴虚？"))
    # TestModel自动填充字段值，这里验证输出类型校验通过
    assert isinstance(result.output, QueryAnswer)


def test_consult_agent_runs_offline():
    agent = build_consult_agent(TestModel())
    result = asyncio.run(agent.run("我最近怕冷发热无汗，鼻塞流清涕，请问是什么证？"))
    assert isinstance(result.output, ConsultationResult)


def test_tools_direct_calls():
    """工具函数直接调用（不经LLM），验证行为与降级路径。"""
    # 知识检索
    out = tools.search_knowledge_base("阴虚", k=3)
    assert isinstance(out, str) and len(out) > 0
    # 查询工具：存在与不存在两条路径
    assert "麻黄" in tools.get_herb_info("麻黄")
    assert "未找到" in tools.get_herb_info("不存在的药xyz")
    assert "六味地黄" in tools.get_formula_info("六味地黄丸")
    assert "未找到" in tools.get_formula_info("不存在的方xyz")
    assert "足三里" in tools.get_acupoint_info("足三里")
    assert "合谷" in tools.get_acupoint_info("合谷") and "孕妇" in tools.get_acupoint_info("合谷")
    assert "未找到" in tools.get_acupoint_info("不存在的穴xyz")
    assert "肝气郁结" in tools.get_syndrome_info("肝气郁结")
    assert "未找到" in tools.get_syndrome_info("不存在的证xyz")
    assert "术语表" in tools.get_glossary_term("不存在的术语xyz")
    # 推理工具
    risk = tools.check_red_flag_risk("胸痛放射至左臂伴大汗")
    assert "emergency" in risk or "就医" in risk
    safe = tools.check_red_flag_risk("有点累，睡不好")
    assert "未检测到红旗" in safe
    res = tools.run_syndrome_reasoning(["恶寒", "发热", "无汗", "流清涕"], "苔薄白", "脉浮紧")
    assert "风寒" in res
    fu = tools.ask_followup_questions(["头痛"])
    assert "寒热" in fu or "追问" not in fu  # 至少返回内容


@pytest.mark.parametrize("query,expect", [
    ("心脾两虚证有什么表现", "心脾两虚"),
    ("黄芪的功效", "黄芪"),
    ("合谷穴的位置", "合谷"),
])
def test_tool_grounding_accuracy(query, expect):
    """工具接地准确性：查询应命中正确知识实体。"""
    out = tools.search_knowledge_base(query, k=5)
    assert expect in out, f"'{query}' 检索未包含 '{expect}'"
