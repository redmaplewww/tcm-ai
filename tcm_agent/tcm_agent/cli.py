"""命令行入口：tcm query / tcm ask / tcm consult / tcm demo / tcm info。

用法示例：
  python -m tcm_agent query "什么是阴虚"                      # 离线知识库检索
  python -m tcm_agent ask "什么是阴虚？有什么表现？"           # 真实LLM问答（需凭据）
  python -m tcm_agent consult --symptoms "怕冷,发烧,无汗" ...  # 离线确定性推理
  python -m tcm_agent consult --llm --symptoms "怕冷,发烧,清鼻涕" --pulse "浮紧"
  python -m tcm_agent demo
  python -m tcm_agent info

LLM凭据：环境变量 TCM_LLM_KEY / TCM_LLM_BASE_URL / TCM_LLM_MODEL，
或自动复用本机ZCode CLI登录的GLM凭据（~/.zcode/v2/config.json）。
第一阶段仅支持文字交互。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional, Sequence

from . import reasoning
from .kb import get_kb
from .models import ConsultationResult, PatientCase


def _ensure_utf8() -> None:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------- 子命令

def _case_prompt(case: PatientCase) -> str:
    """把患者信息组装成问诊Agent的中文提示（第一阶段纯文字）。"""
    parts = []
    if case.sex or case.age is not None:
        seg = f"患者信息：{case.sex or '性别未知'}，{case.age if case.age is not None else '年龄未知'}岁"
        if case.pregnant:
            seg += "，怀孕中"
        parts.append(seg)
    if case.free_text:
        parts.append(f"病情自述：{case.free_text}")
    parts.append(f"主要症状：{'、'.join(case.symptoms)}")
    if case.tongue:
        parts.append(f"舌象：{case.tongue}")
    if case.pulse:
        parts.append(f"脉象：{case.pulse}")
    else:
        parts.append("脉象：未提供（线上问诊无法切脉）")
    parts.append("请先做红旗急症检测，再进行辨证分析。")
    return "\n".join(parts)


def cmd_ask(args: argparse.Namespace) -> int:
    """真实LLM知识问答（query_agent）。"""
    from . import llm
    if not llm.llm_available():
        print("未检测到LLM凭据：请设置 TCM_LLM_KEY/TCM_LLM_BASE_URL，"
              "或在已登录ZCode CLI的机器上运行。")
        return 2
    from .agents import build_query_agent
    agent = build_query_agent()
    print(f"模型：{llm.llm_credentials()[2]}（{llm.llm_credentials()[1]}）")
    print(f"问题：{args.question}")
    print("=" * 60)
    try:
        result = asyncio.run(agent.run(args.question))
    except Exception as e:  # noqa: BLE001 - CLI需要展示任何LLM错误
        print(f"LLM调用失败：{e}")
        return 1
    out = result.output
    print(out.answer)
    if out.key_points:
        print("\n要点：")
        for kp in out.key_points:
            print(f"  · {kp}")
    if out.sources:
        print("\n知识来源：")
        for s in out.sources:
            loc = s.file + (f" :: {s.section}" if s.section else "")
            print(f"  · {loc}")
    print("-" * 60)
    print(f"免责：{out.disclaimer}")
    return 0


def cmd_consult_llm(case: PatientCase) -> Optional[ConsultationResult]:
    """真实LLM问诊（consult_agent）；失败时返回None。"""
    from . import llm
    if not llm.llm_available():
        print("未检测到LLM凭据，--llm不可用，已回退离线推理。")
        return None
    from .agents import build_consult_agent
    agent = build_consult_agent()
    try:
        result = asyncio.run(agent.run(_case_prompt(case)))
        return result.output
    except Exception as e:  # noqa: BLE001
        print(f"LLM调用失败：{e}")
        return None


def cmd_query(args: argparse.Namespace) -> int:
    kb = get_kb()
    results = kb.search(args.question, k=args.k)
    print(f"检索：{args.question}（命中{len(results)}条）")
    print("=" * 60)
    for i, ch in enumerate(results, 1):
        loc = ch.file + (f" :: {ch.section}" if ch.section else "")
        print(f"[{i}] {ch.title}（{loc}）")
        print(f"    {ch.excerpt(400)}")
        print("-" * 60)
    if not results:
        print("知识库中未检索到相关内容。")
    return 0


def _print_consult(result: ConsultationResult) -> None:
    if result.is_emergency:
        print("⚠️  检测到急症红旗，请立即就医！")
        print(f"   {result.emergency_advice}")
        print("=" * 60)
    print(f"规范化症状：{'、'.join(result.symptoms_canonical) or '（无）'}")
    print("=" * 60)
    if result.syndromes:
        for i, c in enumerate(result.syndromes, 1):
            tag = "【主】" if i == 1 else "【备】"
            print(f"{tag} {c.name}（{c.system}｜置信度{c.confidence:.0%}｜{c.organ}）")
            print(f"    治法：{c.treatment}")
            print(f"    匹配症状：{'、'.join(c.matched_symptoms)}")
            print(f"    舌脉：{c.tongue} / {c.pulse}")
            print(f"    依据：{c.rationale}")
            print("-" * 60)
        print(f"最可能证型：{result.primary_syndrome}")
        print(f"治法：{result.treatment_principle}")
        if result.formula_suggestions:
            print(f"代表方剂（须经执业中医师辨证）：{'、'.join(result.formula_suggestions)}")
        if result.otc_suggestions:
            print(f"常用中成药参考：{'、'.join(result.otc_suggestions)}")
    else:
        print("症状信息不足，暂无法辨证，请补充以下信息：")
    print("=" * 60)
    if result.followup_questions:
        print("追问（十问歌未覆盖维度）：")
        for q in result.followup_questions:
            print(f"  ·（{q.dimension}）{q.question}")
    if result.care_advice:
        print("调护建议：")
        for c in result.care_advice:
            print(f"  · {c}")
    if result.special_population_notes:
        print("特殊人群提示：")
        for n in result.special_population_notes:
            print(f"  · {n}")
    if result.red_flags and not result.is_emergency:
        print("就医提示：")
        for f in result.red_flags:
            print(f"  · [{f.level}] {f.name}：{f.advice}")
    print("=" * 60)
    print(f"免责：{result.disclaimer}")
    print()


def cmd_consult(args: argparse.Namespace) -> int:
    symptoms = [s.strip() for s in args.symptoms.replace("，", ",").split(",") if s.strip()]
    case = PatientCase(
        symptoms=symptoms,
        free_text=args.text or "",
        tongue=args.tongue or "",
        pulse=args.pulse or "",
        sex=args.sex or None,
        age=args.age,
        pregnant=args.pregnant,
    )
    result = None
    if getattr(args, "llm", False):
        print("（--llm 模式：真实LLM辨证，附离线引擎对照）")
        print("=" * 60)
        llm_result = cmd_consult_llm(case)
        if llm_result is not None:
            result = llm_result
            if args.json:
                print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
                return 0
            _print_consult(result)
            print()
            print("◆ 离线确定性引擎对照：")
            print("=" * 60)
            _print_consult(reasoning.consult(case))
            return 0
        print("回退到离线确定性推理：")
        print("=" * 60)
    result = reasoning.consult(case)
    if args.json:
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    else:
        _print_consult(result)
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    cases = [
        ("风寒表证", PatientCase(
            symptoms=["怕冷", "发烧", "无汗", "头痛", "身痛", "鼻塞", "清鼻涕"],
            tongue="舌淡红苔薄白", pulse="脉浮紧", sex="男", age=30,
        )),
        ("失眠（心脾两虚方向）", PatientCase(
            symptoms=["心慌", "睡不着", "多梦", "健忘", "食少", "疲乏"],
            tongue="舌淡", sex="女", age=35,
        )),
        ("急症红旗（胸痛）", PatientCase(
            symptoms=["胸口像压了石头", "胸痛放射到左臂", "出冷汗"], sex="男", age=58,
        )),
    ]
    for title, case in cases:
        print(f"◆ 演示病例：{title}")
        print(f"  输入症状：{'、'.join(case.symptoms)}；舌：{case.tongue or '未提供'}；脉：{case.pulse or '未提供'}")
        print("=" * 60)
        _print_consult(reasoning.consult(case))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """交互式多轮问诊：LLM对话收集症状，随时/诊断 出结果。"""
    from . import llm
    from .chat_session import ConsultSession, make_agent_llm_fn
    from .models import PatientCase

    case = PatientCase(sex=args.sex or None, age=args.age, pregnant=args.pregnant)
    print("中医问诊（多轮文字对话）——描述您的症状，助手会逐步追问。")
    print("命令：/诊断（立即辨证） /重置（清空重来） /退出")
    use_llm = llm.llm_available()
    if use_llm:
        print(f"[LLM模式：{llm.llm_credentials()[2]}]")
        session = ConsultSession(make_agent_llm_fn(), case=case)
    else:
        print("[离线问卷模式：未检测到LLM凭据，按十问歌逐项询问]")
        session = ConsultSession(case=case)

    first = True
    while True:
        try:
            if first and use_llm:
                user_text = input("\n您：").strip()
            elif first:
                user_text = input("\n您（先说说主要不舒服的地方）：").strip()
            else:
                user_text = input("\n您：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        first = False
        if not user_text:
            continue
        if user_text in ("/退出", "/quit", "/exit"):
            print("再见。")
            return 0
        if user_text == "/重置":
            session = ConsultSession(
                make_agent_llm_fn() if use_llm else None, case=PatientCase(
                    sex=args.sex or None, age=args.age, pregnant=args.pregnant)
            )
            print("[已重置会话]")
            continue
        force = user_text in ("/诊断", "/分析", "/辨证")

        result = asyncio.run(session.step(
            "请立即给出辨证分析" if force else user_text, force_analysis=force
        ))
        print(f"\n助手：{result.reply}")
        for c in result.challenges:
            print(f"[需核实] {c.question}")
        if result.excluded_evidence:
            print(f"[存疑/说明] {'；'.join(result.excluded_evidence)}")
        if result.assessment is not None and result.final is None:
            top = result.assessment
            if top.primary_syndrome:
                conf = top.syndromes[0].confidence if top.syndromes else 0
                print(f"[即时辨证] {top.primary_syndrome}（置信度{conf:.0%}，信息仍在收集中）")
        if result.syndrome_change:
            print(f"[证型更新] {result.syndrome_change}")
        if result.is_emergency:
            print("=" * 60)
            print(f"⚠️  {result.emergency_advice}")
            print("=" * 60)
            return 0
        if result.final is not None:
            print("=" * 60)
            _print_consult(result.final)
            return 0


def cmd_info(_args: argparse.Namespace) -> int:
    from . import data
    kb = get_kb()
    stats = kb.stats()
    print("分级Markdown知识库统计：")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print("结构化数据层：")
    print(f"  证型 syndromes: {len(data.syndromes())}")
    print(f"  中药 herbs: {len(data.herbs())}")
    print(f"  方剂 formulas: {len(data.formulas())}")
    print(f"  穴位 acupoints: {len(data.acupoints())}")
    print(f"  疾病 diseases: {len(data.diseases())}")
    print(f"  红旗 red_flags: {len(data.red_flags())}")
    print(f"  术语 glossary: {len(data.glossary())}")
    return 0


# ---------------------------------------------------------------- 入口

def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        prog="tcm", description="中医AI问诊与查询（离线知识库+确定性推理引擎）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="知识库检索查询（离线）")
    p_query.add_argument("question", help="查询问题")
    p_query.add_argument("-k", type=int, default=5, help="返回条数")
    p_query.set_defaults(func=cmd_query)

    p_ask = sub.add_parser("ask", help="真实LLM知识问答（需LLM凭据）")
    p_ask.add_argument("question", help="问题，如'风寒和风热感冒怎么区别'")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式多轮问诊对话（文字）")
    p_chat.add_argument("--sex", choices=["男", "女"], default="")
    p_chat.add_argument("--age", type=int, default=None)
    p_chat.add_argument("--pregnant", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    p_consult = sub.add_parser("consult", help="病情推理（辨证参考）")
    p_consult.add_argument("--symptoms", required=True, help="症状列表，逗号分隔（口语可）")
    p_consult.add_argument("--text", default="", help="病情补充描述")
    p_consult.add_argument("--tongue", default="", help="舌象，如'舌红苔黄'")
    p_consult.add_argument("--pulse", default="", help="脉象，如'脉弦'")
    p_consult.add_argument("--sex", choices=["男", "女"], default="")
    p_consult.add_argument("--age", type=int, default=None)
    p_consult.add_argument("--pregnant", action="store_true", help="是否怀孕")
    p_consult.add_argument("--llm", action="store_true", help="使用真实LLM辨证（附离线引擎对照）")
    p_consult.add_argument("--json", action="store_true", help="输出JSON")
    p_consult.set_defaults(func=cmd_consult)

    p_demo = sub.add_parser("demo", help="运行内置演示病例")
    p_demo.set_defaults(func=cmd_demo)

    p_info = sub.add_parser("info", help="知识库与数据统计")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
