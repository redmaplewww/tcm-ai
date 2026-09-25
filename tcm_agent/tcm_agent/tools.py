"""供LLM调用的工具函数（知识查询 + 推理引擎封装）。

工具返回格式化中文文本；所有查询均基于分级知识库与结构化数据层，
检索不到时明确说明，避免幻觉。
"""

from __future__ import annotations

import json
from typing import Any

from . import data
from . import reasoning
from .kb import format_search_results, get_kb


def _fmt_entry(entry: dict[str, Any], fields: list[tuple[str, str]]) -> str:
    lines = []
    for key, label in fields:
        v = entry.get(key)
        if v in (None, "", []):
            continue
        if isinstance(v, list):
            v = "、".join(str(x) for x in v)
        lines.append(f"{label}：{v}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 知识检索

def search_knowledge_base(query: str, k: int = 5) -> str:
    """在分级Markdown知识库中检索与query最相关的章节。"""
    return format_search_results(get_kb().search(query, k=min(max(k, 1), 10)))


def get_glossary_term(term: str) -> str:
    """查询中医术语定义（阴阳、藏象、辨证论治等）。"""
    e = data.get_glossary_term(term)
    if not e:
        return f"术语表中未收录「{term}」，请勿编造定义；可尝试用search_knowledge_base检索。"
    related = "、".join(str(x) for x in e.get("related_terms", []))
    return _fmt_entry(e, [("term", "术语"), ("definition", "定义")]) + (
        f"\n相关术语：{related}" if related else ""
    )


def get_herb_info(name: str) -> str:
    """查询中药信息：性味归经、功效、主治、用量、禁忌。"""
    e = data.get_herb(name)
    if not e:
        return f"中药库中未找到「{name}」；请勿编造药性，可建议咨询医师。"
    toxicity = e.get("toxicity") or "无毒（未标注毒性）"
    preg = "妊娠禁用" if e.get("pregnancy_forbidden") else "未见妊娠禁用标记（仍须医师评估）"
    return (
        f"【{e.get('name')}】（{e.get('category', '')}）\n"
        + _fmt_entry(e, [
            ("nature", "性"), ("flavor", "味"), ("meridians", "归经"),
            ("functions", "功效"), ("indications", "主治"), ("dosage", "用量"),
            ("cautions", "使用注意"),
        ])
        + f"\n毒性：{toxicity}；妊娠：{preg}"
    )


def get_formula_info(name: str) -> str:
    """查询方剂/中成药信息：组成、功用、主治证、禁忌。"""
    e = data.get_formula(name)
    if not e:
        return f"方剂库中未找到「{name}」；请勿编造组成，可建议咨询医师。"
    kind = "中成药(OTC)" if e.get("otc") else "经典方剂"
    return (
        f"【{e.get('name')}】（{kind}，{e.get('category', '')}）\n"
        + _fmt_entry(e, [
            ("source", "出处"), ("composition", "组成"), ("functions", "功用"),
            ("indications", "主治"), ("syndromes", "对应证型"),
            ("cautions", "禁忌与注意"), ("notes", "要点"),
        ])
    )


def get_acupoint_info(name: str) -> str:
    """查询穴位信息：定位、归经、主治、针法、孕妇禁忌。"""
    e = data.get_acupoint(name)
    if not e:
        return f"穴位库中未找到「{name}」；请勿编造定位，可建议咨询针灸医师。"
    preg = "【孕妇禁针/慎用】" if e.get("pregnancy_caution") else "未见孕妇禁忌标记"
    return (
        f"【{e.get('name')}】（{e.get('code', '')} {e.get('meridian', '')}）{preg}\n"
        + _fmt_entry(e, [
            ("location", "定位"), ("indications", "主治"), ("method", "操作"),
            ("caution_notes", "注意"),
        ])
    )


def get_syndrome_info(name: str) -> str:
    """查询证型信息：关键症状、舌脉、治法、代表方、鉴别要点。"""
    e = data.get_syndrome(name)
    if not e:
        return f"证型库中未找到「{name}」；请勿编造证型内容。"
    return _fmt_entry(e, [
        ("name", "证型"), ("system", "辨证体系"), ("organ", "病位"),
        ("key_symptoms", "关键症状"), ("supporting_symptoms", "次症"),
        ("tongue", "舌象"), ("pulse", "脉象"), ("treatment", "治法"),
        ("formulas", "代表方"), ("differentiation", "鉴别"),
        ("related_diseases", "相关疾病"),
    ])


# ---------------------------------------------------------------- 推理引擎

def check_red_flag_risk(text: str) -> str:
    """红旗症状检测：判断用户描述是否包含需立即/尽快就医的急症表现。"""
    flags = reasoning.check_red_flags([text])
    if not flags:
        return "未检测到红旗症状。"
    lines = []
    for f in flags:
        lines.append(
            f"[{f.level}] {f.name}（命中：{'、'.join(f.matched)}）\n建议：{f.advice}"
        )
    return "\n".join(lines)


def run_syndrome_reasoning(symptoms: list[str], tongue: str = "", pulse: str = "") -> str:
    """确定性辨证：传入症状列表，返回候选证型打分与治法方剂参考。

    symptoms请传用户的原始症状词（口语表述即可，如"吃不下饭"），引擎会自动
    做同义词/口语规范化；不要预先合并或改写成其他术语。
    """
    canonical = reasoning.normalize_symptoms(symptoms)
    cands = reasoning.differentiate(canonical, tongue, pulse)
    if not cands:
        return json.dumps(
            {"规范化症状": canonical, "候选证型": [],
             "提示": "症状信息不足或未见匹配证型，请按十问歌追问补充"},
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "规范化症状": canonical,
            "候选证型": [
                {
                    "证型": c.name,
                    "置信度": c.confidence,
                    "匹配症状": c.matched_symptoms,
                    "舌": c.tongue,
                    "脉": c.pulse,
                    "治法": c.treatment,
                    "代表方": c.formulas,
                    "依据": c.rationale,
                }
                for c in cands
            ],
        },
        ensure_ascii=False,
        indent=1,
    )


def ask_followup_questions(symptoms: list[str], sex: str = "", age: int = 0) -> str:
    """按十问歌生成尚未覆盖维度的追问问题。"""
    qs = reasoning.missing_dimensions(
        symptoms, sex=sex or None, age=age or None
    )
    if not qs:
        return "问诊维度已基本覆盖。"
    return "\n".join(f"（{q.dimension}）{q.question}" for q in qs)


def check_herb_safety(herbs: list[str], pregnant: bool = False) -> str:
    """中药配伍安全检查：十八反/十九畏禁忌、妊娠禁忌、毒性标注。

    Args:
        herbs: 药名列表，如["甘草","甘遂"]，可为方剂组成药物。
        pregnant: 用户是否怀孕（用于妊娠禁忌检查）。
    """
    result = reasoning.check_herb_combination(herbs, pregnant=pregnant)
    lines = [f"检查药物：{'、'.join(result.herbs)}"]
    if result.conflicts:
        lines.append("⚠️ 配伍禁忌：" + "；".join(result.conflicts))
    if result.pregnancy_warnings:
        lines.append("⚠️ 妊娠警示：" + "；".join(result.pregnancy_warnings))
    if result.toxicity_notes:
        lines.append("毒性标注：" + "；".join(result.toxicity_notes))
    lines.append("结论：" + result.advice)
    return "\n".join(lines)
