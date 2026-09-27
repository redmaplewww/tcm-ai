"""确定性中医推理引擎：不依赖LLM的病情推理核心。

流程：症状规范化（同义词/口语映射）-> 红旗检测 -> 证型辨证打分
-> 方剂/中成药/调护关联 -> 十问歌追问生成 -> 特殊人群提示。

该引擎是问诊正确性的基线保障；LLM（pydantic-ai Agent）在其之上做
对话编排与解释性输出，推理结论以本引擎为准。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from . import data
from .models import (
    ConsultationResult,
    FollowUpQuestion,
    HerbSafetyResult,
    PatientCase,
    RedFlag,
    SyndromeCandidate,
)

_LEVEL_ORDER = {"emergency": 3, "urgent": 2, "see_doctor": 1}

# 辨证体系呈现偏好：八纲/脏腑证名为通用表述层（如"风寒束表证"比"太阳伤寒证"
# 更适合作为主证名呈现），对得分接近的候选给予小幅加分，仅影响排序不影响命中。
_SYSTEM_BONUS = {
    "八纲辨证": 1.5,
    "脏腑辨证": 1.5,
    "气血津液辨证": 1.0,
    "脏腑辨证（兼证）": 0.5,
}

# 十问歌问诊维度：(维度名, 关键词, 追问问题)
_FOLLOWUP_DIMENSIONS: list[tuple[str, list[str], str]] = [
    ("寒热", ["发热", "恶寒", "畏寒", "怕冷", "发烧", "寒热", "潮热", "低热"],
     "您近期有没有发热或怕冷的感觉？"),
    ("汗", ["出汗", "自汗", "盗汗", "无汗", "汗多"],
     "您出汗情况如何（白天稍动就出汗/夜间睡着出汗/基本不出汗）？"),
    ("头身", ["头痛", "头晕", "身痛", "肢体", "腰", "关节", "耳鸣"],
     "有没有头痛、头晕、身体或关节酸痛？"),
    ("二便", ["大便", "小便", "便秘", "泄泻", "便溏", "尿", "腹泻"],
     "大便和小便情况如何（次数、干稀、颜色）？"),
    ("饮食口味", ["食欲", "食量", "口渴", "口干", "口苦", "口淡", "饮食", "胃口", "食少"],
     "胃口怎么样？有无口渴、口苦、口淡等口味异常？"),
    ("胸腹", ["胸闷", "胸痛", "心悸", "腹胀", "腹痛", "胃痛", "胁"],
     "胸口、胃脘或腹部有没有胀闷疼痛等不适？"),
    ("睡眠", ["失眠", "睡眠", "入睡", "多梦", "易醒", "嗜睡"],
     "睡眠怎么样（入睡难不难、是否易醒多梦）？"),
    ("情志", ["情绪", "抑郁", "烦躁", "易怒", "焦虑", "叹气"],
     "最近情绪状态如何（是否压力大、烦躁或情绪低落）？"),
    ("舌象", ["舌"],
     "方便的话请对镜观察舌头：舌质偏红还是偏淡？有无齿痕、裂纹？舌苔厚薄与颜色？"),
]

_GYNE_DIMENSION: tuple[str, list[str], str] = (
    "月经带下", ["月经", "经期", "痛经", "经量", "带下", "白带"],
    "月经周期、量、颜色如何？有无痛经？白带是否正常？",
)


# ---------------------------------------------------------------- 规范化

def normalize_symptoms(raw: list[str]) -> list[str]:
    """把口语/俗称症状映射为规范术语并去重。"""
    syn, collo = data.symptom_maps()
    merged = {**syn, **collo}
    out: list[str] = []
    for term in raw:
        t = (term or "").strip()
        if not t:
            continue
        if t in merged:
            for m in merged[t]:
                if m and m not in out:
                    out.append(m)
        elif t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------- 红旗检测

def check_red_flags(texts: list[str]) -> list[RedFlag]:
    """对全部输入文本做红旗规则匹配，按危险等级降序返回。"""
    blob = " ".join(x for x in texts if x)
    flags: list[RedFlag] = []
    for rule in data.red_flags():
        any_of = [str(p) for p in rule.get("any_of", []) if p]
        all_of = [str(p) for p in rule.get("all_of", []) if p]
        if not any_of and not all_of:
            continue
        matched = [p for p in any_of if p in blob]
        any_ok = bool(matched) if any_of else True
        all_ok = all(p in blob for p in all_of)
        if any_ok and all_ok:
            flags.append(
                RedFlag(
                    id=str(rule.get("id", "")),
                    level=str(rule.get("level", "urgent")),
                    name=str(rule.get("name", "")),
                    matched=matched + [p for p in all_of],
                    advice=str(rule.get("advice", "")),
                    note=str(rule.get("note", "")),
                )
            )
    flags.sort(key=lambda f: _LEVEL_ORDER.get(f.level, 0), reverse=True)
    return flags


# ---------------------------------------------------------------- 辨证打分

def _sym_hit(pattern: str, terms: list[str]) -> bool:
    """症状词条匹配。

    方向A（词条包含症状）：pattern为"恶寒发热"式复合词条、用户说"恶寒"时命中；
    方向B（症状包含词条）：用户说"午后骨蒸潮热"式更长表述、pattern为"潮热"时
    命中，但要求pattern长度≥3，避免"五心烦热"误命中"心烦"这类假阳性
    （口语变体由symptom_synonyms.json规范化处理）。
    """
    p = (pattern or "").strip()
    if not p:
        return False
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        if p == t:
            return True
        if len(t) >= 2 and t in p:
            return True
        if len(p) >= 3 and p in t:
            return True
    return False


def _feature_hit(rule: str, given: str) -> int:
    """舌/脉匹配强度：0未提供或不符，1部分相符，2相符。"""
    r = (rule or "").replace("，", "").replace("、", "").replace(" ", "")
    g = (given or "").replace("，", "").replace("、", "").replace(" ", "")
    if not r or not g:
        return 0
    if r in g or g in r:
        return 2
    grams = {r[i:i + 2] for i in range(len(r) - 1)}
    if any(gm in g for gm in grams):
        return 1
    return 0


def differentiate(
    symptoms: list[str],
    tongue: str = "",
    pulse: str = "",
    topn: int = 3,
) -> list[SyndromeCandidate]:
    """症状->证型打分辨证，返回候选证型（得分降序）。"""
    symptoms = [s for s in symptoms if s]
    results: list[SyndromeCandidate] = []
    for s in data.syndromes():
        keys = [str(k) for k in s.get("key_symptoms", []) if k]
        sups = [str(k) for k in s.get("supporting_symptoms", []) if k]
        if not keys:
            continue
        key_hits = [k for k in keys if _sym_hit(k, symptoms)]
        sup_hits = [x for x in sups if _sym_hit(x, symptoms)]
        # 至少命中1个关键症状，或命中≥2个次症，才作为候选
        if not key_hits and len(sup_hits) < 2:
            continue
        t_hit = _feature_hit(str(s.get("tongue", "")), tongue)
        p_hit = _feature_hit(str(s.get("pulse", "")), pulse)
        raw_score = 3.0 * len(key_hits) + 1.0 * len(sup_hits) + 2.0 * t_hit + 2.0 * p_hit
        score = raw_score + _SYSTEM_BONUS.get(str(s.get("system", "")), 0.0)
        max_score = 3.0 * max(len(keys), 1) + 1.0 * max(len(sups), 1) + 4.0
        confidence = min(0.95, round(raw_score / max_score, 3))
        missing = [k for k in keys if k not in key_hits]
        rationale = (
            f"命中关键症状{len(key_hits)}/{len(keys)}项：{'、'.join(key_hits) if key_hits else '无'}；"
            f"次症{len(sup_hits)}项；"
            + (f"舌脉相符；" if (t_hit == 2 or p_hit == 2) else ("舌脉部分相符；" if (t_hit or p_hit) else "舌脉未提供或不相符；"))
        )
        results.append(
            SyndromeCandidate(
                name=str(s.get("name", "")),
                system=str(s.get("system", "")),
                organ=str(s.get("organ", "")),
                score=round(score, 2),
                confidence=confidence,
                matched_symptoms=key_hits + sup_hits,
                missing_key_symptoms=missing,
                tongue=str(s.get("tongue", "")),
                pulse=str(s.get("pulse", "")),
                treatment=str(s.get("treatment", "")),
                formulas=[str(f) for f in s.get("formulas", []) if f],
                rationale=rationale,
            )
        )
    results.sort(key=lambda c: c.score, reverse=True)
    top = results[:topn]
    # 附上与第二候选的鉴别要点
    for i, cand in enumerate(top):
        src = data.get_syndrome(cand.name)
        diff = str((src or {}).get("differentiation", "")).strip()
        if diff:
            cand.rationale = cand.rationale + " 鉴别：" + diff
    return top


# ---------------------------------------------------------------- 疾病关联

def _name_segments(name: str) -> list[str]:
    """证名分段：'表寒证（风寒束表证）' -> ['表寒', '风寒束表']（去证字、按括号拆分）。"""
    n = (name or "").replace("（", "(").replace("）", ")")
    segs: list[str] = []
    for part in re.split(r"[()]", n):
        seg = part.strip().rstrip("证").strip()
        if seg and seg not in segs:
            segs.append(seg)
    return segs


def _match_disease(primary_syndrome: str, symptoms: list[str]) -> Optional[dict[str, Any]]:
    """按证型名匹配疾病（用于补充中成药与调护建议）。

    同一证型常出现在多个疾病分型中（如"风寒束表证"既是感冒也是荨麻疹的分型），
    择优规则：证名分段匹配质量（全等>包含）> 疾病intro与用户症状的重叠度 >
    疾病名更短（更基础）者优先。
    """
    segs = _name_segments(primary_syndrome)
    if not segs:
        return None
    best: Optional[dict[str, Any]] = None
    best_key: Optional[tuple] = None
    for d in data.diseases():
        d_quality = 0
        for st in d.get("syndrome_types", []):
            sc_segs = _name_segments(str(st.get("syndrome", "")))
            for seg in segs:
                for sc in sc_segs:
                    if seg == sc:
                        d_quality = max(d_quality, 2)
                    elif seg in sc or sc in seg:
                        d_quality = max(d_quality, 1)
        if d_quality == 0:
            continue
        intro = str(d.get("intro", ""))
        overlap = sum(1 for s in symptoms if s and s in intro)
        key = (-d_quality, -overlap, len(str(d.get("name", ""))))
        if best_key is None or key < best_key:
            best, best_key = d, key
    return best


# ---------------------------------------------------------------- 追问生成

def missing_dimensions(evidence_texts: list[str], sex: Optional[str] = None,
                       age: Optional[int] = None) -> list[FollowUpQuestion]:
    """按十问歌检查未覆盖的问诊维度，生成追问（妇科维度插在情志之前）。"""
    blob = " ".join(evidence_texts)
    dims = list(_FOLLOWUP_DIMENSIONS)
    if sex == "女" and (age is None or 12 <= (age or 0) <= 55):
        # 插在"情志"维度之前，保证不被截断
        pos = next(
            (i for i, (n, _, _) in enumerate(dims) if n == "情志"), len(dims)
        )
        dims.insert(pos, _GYNE_DIMENSION)
    questions: list[FollowUpQuestion] = []
    for name, kws, q in dims:
        if not any(k in blob for k in kws):
            questions.append(FollowUpQuestion(dimension=name, question=q))
    return questions[:9]


# ---------------------------------------------------------------- 配伍安全

def _herb_in_group(herb: str, group: list[str]) -> bool:
    """药名与规则组成员匹配（兼容'炙甘草'/'制附子'等炮制名）。"""
    h = (herb or "").strip()
    if not h:
        return False
    for member in group:
        m = str(member).strip()
        if not m:
            continue
        if h == m or (len(m) >= 2 and m in h) or (len(h) >= 2 and h in m):
            return True
    return False


def check_herb_combination(herbs: list[str], pregnant: bool = False) -> HerbSafetyResult:
    """十八反/十九畏配伍禁忌 + 妊娠禁忌 + 毒性标注检查。"""
    herbs = [h.strip() for h in herbs if h and h.strip()]
    conflicts: list[str] = []
    for rule in data.incompatibilities():
        group_a = [str(x) for x in rule.get("a", [])]
        group_b = [str(x) for x in rule.get("b", [])]
        hits_a = [h for h in herbs if _herb_in_group(h, group_a)]
        hits_b = [h for h in herbs if _herb_in_group(h, group_b)]
        for ha in hits_a:
            for hb in hits_b:
                if ha != hb:
                    conflicts.append(f"{ha}与{hb}相畏相反（{rule.get('type', '')}），禁止同用")
    # 去重
    conflicts = list(dict.fromkeys(conflicts))

    pregnancy_warnings: list[str] = []
    toxicity_notes: list[str] = []
    for h in herbs:
        info = data.get_herb(h)
        if not info:
            continue
        if info.get("toxicity"):
            toxicity_notes.append(f"{info.get('name')}：{info.get('toxicity')}之品，须医师指导使用")
        if pregnant:
            if info.get("pregnancy_forbidden"):
                pregnancy_warnings.append(f"{info.get('name')}：妊娠禁用")
            else:
                for c in info.get("cautions", []):
                    if "孕妇" in str(c) or "妊娠" in str(c):
                        pregnancy_warnings.append(f"{info.get('name')}：{c}")
                        break

    is_safe = not conflicts and not pregnancy_warnings
    parts = []
    if conflicts:
        parts.append("；".join(conflicts))
    if pregnancy_warnings:
        parts.append("；".join(pregnancy_warnings))
    if not parts:
        advice = "未检出十八反/十九畏配伍禁忌" + ("（妊娠期用药仍须医师评估）" if pregnant else "")
    else:
        advice = parts[0] + "。请务必由执业中医师审核后再用药。"
    return HerbSafetyResult(
        herbs=herbs,
        conflicts=conflicts,
        pregnancy_warnings=pregnancy_warnings,
        toxicity_notes=toxicity_notes,
        is_safe=is_safe,
        advice=advice,
    )


# ---------------------------------------------------------------- 特殊人群

def special_population_notes(case: PatientCase) -> list[str]:
    notes: list[str] = []
    if case.pregnant:
        notes.append("您处于妊娠期：用药必须由医师指导，妊娠禁忌药（破血、峻下、行气破滞及毒性药）不可自行服用；"
                     "合谷、三阴交、肩井、昆仑等穴孕期禁针，相关疗法须由执业医师操作。")
    age = case.age
    if age is not None and age <= 6:
        notes.append("患儿年龄小（稚阴稚阳之体，易虚易实、传变迅速）：小儿用药剂量与成人不同，"
                     "务必在儿科医师指导下用药，出现精神萎靡、高热不退等情况立即就医。")
    if age is not None and age >= 70:
        notes.append("高龄人群多有基础疾病、多药并用，中药与西药可能存在相互作用，"
                     "用药前请告知医师全部在服药物，由医师评估。")
    return notes


# ---------------------------------------------------------------- 主推理流程

def consult(case: PatientCase, constitution: str = "") -> ConsultationResult:
    """离线完整问诊推理（不依赖LLM），是Agent问诊的确定性基线。

    constitution: 用户体质倾向（如"阴虚质"）时附加生活习惯建议。
    """
    from .profile import lifestyle_advice
    canonical = normalize_symptoms(case.symptoms)
    evidence = [case.free_text, " ".join(case.symptoms), " ".join(canonical)]
    flags = check_red_flags(evidence)
    is_emergency = any(f.level == "emergency" for f in flags)
    emergency_advice = next(
        (f"{f.name}：{f.advice}" for f in flags if f.level == "emergency"), ""
    )

    result = ConsultationResult(
        red_flags=flags,
        is_emergency=is_emergency,
        emergency_advice=emergency_advice,
        symptoms_canonical=canonical,
    )

    # 急症：只输出就医指引，不做辨证建议
    if is_emergency:
        result.followup_questions = []
        return result

    candidates = differentiate(canonical, case.tongue, case.pulse)
    result.syndromes = candidates
    if candidates:
        top = candidates[0]
        result.primary_syndrome = top.name
        result.treatment_principle = top.treatment
        result.formula_suggestions = top.formulas
        disease = _match_disease(top.name, canonical)
        if disease:
            core_segs = _name_segments(top.name)
            otc: list[str] = []
            for st in disease.get("syndrome_types", []):
                sc_segs = _name_segments(str(st.get("syndrome", "")))
                matched = any(
                    seg == sc or seg in sc or sc in seg
                    for seg in core_segs for sc in sc_segs
                )
                if matched:
                    otc.extend(str(x) for x in st.get("otc", []) if x)
            result.otc_suggestions = list(dict.fromkeys(otc))
            result.care_advice = [str(c) for c in disease.get("care", []) if c]

    result.followup_questions = missing_dimensions(
        evidence, sex=case.sex, age=case.age
    )
    result.special_population_notes = special_population_notes(case)
    # 体质建议：优先用户画像体质；无画像时从本次主证推导（慢性证型才映射）
    if not constitution and result.primary_syndrome:
        from .profile import syndrome_to_constitution
        constitution = syndrome_to_constitution(result.primary_syndrome) or ""
    if constitution:
        result.lifestyle_advice = lifestyle_advice(constitution)
    return result
