"""脉象→病理方向推理：典籍主病 + 证型库反查 + 验证清单 + 红旗。

能力边界（重要）：
- 本模块输出的是"证候假说与风险提示"，不是病理诊断。仅凭脉象不定证
  （四诊合参原则，参仲景"脉证并治"）；症状缺失时只列假设与验证问题，
  症状提供时才调用 differentiate() 出候选证型。
- 危候脉（疾/微/散/代及促而无力）检出即触发就医建议，不参与常规辨证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import data
from .kb import get_kb

#: 危候脉：检出即建议就医（参切诊-脉诊.md"AI应用要点"）
DANGER_PULSES = {"疾", "微", "散", "代"}

#: 结促类：现代对应心律不齐，建议ECG核实
ARRHYTHMIA_PULSES = {"结", "促", "代"}


def parse_pulse_names(pulse_text: str) -> list[str]:
    """从"脉数滑"式文本解析脉名列表（按规则库收录名匹配）。

    截断括号注释（如"脉平（至数节律正常）"只取"脉平"），避免注释文字
    中的单字误匹配为脉名。
    """
    known = {p["name"] for p in data.pulses()}
    text = (pulse_text or "").replace("脉", "")
    for cut in ("（", "(", "：", ":"):
        text = text.split(cut)[0]
    names = []
    for ch in text:
        if ch in known:
            names.append(ch)
    return names


@dataclass
class PulseInterpretation:
    """脉象推理结果：主病、证候假说、验证问题与红旗。"""

    pulse_text: str
    components: list[str] = field(default_factory=list)
    indications: list[dict] = field(default_factory=list)       # 各脉主病（规则库）
    classical: list[dict] = field(default_factory=list)         # 典籍检索摘录
    candidate_syndromes: list[dict] = field(default_factory=list)  # 证候假说
    verified_syndromes: list[dict] = field(default_factory=list)   # 症状验证后的候选
    verification_questions: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    mode: str = "hypothesis"  # hypothesis(仅脉象) | verified(脉+症状)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pulse": self.pulse_text, "components": self.components, "mode": self.mode,
            "indications": self.indications, "classical": self.classical,
            "candidate_syndromes": self.candidate_syndromes,
            "verified_syndromes": self.verified_syndromes,
            "verification_questions": self.verification_questions,
            "red_flags": self.red_flags,
            "disclaimer": "以上为脉象推理参考，不构成诊断；请执业中医师四诊合参辨证施治。",
        }


def _syndrome_reverse_lookup(names: list[str], topn: int = 5) -> list[dict]:
    """证型库反查：证型pulse字段与脉名集合的重叠计数排序。"""
    if not names:
        return []
    scored = []
    for s in data.syndromes():
        spulse = str(s.get("pulse", ""))
        hits = [n for n in names if n in spulse]
        if not hits:
            continue
        scored.append((len(hits), len(spulse), s))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for hits, _, s in scored[:topn]:
        out.append({
            "name": s.get("name", ""), "system": s.get("system", ""),
            "pulse": s.get("pulse", ""),
            "key_symptoms": [str(k) for k in s.get("key_symptoms", [])][:6],
            "tongue": s.get("tongue", ""),
            "matched": hits,
        })
    return out


def _collect_questions(synds: list[dict], limit: int = 8) -> list[str]:
    """由候选证型的关键症状生成验证问题（去重，保留高频）。"""
    seen = {}
    for s in synds:
        for k in s.get("key_symptoms", []):
            seen[k] = seen.get(k, 0) + 1
    ordered = sorted(seen.items(), key=lambda kv: -kv[1])
    return [f"是否伴有「{k}」？" for k, _ in ordered[:limit]]


def interpret_pulse(
    pulse_text: str,
    symptoms: Optional[list[str]] = None,
    include_classical: bool = True,
) -> PulseInterpretation:
    """脉象推理入口。

    - symptoms=None：假说模式，输出证候假说与验证问题（不定证）；
    - symptoms提供：调用 differentiate() 出候选证型（脉证互参打分）。
    """
    names = parse_pulse_names(pulse_text)
    interp = PulseInterpretation(pulse_text=pulse_text, components=names)

    # 1) 各脉主病（规则库 pulses.json）
    for n in names:
        p = data.get_pulse(n)
        if p:
            interp.indications.append({
                "name": n, "definition": p.get("definition", ""),
                "main_indications": p.get("main_indications", ""),
                "combined_examples": p.get("combined_examples", []),
            })

    # 2) 红旗
    danger = [n for n in names if n in DANGER_PULSES]
    for n in danger:
        interp.red_flags.append(
            f"危候脉「{n}」：中医主脏气衰微/元气将脱；现代对应严重心律失常或循环衰竭可能，建议立即就医评估。")
    arr = [n for n in names if n in ARRHYTHMIA_PULSES and n not in danger]
    if arr:
        interp.red_flags.append(
            "结/促类歇止脉：现代对应心律不齐（早搏/房颤等），建议心电图核实。")

    # 3) 证候假说（反查证型库）
    interp.candidate_syndromes = _syndrome_reverse_lookup(names)
    interp.verification_questions = _collect_questions(interp.candidate_syndromes)

    # 4) 症状验证后的辨证（脉证互参）
    if symptoms:
        from .reasoning import differentiate
        res = differentiate(symptoms=list(symptoms), pulse=pulse_text, topn=3)
        interp.verified_syndromes = [{
            "name": c.name, "confidence": c.confidence, "rationale": c.rationale,
            "treatment": c.treatment, "formulas": c.formulas,
        } for c in res]
        interp.mode = "verified"

    # 5) 典籍检索摘录
    if include_classical and names:
        kb = get_kb()
        for ch in kb.search(f"{''.join(names)}脉 主病 相兼", k=3):
            interp.classical.append({
                "title": ch.title, "section": ch.section, "file": ch.file,
                "excerpt": ch.excerpt(200),
            })
    return interp
