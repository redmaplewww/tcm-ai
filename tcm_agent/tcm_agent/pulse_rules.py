"""脉象规则判别引擎：结构化信号特征 -> 脉名候选。

经验驱动的"翻译层"：把脉图学/诊断学文献中的量化判别标准
（tcm_kb/data/pulses.json）编码为规则，将传感器信号特征映射为中医
脉名候选，供 reasoning.differentiate() 与 LLM 辨证层消费。

设计要点：
- 零训练数据：纯规则+软隶属度打分，阈值为文献近似值，可经自采数据校准；
- 模态门控：PPG只能判至数/节律/部分形态，脉位（浮沉）脉力（虚实）等
  维度按 assessability 降权或直接不参评，绝不硬猜；
- 歇止（结/代/促）为硬规则：无歇止证据时不输出这三类脉；
- 输出附带经典出处与依据，可直接进入知识库检索与辨证管线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import data

# 维度权重：至数与歇止最可靠，形态次之，类别维度（脉位/脉力）再次
_DIM_WEIGHTS = {
    "rate_bpm": 1.0,
    "pauses": 1.2,
    "rhythm_cv_max": 0.6,
    "h3_h1": 0.8,
    "h4_h1": 0.8,
    "h5_h1": 0.8,
    "rise_time_ms": 0.8,
    "amplitude_class": 0.7,
    "width_class": 0.8,
    "depth_class": 0.9,
    "strength_class": 0.9,
}

_RATE_RHYTHM_NAMES = {"平", "迟", "缓", "数", "疾", "结", "代", "促", "散"}


@dataclass
class PulseFeatures:
    """传感器信号特征（上游参数提取层的输出）。

    未提供的维度留 None，引擎只对已提供且该模态可评的维度打分。
    """

    modality: str = "ppg"                    # 'ppg' | 'pressure'
    rate_bpm: Optional[float] = None         # 平均脉率（次/分）
    rhythm_cv_pct: Optional[float] = None    # RR间期变异系数（%）
    pause_present: Optional[bool] = None     # 是否检出歇止
    pause_fixed: Optional[bool] = None       # True=止有定数（代），False=止无定数
    h3_h1: Optional[float] = None            # 重搏前波/主波
    h4_h1: Optional[float] = None            # 降中峡/主波
    h5_h1: Optional[float] = None            # 重搏波/主波
    rise_time_ms: Optional[float] = None     # 主波上升时间10-90%
    amplitude_class: Optional[str] = None    # '极低'|'低'|'中'|'高'
    width_class: Optional[str] = None        # '细'|'中'|'宽'（脉宽）
    depth_class: Optional[str] = None        # '浮'|'中'|'沉'|'极沉'（压力传感器）
    strength_class: Optional[str] = None     # '有力'|'无力'（压力传感器）
    n_beats: Optional[int] = None            # 参与统计的搏动数
    source: str = ""                         # 数据来源说明


@dataclass
class PulseCandidate:
    """脉名候选：含置信度、依据与经典出处。"""

    name: str
    category: str = ""
    confidence: float = 0.0
    basis: list[str] = field(default_factory=list)
    assessability: float = 1.0
    classical: str = ""
    main_indications: str = ""
    combined_examples: list[str] = field(default_factory=list)
    note: str = ""
    is_reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "category": self.category,
            "confidence": round(self.confidence, 3), "basis": self.basis,
            "assessability": self.assessability, "classical": self.classical,
            "main_indications": self.main_indications,
            "combined_examples": self.combined_examples, "note": self.note,
            "is_reference": self.is_reference,
        }


# ---------------------------------------------------------------- 隶属度

def _band_membership(value: float, lo: float, hi: float, cap: float = 1e9) -> float:
    """区间软隶属度：区间内为1，区外线性衰减到0（边缘容差=区间半宽，不超过cap）。"""
    if lo <= value <= hi:
        return 1.0
    margin = max(min((hi - lo) * 0.5, cap), 1e-6)
    if value < lo:
        return max(0.0, 1.0 - (lo - value) / margin)
    return max(0.0, 1.0 - (value - hi) / margin)


#: 各维度的容差上限：防宽区间脉的衰减带过宽（如脉率103不应存活于迟脉边缘）
_MARGIN_CAPS = {"rate_bpm": 15.0, "rise_time_ms": 40.0,
                "h3_h1": 0.15, "h4_h1": 0.15, "h5_h1": 0.15}


def _upper_membership(value: float, hi: float) -> float:
    """上界软隶属度：value<=hi 为1，超出后线性衰减（容差=hi*0.8）。"""
    if value <= hi:
        return 1.0
    return max(0.0, 1.0 - (value - hi) / max(hi * 0.8, 1e-6))


def _class_membership(given: Optional[str], wanted: list[str]) -> Optional[float]:
    """类别维度隶属：given在wanted中为1，未提供为None（不参评）。"""
    if given is None:
        return None
    if not wanted:
        return None
    return 1.0 if any(w in given or given in w for w in wanted) else 0.0


# ---------------------------------------------------------------- 主引擎

def classify_pulse(features: PulseFeatures, topn: int = 8) -> list[PulseCandidate]:
    """规则判别：返回按置信度降序的脉名候选。"""
    assess_key = "ppg_assessable" if features.modality == "ppg" else "pressure_assessable"
    results: list[PulseCandidate] = []

    for p in data.pulses():
        rules = p.get("rules", {}) or {}
        assess = float(p.get(assess_key, 0.0))
        if assess <= 0.01:
            continue  # 该模态完全不可评（如PPG之于浮沉）
        # 空间/部位/中空类定义性信息（关部、长短、芤革中空）单点信号无法提供 -> 不参评
        if any(k in rules for k in ("position", "spatial", "hollow")):
            continue
        # 类别维度（脉位/脉力/脉幅/脉宽）均为定性指感锚点：规则要求而信号未提供
        # 即不参评——单通道信号的判别空间收缩为至数节律+可量化形态
        if any(rules.get(k) and f is None for k, f in [
            ("depth_class", features.depth_class),
            ("strength_class", features.strength_class),
            ("amplitude_class", features.amplitude_class),
            ("width_class", features.width_class),
        ]):
            continue

        dim_scores: list[tuple[float, float, str]] = []  # (weight, membership, evidence)
        veto = False

        # 歇止：硬规则——结/代/促以歇止为定义，无歇止证据即不参评
        want_pause = rules.get("pauses")
        if want_pause == "irregular":
            if features.pause_present is not True or features.pause_fixed is True:
                continue  # 无歇止、歇止情况未知或止有定数 -> 不判结/促
            dim_scores.append((1.2, 1.0, "检出歇止且止无定数"))
        elif want_pause == "fixed":
            if features.pause_present is not True or features.pause_fixed is not True:
                continue  # 必须有规律歇止（止有定数）
            dim_scores.append((1.2, 1.0, "检出歇止且止有定数（规律间歇）"))
        else:
            if features.pause_present is True:
                continue  # 有歇止则排除无歇止脉（平/迟/数/弦等）

        # 至数（定义性维度：被明确反驳即否决）
        band = rules.get("rate_bpm")
        if band and features.rate_bpm is not None:
            m = _band_membership(features.rate_bpm, float(band[0]), float(band[1]), _MARGIN_CAPS["rate_bpm"])
            if m < 0.15:
                continue
            ev = f"脉率{features.rate_bpm:.0f}次/分（{p['name']}脉带{band[0]}~{band[1]}）"
            dim_scores.append((1.0, m, ev))
        elif band and features.rate_bpm is None and want_pause is None:
            continue  # 该脉以至数为核心但未提供脉率 -> 不可评

        # 节律规整度
        cv_max = rules.get("rhythm_cv_max")
        if cv_max is not None and features.rhythm_cv_pct is not None:
            m = _upper_membership(features.rhythm_cv_pct, float(cv_max))
            ev = f"RR变异{features.rhythm_cv_pct:.1f}%（要求≤{cv_max}%）"
            dim_scores.append((0.6, m, ev))

        # 形态参数：上升时间为定义性维度（否决）；h比值为支持性维度（降权）
        h_penalty = 1.0
        for key, feat, label, defining in [
            ("h3_h1", features.h3_h1, "重搏前波h3/h1", False),
            ("h4_h1", features.h4_h1, "降中峡h4/h1", False),
            ("h5_h1", features.h5_h1, "重搏波h5/h1", False),
            ("rise_time_ms", features.rise_time_ms, "上升时间", True),
        ]:
            band = rules.get(key)
            if not (band and feat is not None):
                continue
            m = _band_membership(float(feat), float(band[0]), float(band[1]), _MARGIN_CAPS[key])
            if m < 0.15:
                if defining:
                    veto = True
                    break
                h_penalty = min(h_penalty, 0.6)
                dim_scores.append((_DIM_WEIGHTS[key], 0.0, f"{label}={feat:.2f}与{p['name']}参考{band[0]}~{band[1]}不符"))
                continue
            unit = "ms" if key == "rise_time_ms" else ""
            ev = f"{label}={feat:.2f}{unit}（{p['name']}参考{band[0]}~{band[1]}{unit}）"
            dim_scores.append((_DIM_WEIGHTS[key], m, ev))
        if veto:
            continue

        for key, feat, label in [
            ("amplitude_class", features.amplitude_class, "脉幅"),
            ("width_class", features.width_class, "脉宽"),
            ("depth_class", features.depth_class, "脉位"),
            ("strength_class", features.strength_class, "脉力"),
        ]:
            wanted = rules.get(key)
            if wanted:
                m = _class_membership(feat, [str(w) for w in wanted])
                if m is not None:
                    ev = f"{label}={feat}（{p['name']}要求{'/'.join(map(str, wanted))}）"
                    dim_scores.append((_DIM_WEIGHTS[key], m, ev))

        if not dim_scores or veto:
            continue

        total_w = sum(w for w, _, _ in dim_scores)
        raw = sum(w * m for w, m, _ in dim_scores) / total_w if total_w else 0.0
        # 完备度：单维证据不足以支持高置信
        completeness = 0.55 + 0.45 * min(1.0, len(dim_scores) / 3.0)
        confidence = raw * assess * completeness * h_penalty
        if confidence <= 0.12:
            continue

        results.append(PulseCandidate(
            name=str(p.get("name", "")),
            category=str(p.get("category", "")),
            confidence=confidence,
            basis=[ev for _, _, ev in dim_scores if True],
            assessability=assess,
            classical=str(p.get("classical", "")),
            main_indications=str(p.get("main_indications", "")),
            combined_examples=[str(x) for x in p.get("combined_examples", [])],
            note=str(p.get("note", "")),
            is_reference=bool(p.get("is_reference", False)),
        ))

    results.sort(key=lambda c: c.confidence, reverse=True)
    return results[:topn]


# ---------------------------------------------------------------- 输出整合

def to_pulse_text(candidates: list[PulseCandidate], min_conf: float = 0.3) -> str:
    """把候选合并为供 reasoning.differentiate(pulse=...) 消费的文本，如"脉数滑"。

    结构：至数/节律类脉取1个（数/迟/缓/疾/结/代/促/平），形态类脉取至多2个。
    """
    rate_pick, shape_names = "", []
    for c in candidates:
        if c.confidence < min_conf:
            continue
        if c.name in _RATE_RHYTHM_NAMES:
            if not rate_pick:
                rate_pick = c.name
        elif len(shape_names) < 2 and not c.is_reference:
            shape_names.append(c.name)
    # 至数正常（平）不进入脉名串；无任何异常脉时才单独输出"脉平"
    parts = ([rate_pick] if rate_pick and rate_pick != "平" else []) + shape_names
    if not parts:
        if rate_pick == "平":
            return "脉平（至数节律正常）"
        return ""
    return "脉" + "".join(parts)


def classify_and_describe(features: PulseFeatures, topn: int = 5) -> dict[str, Any]:
    """一步到位：判别+文本化，返回可JSON化的结构（供server/LLM消费）。"""
    cands = classify_pulse(features, topn=topn)
    return {
        "modality": features.modality,
        "source": features.source,
        "pulse_text": to_pulse_text(cands),
        "candidates": [c.to_dict() for c in cands],
    }
