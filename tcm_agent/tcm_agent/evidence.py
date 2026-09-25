"""证据可信度引擎：分类、矛盾检测、质询生成、按最新陈述解决争议。

原则（"不默认用户输入完全正确"）：
- 症状自述（"我怕冷""流清鼻涕"）→ high，直接参与辨证；
- 自诊术语（"我肾虚""上火"）、病史自述、舌象自述 → medium，不直接参与推理
 （自诊/病史）或仅作参考（舌象），并触发质询核实；
- 普通用户脉象自述 → low（线上不可切脉），医师转述 → medium；
- 同维度对立信息（无汗vs自汗、便秘vs泄泻、舌红vs舌淡）→ 双方转 disputed
 并生成质询；用户澄清后，该维度按最新陈述整体替换；
- "低烧"+实测≥38.5℃ → 数值冲突，质询并按实测修正为客观"发热"。
"""

from __future__ import annotations

import re
from typing import Optional

from . import reasoning
from .models import EvidenceChallenge, EvidenceItem

_TEMP_RE = re.compile(r"(3[89]|4[01])(?:\.\d)?\s*[度℃]")
_LOW_FEVER_TOKENS = ("低热", "低烧")

#: 同维度对立组：A与B侧同时存在（且未解决）视为矛盾
DIMENSION_GROUPS: dict[str, dict[str, list[str]]] = {
    "汗": {"A": ["无汗", "不出汗"], "B": ["自汗", "盗汗", "汗多", "出汗多", "大汗", "动则汗出"]},
    "二便": {"A": ["便秘", "大便干", "大便硬"], "B": ["泄泻", "腹泻", "便溏", "拉肚子"]},
    "睡眠": {"A": ["失眠", "睡不着", "入睡困难", "易醒"], "B": ["嗜睡", "睡不够", "昏昏欲睡"]},
    "食欲": {"A": ["食欲不振", "食少", "吃不下"], "B": ["消谷善饥", "容易饿", "吃得多"]},
    "舌象": {"A": ["舌红", "舌绛", "边尖红"], "B": ["舌淡"]},
}

_CONTRADICTION_QUESTIONS = {
    "汗": "关于出汗，您前后描述不一致（{items}）。请以最近的情况为准说明：白天稍活动会出汗吗？夜里睡着后会不会出汗？",
    "二便": "关于大便，您前后描述不一致（{items}）。请说明最近一周的实际情况：偏干还是偏稀？每天几次？",
    "睡眠": "关于睡眠，您前后描述不一致（{items}）。请确认目前最主要的表现：入睡难、易醒，还是困倦嗜睡？",
    "食欲": "关于胃口，您前后描述不一致（{items}）。请确认：是吃不下饭，还是容易饿吃得多？",
    "舌象": "舌色前后描述不一致（{items}）。请对着镜子确认：整个舌体偏淡红（正常）、偏红，还是偏淡白？",
}


def dimension_side(text: str, dim: str) -> Optional[str]:
    """判断词条属于维度哪一侧；无属返回None。"""
    groups = DIMENSION_GROUPS.get(dim)
    if not groups:
        return None
    if dim == "舌象":
        if "淡红" in text:
            return None  # 淡红舌为正常舌象，不参与红/淡对立
        if any(k in text for k in groups["A"]):
            return "A"
        if any(k in text for k in groups["B"]):
            return "B"
        return None
    for side in ("A", "B"):
        if any(k in text for k in groups[side]):
            return side
    return None


def symptom_dimension(text: str) -> tuple[str, Optional[str]]:
    """返回症状词条所属（维度, 侧）；无对立维度时维度为空。

    先按原词匹配；未命中时用同义词/口语映射展开后再匹配
    （"没有汗"→"无汗"、"出汗挺多"→"汗多"）。
    """
    for dim in DIMENSION_GROUPS:
        side = dimension_side(text, dim)
        if side is not None:
            return dim, side
    for expanded in reasoning.normalize_symptoms([text]):
        if expanded == text:
            continue
        for dim in DIMENSION_GROUPS:
            side = dimension_side(expanded, dim)
            if side is not None:
                return dim, side
    return "", None


def classify(item: EvidenceItem) -> EvidenceItem:
    """按类型与来源赋予可信度。"""
    if item.kind == "self_diagnosis":
        item.reliability = "medium"
        item.reason = "用户自诊术语，未经核实，不参与辨证"
    elif item.kind == "history":
        item.reliability = "medium"
        item.reason = "病史自述，供参考，不参与辨证"
    elif item.kind == "tongue":
        item.reliability = "medium"
        item.reason = "舌象为用户自述，仅供参考"
    elif item.kind == "pulse":
        if item.source == "doctor":
            item.reliability = "medium"
            item.reason = "医师面诊脉象转述"
        else:
            item.reliability = "low"
            item.reason = "普通用户脉象自述不可靠，未采用（线上无法切脉）"
    return item


class EvidenceEngine:
    """累积证据、管理可信度与维度争议。"""

    def __init__(self) -> None:
        self.items: list[EvidenceItem] = []
        #: 已解决争议的维度 -> 解决说明
        self.resolved: dict[str, str] = {}
        self._challenged_self_dx: set[str] = set()
        self._pulse_challenged = False

    # ------------------------------------------------------------ 录入

    def add_items(self, new_items: list[EvidenceItem]) -> list[EvidenceChallenge]:
        """录入一批新证据，返回需要用户核实的问题。"""
        challenges: list[EvidenceChallenge] = []
        for it in new_items:
            if not (it.text or "").strip():
                continue
            classify(it)
            if it.kind == "symptom":
                it.dimension, _ = symptom_dimension(it.text)
            elif it.kind == "tongue":
                it.dimension = "舌象"
            # 去重：同文本已存在则跳过
            if any(e.text == it.text and e.kind == it.kind for e in self.items):
                continue
            self.items.append(it)

        # 维度级处理：新证据到达的维度
        def _side(e: EvidenceItem) -> Optional[str]:
            return symptom_dimension(e.text)[1] if e.dimension else None

        touched_dims = {it.dimension for it in new_items if it.dimension}
        for dim in touched_dims:
            cur = [e for e in self.items if e.dimension == dim]
            if len(cur) <= 1:
                continue
            has_disputed = any(e.reliability == "disputed" for e in cur)
            if has_disputed:
                # 该维度已有争议：用户最新陈述即为澄清，最新侧别项替换全部旧项
                latest_turn = max(e.turn for e in cur)
                kept = [e for e in cur if e.turn == latest_turn and _side(e) is not None]
                if not kept:
                    kept = [e for e in cur if e.turn == latest_turn]
                if kept:
                    self._replace_dimension(dim, kept)
                    self.resolved[dim] = "已按您最新说明更新：" + "、".join(e.text for e in kept)
                continue
            sides = {_side(e) for e in cur}
            if "A" in sides and "B" in sides:
                for e in cur:
                    if _side(e) in ("A", "B"):
                        e.reliability = "disputed"
                        e.reason = "与同一维度其他描述矛盾，待您澄清"
                involved = [e.text for e in cur if _side(e)]
                q = _CONTRADICTION_QUESTIONS.get(
                    dim, "前后描述不一致（{items}），请以最新情况澄清。"
                ).format(items=" vs ".join(involved))
                challenges.append(EvidenceChallenge(
                    kind="contradiction", dimension=dim, question=q, involved=involved,
                ))
            elif dim == "舌象" and len(cur) > 1:
                # 舌象无矛盾但多次描述：以最新一次为准
                latest = max(cur, key=lambda e: e.turn)
                self._replace_dimension(dim, [latest])

        # 自诊质询（每个术语只问一次）
        for it in new_items:
            if it.kind == "self_diagnosis" and it.text not in self._challenged_self_dx:
                self._challenged_self_dx.add(it.text)
                challenges.append(EvidenceChallenge(
                    kind="self_diagnosis", dimension="自诊",
                    question=f"您提到“{it.text}”——这是医师的诊断，还是自己的判断？"
                             f"请具体说说有哪些不舒服（部位、感觉、持续时间），我们先按具体症状来分析。",
                    involved=[it.text],
                ))

        # 脉象质询（仅首次、仅自述）
        pulse_new = [it for it in new_items if it.kind == "pulse" and it.source != "doctor"]
        if pulse_new and not self._pulse_challenged:
            self._pulse_challenged = True
            challenges.append(EvidenceChallenge(
                kind="pulse_unverified", dimension="脉象",
                question="您提到的脉象是中医师当面诊脉的结果吗？如果不是，我们将不采用脉象信息进行辨证（线上无法切脉）。",
                involved=[it.text for it in pulse_new],
            ))
        return challenges

    def _replace_dimension(self, dim: str, kept: list[EvidenceItem]) -> None:
        """维度内只保留kept（最新陈述），其余移除。"""
        self.items = [e for e in self.items if e.dimension != dim] + kept

    # ------------------------------------------------------------ 数值核验

    def check_temperature(self, user_text: str, turn: int) -> list[EvidenceChallenge]:
        """低热表述与实测体温数值矛盾时质询，并按实测修正。"""
        text = user_text or ""
        if not any(t in text for t in _LOW_FEVER_TOKENS):
            return []
        m = _TEMP_RE.search(text)
        if not m:
            return []
        value = float(m.group(1))
        if value < 38.5:
            return []
        # 实测≥38.5与"低烧"矛盾：质询 + 按客观数值修正
        for e in self.items:
            if any(t in e.text for t in _LOW_FEVER_TOKENS) and e.reliability != "disputed":
                e.reliability = "disputed"
                e.reason = f"与实测体温{value}℃矛盾（低热应<38℃）"
        corrected = EvidenceItem(
            text=f"发热（实测{value}℃）", kind="temperature",
            reliability="high", reason="按实测体温客观修正", turn=turn,
        )
        self.items.append(corrected)
        return [EvidenceChallenge(
            kind="numeric_conflict", dimension="寒热",
            question=f"您说“低烧”，但提到实测{value}℃——{value}℃已属中高热而非低热。"
                     f"我们已按实测体温计入，请确认测量是否规范（腋下≥5分钟、安静状态）。",
            involved=["低热", f"{value}℃"],
        )]

    # ------------------------------------------------------------ 输出

    def credible_symptoms(self) -> list[str]:
        """参与辨证的症状：high/medium可信且非disputed（自诊/病史不参与）。"""
        return [
            e.text for e in self.items
            if e.kind in ("symptom", "temperature")
            and e.reliability in ("high", "medium")
        ]

    def credible_tongue(self) -> str:
        """最新一条非disputed舌象（用户自述，仅供参考）。"""
        tongues = [e for e in self.items if e.kind == "tongue" and e.reliability != "disputed"]
        return tongues[-1].text if tongues else ""

    def credible_pulse(self) -> str:
        """仅采用医师转述脉象。"""
        pulses = [e for e in self.items if e.kind == "pulse" and e.source == "doctor"
                  and e.reliability != "disputed"]
        return pulses[-1].text if pulses else ""

    def excluded_notes(self) -> list[str]:
        """未参与辨证的证据及原因（disputed与low）。"""
        notes = []
        for e in self.items:
            if e.reliability in ("disputed", "low"):
                notes.append(f"{e.text}（{e.reason}）")
            elif e.kind in ("self_diagnosis", "history"):
                notes.append(f"{e.text}（{e.reason}）")
        return notes

    def resolution_notes(self) -> list[str]:
        return list(self.resolved.values())
