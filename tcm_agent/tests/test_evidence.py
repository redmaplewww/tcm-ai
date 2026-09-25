"""证据可信度引擎测试：分类、矛盾检测、质询、维度解决。"""

from tcm_agent.evidence import (
    EvidenceEngine,
    dimension_side,
    symptom_dimension,
)
from tcm_agent.models import EvidenceItem


def _item(text: str, kind: str = "symptom", turn: int = 1, **kw) -> EvidenceItem:
    return EvidenceItem(text=text, kind=kind, turn=turn, **kw)


# ---------------------------------------------------------------- 可信度分类

def test_classify_reliability():
    eng = EvidenceEngine()
    eng.add_items([_item("怕冷"), _item("我肾虚", kind="self_diagnosis"),
                   _item("高血压病史", kind="history"),
                   _item("脉弦", kind="pulse"),
                   _item("脉沉细", kind="pulse", source="doctor"),
                   _item("舌红苔黄", kind="tongue")])
    by_text = {e.text: e for e in eng.items}
    assert by_text["怕冷"].reliability == "high"
    assert by_text["我肾虚"].reliability == "medium"
    assert by_text["高血压病史"].reliability == "medium"
    assert by_text["脉弦"].reliability == "low"
    assert by_text["脉沉细"].reliability == "medium"

    # 辨证只采用可信症状；自诊/病史/自述脉象排除
    assert eng.credible_symptoms() == ["怕冷"]
    assert eng.credible_pulse() == "脉沉细"  # 仅医师转述
    assert eng.credible_tongue() == "舌红苔黄"  # 舌象自述参考参与


# ---------------------------------------------------------------- 矛盾检测

def test_contradiction_sweat():
    eng = EvidenceEngine()
    eng.add_items([_item("无汗", turn=1), _item("恶寒", turn=1)])
    ch = eng.add_items([_item("自汗", turn=2)])
    assert len(ch) == 1 and ch[0].kind == "contradiction" and ch[0].dimension == "汗"
    assert "无汗" in ch[0].question
    # 双方均被排除
    assert "无汗" not in eng.credible_symptoms()
    assert "自汗" not in eng.credible_symptoms()
    assert any("无汗" in n for n in eng.excluded_notes())


def test_contradiction_resolution_latest_wins():
    eng = EvidenceEngine()
    eng.add_items([_item("无汗", turn=1)])
    eng.add_items([_item("自汗", turn=2)])
    # 用户澄清：夜里睡着出汗（盗汗），最近为准
    ch = eng.add_items([_item("盗汗", turn=3)])
    assert not any(c.kind == "contradiction" for c in ch), f"澄清后又触发质询: {ch}"
    assert eng.credible_symptoms() == ["盗汗"]
    assert eng.resolution_notes()


def test_contradiction_stool_and_tongue():
    eng = EvidenceEngine()
    eng.add_items([_item("便秘", turn=1)])
    ch = eng.add_items([_item("泄泻", turn=2)])
    assert any(c.dimension == "二便" for c in ch)
    eng2 = EvidenceEngine()
    eng2.add_items([_item("舌红", kind="tongue", turn=1)])
    ch2 = eng2.add_items([_item("舌淡", kind="tongue", turn=2)])
    assert any(c.dimension == "舌象" and c.kind == "contradiction" for c in ch2)
    assert eng2.credible_tongue() == ""


def test_tongue_normal_neutral():
    """淡红舌为正常，不参与红/淡对立。"""
    assert dimension_side("舌淡红苔薄白", "舌象") is None
    eng = EvidenceEngine()
    eng.add_items([_item("舌红", kind="tongue", turn=1)])
    ch = eng.add_items([_item("舌淡红苔薄白", kind="tongue", turn=2)])
    assert not any(c.kind == "contradiction" for c in ch)
    assert eng.credible_tongue() == "舌淡红苔薄白"  # 最新为准


def test_no_false_contradiction():
    eng = EvidenceEngine()
    eng.add_items([_item("无汗", turn=1), _item("不出汗", turn=2)])  # 同侧
    assert eng.credible_symptoms() == ["无汗", "不出汗"]


# ---------------------------------------------------------------- 数值核验

def test_temperature_conflict():
    eng = EvidenceEngine()
    eng.add_items([_item("低热", turn=1)])
    ch = eng.check_temperature("我低烧39度，量了好几次", turn=1)
    assert len(ch) == 1 and ch[0].kind == "numeric_conflict"
    # 低热被质疑排除，客观发热计入
    assert "低热" not in eng.credible_symptoms()
    assert any("39" in s for s in eng.credible_symptoms())


def test_temperature_ok_cases():
    eng = EvidenceEngine()
    eng.add_items([_item("低热", turn=1)])
    assert eng.check_temperature("有点低烧，没量体温", turn=1) == []
    eng2 = EvidenceEngine()
    eng2.add_items([_item("低热", turn=1)])
    assert eng2.check_temperature("低烧37.8度", turn=1) == []  # 数值不矛盾
    assert eng2.credible_symptoms() == ["低热"]


# ---------------------------------------------------------------- 维度识别

def test_symptom_dimension():
    assert symptom_dimension("无汗")[0] == "汗"
    assert symptom_dimension("盗汗") == ("汗", "B")
    assert symptom_dimension("大便干结")[0] == "二便"
    assert symptom_dimension("头痛")[0] == ""  # 无对立维度


def test_colloquial_contradiction_detected():
    """口语表述经规范化后仍能检出矛盾（'没有汗' vs '出汗挺多'）。"""
    assert symptom_dimension("没有汗") == ("汗", "A")
    assert symptom_dimension("出汗挺多") == ("汗", "B")
    eng = EvidenceEngine()
    eng.add_items([_item("没有汗", turn=1)])
    ch = eng.add_items([_item("出汗挺多", turn=2)])
    assert any(c.kind == "contradiction" and c.dimension == "汗" for c in ch)
    assert "没有汗" not in eng.credible_symptoms()
    # 澄清后按最新陈述解决
    ch2 = eng.add_items([_item("感冒三天都不出汗", turn=3)])
    assert not any(c.kind == "contradiction" for c in ch2)
    assert "出汗挺多" not in eng.credible_symptoms()
