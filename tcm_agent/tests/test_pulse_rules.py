# -*- coding: utf-8 -*-
"""脉象规则判别引擎与脉学典籍知识库的测试。"""

from __future__ import annotations

from tcm_agent import data
from tcm_agent.pulse_rules import (
    PulseFeatures,
    classify_and_describe,
    classify_pulse,
    to_pulse_text,
)


# ---------------------------------------------------------------- 数据层

def test_pulses_json_loads_28_plus_reference():
    pulses = data.pulses()
    assert len(pulses) >= 29, f"应含28脉+平脉参考，实得{len(pulses)}"
    names = {p["name"] for p in pulses}
    for n in ["浮", "沉", "迟", "数", "滑", "涩", "弦", "紧", "洪", "细",
              "微", "结", "代", "促", "疾", "虚", "实", "缓"]:
        assert n in names, f"缺少脉名：{n}"
    for p in pulses:
        assert "rules" in p and "ppg_assessable" in p, f"{p['name']}缺规则或模态字段"
        assert p.get("main_indications"), f"{p['name']}缺主病"


def test_get_pulse_lookup():
    assert data.get_pulse("数脉")["name"] == "数"
    assert data.get_pulse("滑")["name"] == "滑"
    assert data.get_pulse("不存在脉") is None


# ---------------------------------------------------------------- 规则引擎

def test_shuo_pulse_from_rate():
    """脉率102、节律齐 -> 数脉居首，且不出现结代促。"""
    feats = PulseFeatures(modality="ppg", rate_bpm=102, rhythm_cv_pct=2.0,
                          pause_present=False, n_beats=5)
    cands = classify_pulse(feats)
    assert cands, "应有候选"
    assert cands[0].name == "数"
    names = [c.name for c in cands]
    for bad in ("结", "代", "促"):
        assert bad not in names, f"无歇止不应出现{bad}脉"
    assert any("脉率102" in b for b in cands[0].basis)


def test_chi_pulse_from_rate():
    feats = PulseFeatures(modality="ppg", rate_bpm=52, rhythm_cv_pct=1.5, pause_present=False)
    cands = classify_pulse(feats)
    assert cands[0].name == "迟"


def test_jie_dai_cu_pause_logic():
    """结脉：歇止+止无定数+率偏缓；代脉：止有定数。"""
    jie = classify_pulse(PulseFeatures(
        modality="ppg", rate_bpm=55, rhythm_cv_pct=18,
        pause_present=True, pause_fixed=False))
    names = [c.name for c in jie]
    assert names[0] == "结"
    assert "代" not in names and "促" not in names[:1]

    dai = classify_pulse(PulseFeatures(
        modality="ppg", rate_bpm=68, rhythm_cv_pct=30,
        pause_present=True, pause_fixed=True))
    names = [c.name for c in dai]
    assert names[0] == "代"
    assert "结" not in names and "促" not in names


def test_hua_pulse_from_morphology():
    """滑脉形态：升支陡+重搏前波低+降中峡深+重搏波显。"""
    feats = PulseFeatures(modality="ppg", rate_bpm=78, rhythm_cv_pct=2.5,
                          pause_present=False, rise_time_ms=60,
                          h3_h1=0.15, h4_h1=0.10, h5_h1=0.55)
    cands = classify_pulse(feats)
    top = [c.name for c in cands[:4]]
    assert "滑" in top
    assert "涩" not in top and "弦" not in top[:2]


def test_xian_pulse_from_morphology():
    """弦脉形态：h3融合+h4高+h5平。"""
    feats = PulseFeatures(modality="ppg", rate_bpm=74, rhythm_cv_pct=3.0,
                          pause_present=False, rise_time_ms=130,
                          h3_h1=0.85, h4_h1=0.55, h5_h1=0.08)
    cands = classify_pulse(feats)
    top = [c.name for c in cands[:4]]
    assert "弦" in top
    assert "滑" not in top[:2]


def test_ppg_cannot_assess_depth():
    """PPG模态下浮沉类脉不得出现。"""
    feats = PulseFeatures(modality="ppg", rate_bpm=102, rhythm_cv_pct=2.0,
                          pause_present=False, depth_class="浮")  # 即使误传脉位
    names = [c.name for c in classify_pulse(feats)]
    for depth_pulse in ("浮", "沉", "伏", "牢", "濡", "弱"):
        assert depth_pulse not in names


def test_pressure_modality_unlocks_depth():
    """压力模态可评浮脉。"""
    feats = PulseFeatures(modality="pressure", rate_bpm=80, rhythm_cv_pct=2.0,
                          pause_present=False, depth_class="浮")
    names = [c.name for c in classify_pulse(feats)]
    assert "浮" in names


def test_no_pause_no_jie_dai():
    """无歇止信息时不输出结代促（硬规则：缺证据不判）。"""
    feats = PulseFeatures(modality="ppg", rate_bpm=60, rhythm_cv_pct=10)
    names = [c.name for c in classify_pulse(feats)]
    for bad in ("结", "代", "促"):
        assert bad not in names


def test_to_pulse_text_combination():
    feats = PulseFeatures(modality="ppg", rate_bpm=102, rhythm_cv_pct=2.0,
                          pause_present=False, rise_time_ms=55,
                          h3_h1=0.1, h4_h1=0.0, h5_h1=0.5)
    cands = classify_pulse(feats)
    text = to_pulse_text(cands)
    assert text.startswith("脉数"), f"应以数脉开头：{text}"
    assert "滑" in text, f"应含滑象：{text}"


def test_to_pulse_text_normal():
    feats = PulseFeatures(modality="ppg", rate_bpm=72, rhythm_cv_pct=2.0, pause_present=False)
    text = to_pulse_text(classify_pulse(feats))
    assert "脉数" not in text and "脉迟" not in text and "疾" not in text
    assert "平" in text  # 至数节律正常应输出平脉


def test_classify_and_describe_structure():
    out = classify_and_describe(PulseFeatures(
        modality="ppg", rate_bpm=98, rhythm_cv_pct=2.0,
        pause_present=False, source="MAX30102示例"))
    assert out["pulse_text"]
    assert isinstance(out["candidates"], list) and out["candidates"]
    first = out["candidates"][0]
    assert {"name", "confidence", "basis", "classical", "main_indications"} <= set(first)


# ---------------------------------------------------------------- 知识库接入

def test_kb_includes_pulse_classics():
    from tcm_agent.kb import get_kb
    kb = get_kb()
    stats = kb.stats()
    assert "08_脉学典籍" in stats, f"知识库未收录脉学典籍目录：{stats}"


def test_kb_search_hits_classics():
    from tcm_agent.kb import get_kb
    kb = get_kb()
    hits = [ch.file for ch in kb.search("滑脉如珠 流利", k=3)]
    assert any("濒湖脉学" in f for f in hits), f"检索未命中濒湖脉学：{hits}"
    hits2 = [ch.file for ch in kb.search("h3 h4 h5 降中峡 量化", k=3)]
    assert any("脉图学" in f for f in hits2), f"检索未命中脉图学：{hits2}"


def test_reasoning_accepts_pulse_text():
    """规则层输出可直接进入辨证打分管线。"""
    from tcm_agent.reasoning import differentiate
    feats = PulseFeatures(modality="ppg", rate_bpm=102, rhythm_cv_pct=2.0,
                          pause_present=False, rise_time_ms=55,
                          h3_h1=0.1, h4_h1=0.0, h5_h1=0.5)
    text = to_pulse_text(classify_pulse(feats))
    assert text
    res = differentiate(symptoms=["发热", "口渴", "痰黄稠"], pulse=text, topn=3)
    assert res, f"脉{text} + 症状应产生证型候选"
