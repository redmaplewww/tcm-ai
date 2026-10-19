# -*- coding: utf-8 -*-
"""脉象→病理方向推理模块测试。"""

from __future__ import annotations

from tcm_agent.pulse_diagnosis import (
    DANGER_PULSES,
    interpret_pulse,
    parse_pulse_names,
)


def test_parse_pulse_names():
    assert parse_pulse_names("脉数滑") == ["数", "滑"]
    assert parse_pulse_names("脉结") == ["结"]
    assert parse_pulse_names("脉缓弦滑") == ["缓", "弦", "滑"]
    assert parse_pulse_names("") == []
    assert parse_pulse_names("脉平（至数节律正常）") == ["平"]  # 括号注释不产生额外脉名


def test_hypothesis_mode_no_syndrome_conclusion():
    """仅脉象：给出假说与验证问题，不做证型结论（verified为空）。"""
    it = interpret_pulse("脉滑数")
    assert it.components == ["滑", "数"]
    assert it.mode == "hypothesis"
    assert not it.verified_syndromes
    # 痰热壅肺等"脉滑数"证型应进入假说
    names = [s["name"] for s in it.candidate_syndromes]
    assert any("痰热" in n or "湿热" in n or "胃火" in n for n in names), names
    assert it.verification_questions, "应生成验证问题"


def test_verified_mode_with_symptoms():
    it = interpret_pulse("脉滑数", symptoms=["咳嗽", "痰黄稠", "发热", "口渴"])
    assert it.mode == "verified"
    assert it.verified_syndromes
    assert it.verified_syndromes[0]["confidence"] > 0


def test_danger_pulse_flags():
    it = interpret_pulse("脉代")
    assert any("危候" in f for f in it.red_flags)
    it2 = interpret_pulse("脉疾")
    assert any("就医" in f for f in it2.red_flags)


def test_arrhythmia_pulse_flags():
    it = interpret_pulse("脉促")
    assert any("心律不齐" in f or "心电图" in f for f in it.red_flags)
    it2 = interpret_pulse("脉结")
    assert any("心电图" in f for f in it2.red_flags)


def test_classical_retrieval():
    it = interpret_pulse("脉弦", include_classical=True)
    assert it.classical, "应检索到典籍摘录"
    blob = "".join(c["excerpt"] for c in it.classical)
    assert "弦" in blob


def test_to_dict_structure():
    d = interpret_pulse("脉数滑").to_dict()
    assert {"pulse", "components", "candidate_syndromes",
            "verification_questions", "red_flags", "disclaimer"} <= set(d)
