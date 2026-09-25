"""中药配伍安全校验测试：十八反/十九畏、妊娠禁忌、毒性标注。"""

from tcm_agent import reasoning, tools


def test_shiba_fan_gancao_group():
    """甘草反甘遂/大戟/海藻/芫花。"""
    for bad in (["甘草", "甘遂"], ["炙甘草", "海藻"], ["甘草", "大戟", "芫花"]):
        r = reasoning.check_herb_combination(bad)
        assert not r.is_safe and any("十八反" in c for c in r.conflicts), f"{bad} 未检出: {r.conflicts}"


def test_shiba_fan_wutou_group():
    """乌头（附子）反半夏/贝母/瓜蒌/白蔹/白及。"""
    for bad in (["附子", "半夏"], ["川乌", "川贝母"], ["制附子", "瓜蒌"], ["草乌", "白及"]):
        r = reasoning.check_herb_combination(bad)
        assert not r.is_safe and any("十八反" in c for c in r.conflicts), f"{bad} 未检出: {r.conflicts}"


def test_shiba_fan_lilu_group():
    """藜芦反诸参辛芍。"""
    for bad in (["藜芦", "人参"], ["藜芦", "丹参"], ["藜芦", "细辛"], ["藜芦", "西洋参"]):
        r = reasoning.check_herb_combination(bad)
        assert not r.is_safe and any("十八反" in c for c in r.conflicts), f"{bad} 未检出: {r.conflicts}"


def test_shijiu_wei_cases():
    """十九畏：丁香畏郁金、人参畏五灵脂、官桂畏赤石脂、巴豆畏牵牛。"""
    for bad in (["丁香", "郁金"], ["人参", "五灵脂"], ["肉桂", "赤石脂"], ["巴豆", "牵牛子"]):
        r = reasoning.check_herb_combination(bad)
        assert not r.is_safe and any("十九畏" in c for c in r.conflicts), f"{bad} 未检出: {r.conflicts}"


def test_safe_combination():
    r = reasoning.check_herb_combination(["黄芪", "当归", "白术", "甘草", "陈皮"])
    assert r.is_safe and not r.conflicts, f"误报: {r.conflicts}"
    assert "未检出" in r.advice


def test_same_group_no_conflict():
    """同组药物不冲突（如人参与党参同用）。"""
    r = reasoning.check_herb_combination(["人参", "党参", "沙参"])
    assert r.is_safe and not r.conflicts


def test_pregnancy_and_toxicity():
    # 附子妊娠禁用 + 有毒
    r = reasoning.check_herb_combination(["附子"], pregnant=True)
    assert any("妊娠禁用" in w for w in r.pregnancy_warnings), r.pregnancy_warnings
    assert any("有毒" in t for t in r.toxicity_notes), r.toxicity_notes
    # 非孕期：仍有毒性标注，无妊娠警示
    r2 = reasoning.check_herb_combination(["附子"], pregnant=False)
    assert not r2.pregnancy_warnings and r2.toxicity_notes
    # 无毒常用药：无毒性标注
    r3 = reasoning.check_herb_combination(["黄芪"], pregnant=True)
    assert not r3.toxicity_notes and not r3.pregnancy_warnings


def test_tool_check_herb_safety():
    out = tools.check_herb_safety(["甘草", "甘遂"])
    assert "十八反" in out and "禁止同用" in out
    out2 = tools.check_herb_safety(["当归", "黄芪"])
    assert "未检出" in out2
    out3 = tools.check_herb_safety(["附子"], pregnant=True)
    assert "妊娠禁用" in out3
