"""确定性推理引擎测试：症状规范化、红旗检测、辨证打分、追问生成。

这是"病情推理无误"的核心验证：用教材级经典证型的标准症状组合，
断言引擎给出正确的证型、治法与方剂方向。
"""

from tcm_agent.models import PatientCase
from tcm_agent import reasoning


# ---------------------------------------------------------------- 规范化

def test_normalize_colloquial():
    out = reasoning.normalize_symptoms(["上火", "睡不着"])
    joined = "、".join(out)
    assert "口舌生疮" in joined or "咽喉肿痛" in joined, f"口语'上火'未展开: {out}"
    assert "失眠" in joined or "入睡困难" in joined, f"口语'睡不着'未映射: {out}"


def test_normalize_synonyms():
    out = reasoning.normalize_symptoms(["头疼", "发烧", "怕冷"])
    assert "头痛" in out and "发热" in out and ("恶寒" in out or "畏寒" in out), f"同义词映射失败: {out}"


def test_normalize_keeps_unknown_and_dedupes():
    out = reasoning.normalize_symptoms(["头痛", "头疼", "未知症状xyz"])
    assert out.count("头痛") == 1, f"未去重: {out}"
    assert "未知症状xyz" in out


# ---------------------------------------------------------------- 红旗

def test_red_flag_emergency_chest_pain():
    flags = reasoning.check_red_flags(["胸口像压了石头", "胸痛放射到左臂", "出冷汗"])
    assert any(f.level == "emergency" for f in flags), f"心梗红旗未触发: {flags}"


def test_red_flag_emergency_colloquial_variants():
    """口语变体也应触发（安全宁误报不漏报）。"""
    for text in (
        "胸口剧痛放射到左臂",
        "我爸突然胸口剧痛，疼到左胳膊，满头冷汗",
        "心口剧痛，疼到肩膀",
    ):
        flags = reasoning.check_red_flags([text])
        assert any(f.level == "emergency" for f in flags), f"口语胸痛未触发: {text} -> {flags}"


def test_red_flag_stroke():
    flags = reasoning.check_red_flags(["口角歪斜", "一侧肢体无力", "说话不清楚"])
    assert any(f.level in ("emergency", "urgent") for f in flags), f"中风红旗未触发: {flags}"


def test_red_flag_negative():
    flags = reasoning.check_red_flags(["有点头痛", "睡不好", "胃口一般"])
    assert not any(f.level == "emergency" for f in flags), f"普通症状误报急症: {flags}"


# ---------------------------------------------------------------- 辨证（核心）

def _top(case: PatientCase):
    result = reasoning.consult(case)
    return result


def test_differentiate_wind_cold():
    """风寒表证：恶寒发热无汗+清涕+浮紧脉 -> 风寒类证（八纲表寒/脏腑风寒/太阳伤寒等价），辛温解表方向。"""
    r = _top(PatientCase(
        symptoms=["恶寒", "发热", "无汗", "头痛", "身痛", "鼻塞", "流清涕"],
        tongue="舌淡红苔薄白", pulse="脉浮紧",
    ))
    assert r.primary_syndrome, f"未能辨证: {r.symptoms_canonical}"
    family = ("风寒", "表寒", "太阳伤寒")
    assert any(k in r.primary_syndrome for k in family), \
        f"应为风寒类证型，实为: {r.primary_syndrome}"
    assert "辛温" in r.treatment_principle or "解表" in r.treatment_principle or "散寒" in r.treatment_principle, \
        f"治法应为辛温解表类: {r.treatment_principle}"
    assert not r.is_emergency


def test_differentiate_wind_heat():
    """风热表证：发热重+咽痛+黄涕+口渴 -> 风热类证（表热/卫分等价），辛凉解表方向。"""
    r = _top(PatientCase(
        symptoms=["发热", "微恶风寒", "咽痛", "流黄涕", "口渴", "咳嗽"],
        tongue="舌边尖红苔薄黄", pulse="脉浮数",
    ))
    family = ("风热", "表热", "卫分")
    assert any(k in r.primary_syndrome for k in family), \
        f"应为风热类证型，实为: {r.primary_syndrome}"
    assert "辛凉" in r.treatment_principle or "解表" in r.treatment_principle \
        or "透邪" in r.treatment_principle or "清热" in r.treatment_principle


def test_differentiate_heart_spleen_deficiency():
    """心脾两虚：心悸失眠+食少疲乏+舌淡。"""
    r = _top(PatientCase(
        symptoms=["心悸", "失眠", "多梦", "健忘", "食少", "疲乏", "面色少华"],
        tongue="舌淡", pulse="脉细弱",
    ))
    assert "心脾两虚" in r.primary_syndrome, f"应为心脾两虚，实为: {r.primary_syndrome}"
    assert any("归脾" in f for f in r.formula_suggestions), \
        f"代表方应含归脾汤类: {r.formula_suggestions}"


def test_differentiate_liver_qi_stagnation():
    """肝气郁结：胁胀+抑郁+善太息。"""
    r = _top(PatientCase(
        symptoms=["胁肋胀痛", "情绪抑郁", "善太息", "脘腹胀满", "食欲不振"],
    ))
    assert "肝气郁" in r.primary_syndrome, f"应为肝气郁结类，实为: {r.primary_syndrome}"
    assert any(
        name in "、".join(r.formula_suggestions) or name in str(r.formula_suggestions)
        for name in ("柴胡疏肝散", "逍遥散")
    ), f"代表方应含柴胡疏肝散/逍遥散: {r.formula_suggestions}"


def test_differentiate_kidney_yin_deficiency():
    """肾阴虚：腰膝酸软+五心烦热+盗汗+舌红少苔。"""
    r = _top(PatientCase(
        symptoms=["腰膝酸软", "五心烦热", "盗汗", "耳鸣", "失眠", "口干"],
        tongue="舌红少苔", pulse="脉细数",
    ))
    assert "肾阴虚" in r.primary_syndrome, f"应为肾阴虚，实为: {r.primary_syndrome}"
    assert any("六味地黄" in f or "知柏地黄" in f for f in r.formula_suggestions), \
        f"代表方应含六味地黄丸类: {r.formula_suggestions}"


def test_differentiate_spleen_qi_deficiency():
    """脾气虚：食少腹胀+便溏+疲乏。"""
    r = _top(PatientCase(
        symptoms=["食欲不振", "腹胀", "便溏", "疲乏", "气短", "面色萎黄"],
        tongue="舌淡苔白", pulse="脉缓弱",
    ))
    assert ("脾气虚" in r.primary_syndrome) or ("脾胃虚弱" in r.primary_syndrome), \
        f"应为脾气虚类，实为: {r.primary_syndrome}"


def test_tongue_pulse_improves_confidence():
    """舌脉信息应提高辨证置信度。"""
    base = reasoning.differentiate(
        ["心悸", "失眠", "多梦", "健忘", "食少", "疲乏"]
    )
    with_tp = reasoning.differentiate(
        ["心悸", "失眠", "多梦", "健忘", "食少", "疲乏"],
        tongue="舌淡", pulse="脉细弱",
    )
    assert with_tp[0].confidence >= base[0].confidence


def test_insufficient_info_gives_followups():
    """信息不足时应给出追问而不是硬辨证。"""
    r = _top(PatientCase(symptoms=["头晕"]))
    assert len(r.followup_questions) >= 3, f"追问不足: {r.followup_questions}"


# ---------------------------------------------------------------- 全流程

def test_consult_emergency_short_circuit():
    """急症红旗：只给就医指引，不给方药。"""
    r = _top(PatientCase(
        symptoms=["胸口像压了石头", "胸痛放射到左臂", "出冷汗", "恶心"],
        sex="男", age=58,
    ))
    assert r.is_emergency
    assert r.emergency_advice
    assert not r.formula_suggestions, "急症不应给方剂"
    assert not r.syndromes, "急症不应输出辨证候选"


def test_consult_pregnant_notes():
    r = _top(PatientCase(
        symptoms=["食欲不振", "恶心", "疲乏"], sex="女", age=28, pregnant=True,
    ))
    assert any("妊娠" in n or "孕妇" in n or "孕期" in n for n in r.special_population_notes), \
        f"孕妇提示缺失: {r.special_population_notes}"


def test_consult_child_notes():
    r = _top(PatientCase(symptoms=["咳嗽", "食少"], age=3))
    assert any("儿" in n for n in r.special_population_notes), \
        f"小儿提示缺失: {r.special_population_notes}"


def test_consult_gynecological_followup():
    """育龄女性应追问月经情况（十问歌）。"""
    r = _top(PatientCase(symptoms=["疲乏", "头晕"], sex="女", age=30))
    assert any(q.dimension in ("月经带下", "月经") for q in r.followup_questions), \
        f"缺少月经追问: {[q.dimension for q in r.followup_questions]}"


def test_consult_disclaimer_always_present():
    r = _top(PatientCase(symptoms=["头痛"]))
    assert r.disclaimer and "执业" in r.disclaimer
    r2 = _top(PatientCase(symptoms=["胸痛放射至左臂"]))
    assert r2.disclaimer and "执业" in r2.disclaimer


def test_consult_otc_from_disease():
    """感冒风寒证应关联出常用中成药参考，且调护不串病（荨麻疹也含该分型）。"""
    r = _top(PatientCase(
        symptoms=["恶寒", "发热", "无汗", "鼻塞", "流清涕", "头痛"],
        tongue="苔薄白", pulse="脉浮紧",
    ))
    assert "风寒" in r.primary_syndrome
    otc_blob = "、".join(r.otc_suggestions)
    care_blob = "、".join(r.care_advice)
    assert any(k in otc_blob for k in ("荆防", "感冒", "风寒")), \
        f"应关联感冒类中成药: {r.otc_suggestions}"
    assert "荨麻疹" not in care_blob and "搔抓" not in care_blob, \
        f"调护建议串到荨麻疹: {r.care_advice}"
    assert isinstance(r.otc_suggestions, list)
    assert isinstance(r.care_advice, list)


def test_consult_disease_care_relevant():
    """心脾两虚的调护建议应来自失眠/心悸类疾病而非皮肤科疾病。"""
    r = _top(PatientCase(
        symptoms=["心悸", "失眠", "多梦", "健忘", "食少", "疲乏"],
        tongue="舌淡",
    ))
    assert "心脾两虚" in r.primary_syndrome
    care_blob = "、".join(r.care_advice)
    otc_blob = "、".join(r.otc_suggestions)
    assert "荨麻疹" not in care_blob and "抗组胺" not in care_blob, \
        f"调护串科: {r.care_advice}"
    assert "肤痒" not in otc_blob, f"OTC串科: {r.otc_suggestions}"
