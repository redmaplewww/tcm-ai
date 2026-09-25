"""数据层完整性测试：tcm_kb/data/*.json 的结构与规模。"""

from tcm_agent import data
from tcm_agent.config import DATA_DIR


def _has(items: list, field: str) -> None:
    for it in items:
        assert it.get(field) not in (None, ""), f"条目缺少字段 {field}: {it}"


def test_data_files_exist():
    for name in (
        "syndromes.json", "herbs.json", "formulas.json", "acupoints.json",
        "diseases.json", "red_flags.json", "glossary.json",
        "symptom_synonyms.json", "colloquial_map.json",
    ):
        assert (DATA_DIR / name).exists(), f"缺少数据文件 {name}"


def test_syndromes_quality():
    syn = data.syndromes()
    assert len(syn) >= 70, f"证型数不足：{len(syn)}"
    for s in syn:
        _has([s], "name")
        _has([s], "treatment")
        keys = s.get("key_symptoms") or []
        assert isinstance(keys, list) and len(keys) >= 2, f"{s.get('name')} 关键症状不足2条"
        assert all(isinstance(k, str) and k.strip() for k in keys)
    names = [s["name"] for s in syn]
    assert len(names) == len(set(names)), "证型名存在重复"


def test_herbs_quality():
    herbs = data.herbs()
    assert len(herbs) >= 150, f"中药数不足：{len(herbs)}"
    natures = {"寒", "微寒", "凉", "平", "温", "热", "大寒", "微温", "大热"}
    for h in herbs:
        _has([h], "name")
        assert h.get("nature", "") in natures or h.get("nature", "") == "", \
            f"{h.get('name')} 性味异常: {h.get('nature')}"


def test_formulas_quality():
    fs = data.formulas()
    assert len(fs) >= 110, f"方剂数不足：{len(fs)}"
    otc = [f for f in fs if f.get("otc")]
    classic = [f for f in fs if not f.get("otc")]
    assert len(otc) >= 50, f"OTC中成药不足：{len(otc)}"
    assert len(classic) >= 60, f"经典方不足：{len(classic)}"


def test_acupoints_quality():
    aps = data.acupoints()
    assert len(aps) >= 90, f"穴位数不足：{len(aps)}"
    for a in aps:
        _has([a], "name")
        assert isinstance(a.get("pregnancy_caution", False), bool)


def test_diseases_quality():
    ds = data.diseases()
    assert len(ds) >= 45, f"疾病数不足：{len(ds)}"
    for d in ds:
        _has([d], "name")
        stypes = d.get("syndrome_types") or []
        assert len(stypes) >= 1, f"{d.get('name')} 无证型分型"


def test_red_flags_quality():
    rf = data.red_flags()
    assert len(rf) >= 16, f"红旗规则不足：{len(rf)}"
    levels = {"emergency", "urgent", "see_doctor"}
    for r in rf:
        assert r.get("level") in levels, f"红旗level非法: {r.get('level')}"
        _has([r], "advice")
        assert r.get("any_of") or r.get("all_of"), f"{r.get('id')} 无触发词"
    assert any(r["level"] == "emergency" for r in rf), "缺少emergency级规则"


def test_glossary_quality():
    gl = data.glossary()
    assert len(gl) >= 85, f"术语数不足：{len(gl)}"
    for g in gl:
        _has([g], "term")
        _has([g], "definition")


def test_symptom_maps_quality():
    syn, collo = data.symptom_maps()
    assert len(syn) >= 50, f"同义词映射不足：{len(syn)}"
    assert len(collo) >= 20, f"口语映射不足：{len(collo)}"
    assert "上火" in collo and "失眠多梦" in collo


def test_exact_lookups():
    herb = data.get_herb("麻黄")
    assert herb is not None and "温" in herb.get("nature", "")
    f = data.get_formula("六味地黄丸")
    assert f is not None and "熟地黄" in (f.get("composition") or [])
    ap = data.get_acupoint("足三里")
    assert ap is not None and ap.get("code") == "ST36"
    hegu = data.get_acupoint("合谷")
    assert hegu is not None and hegu.get("pregnancy_caution") is True
    syn = data.get_syndrome("肝气郁结")
    assert syn is not None and "弦" in (syn.get("pulse") or "")
