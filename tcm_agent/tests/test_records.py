"""健康档案存储与Web服务集成测试。"""

import pytest

from tcm_agent.records import RecordStore


def test_record_lifecycle(tmp_path):
    store = RecordStore(tmp_path)
    store.update_patient("s1", sex="女", age=30)
    store.add_message("s1", "user", "我怕冷")
    store.add_message("s1", "assistant", "还有别的症状吗？", meta={"challenges": ["x vs y"]})
    store.add_final("s1", {"primary_syndrome": "表寒证", "symptoms_canonical": ["恶寒", "无汗"]})

    rec = store.get("s1")
    assert rec["patient"] == {"sex": "女", "age": 30, "pregnant": False}
    assert len(rec["messages"]) == 2
    assert rec["messages"][1]["meta"]["challenges"] == ["x vs y"]
    assert rec["finals"][0]["result"]["primary_syndrome"] == "表寒证"

    items = store.list()
    assert len(items) == 1
    assert items[0]["id"] == "s1"
    assert items[0]["primary_syndrome"] == "表寒证"
    assert items[0]["has_final"] is True
    assert "恶寒" in items[0]["symptoms"]


def test_record_persist_across_store_instances(tmp_path):
    store = RecordStore(tmp_path)
    store.add_message("s1", "user", "头痛")
    store2 = RecordStore(tmp_path)  # 重启后新实例
    rec = store2.get("s1")
    assert rec["messages"][0]["text"] == "头痛"


def test_record_new_skeleton_and_delete(tmp_path):
    store = RecordStore(tmp_path)
    rec = store.get("brand-new")
    assert rec["messages"] == [] and rec["patient"]["sex"] is None
    store.add_message("brand-new", "user", "x")
    assert store.delete("brand-new") is True
    assert store.get("brand-new")["messages"] == []
    assert store.delete("brand-new") is False


def test_record_rejects_unsafe_id(tmp_path):
    store = RecordStore(tmp_path)
    with pytest.raises(ValueError):
        store.add_message("../evil", "user", "x")
    with pytest.raises(ValueError):
        store.get("a/b")


def test_record_empty_session_summary(tmp_path):
    store = RecordStore(tmp_path)
    store.add_message("s2", "user", "我最近头晕")  # 无final（进行中）
    items = store.list()
    assert items[0]["primary_syndrome"] == "" and items[0]["has_final"] is False
    assert items[0]["symptoms"] == ["我最近头晕"]


def test_record_evidence_persistence(tmp_path):
    store = RecordStore(tmp_path)
    store.update_evidence("s3", ["心悸", "失眠"], tongue="舌淡", pulse="脉细弱")
    rec = RecordStore(tmp_path).get("s3")  # 新实例读取（模拟重启）
    assert rec["evidence"] == {"symptoms": ["心悸", "失眠"], "tongue": "舌淡", "pulse": "脉细弱"}
