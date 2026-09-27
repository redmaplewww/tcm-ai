"""Web服务API集成测试（离线：禁用LLM，走离线问卷/确定性引擎路径）。"""

import pytest
from fastapi.testclient import TestClient

from tcm_agent import server as srv
from tcm_agent.records import RecordStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv.llm, "llm_available", lambda: False)
    monkeypatch.setattr(srv, "_records", RecordStore(tmp_path))
    monkeypatch.setattr(srv, "_SESSIONS", {})
    return TestClient(srv.app)


def test_chat_offline_and_recorded(client):
    # 离线模式：llm_fn=None，返回追问
    r = client.post("/api/chat", json={"session": "w1", "message": "我头痛",
                                       "sex": "女", "age": 28})
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] and data["done"] is False

    # 档案已记录患者信息与对话
    recs = client.get("/api/records").json()["records"]
    assert any(x["id"] == "w1" for x in recs)
    detail = client.get("/api/records/w1").json()
    assert detail["patient"]["sex"] == "女" and detail["patient"]["age"] == 28
    assert detail["messages"][0]["text"] == "我头痛"

    # 会话未销毁：可继续发消息（服务端不主动结束）
    r2 = client.post("/api/chat", json={"session": "w1", "message": "还有点怕冷"})
    assert r2.status_code == 200
    assert len(client.get("/api/records/w1").json()["messages"]) == 4


def test_chat_emergency_keeps_session_and_records(client):
    r = client.post("/api/chat", json={
        "session": "w2", "message": "胸口剧痛放射到左臂，满头冷汗"})
    data = r.json()
    assert data["is_emergency"] is True
    assert data["final"] is not None and data["final"]["is_emergency"] is True
    # 急症后仍可继续留言（会话不销毁）
    r2 = client.post("/api/chat", json={"session": "w2", "message": "现在好一点了"})
    assert r2.status_code == 200


def test_records_delete(client):
    client.post("/api/chat", json={"session": "w3", "message": "头晕"})
    assert client.delete("/api/records/w3").json()["ok"] is True
    assert client.get("/api/records/w3").status_code == 404


def test_info_and_herb_safety(client):
    info = client.get("/api/info").json()
    assert info["kb"]["files"] > 100 and info["llm_available"] is False
    r = client.post("/api/herb-safety", json={"herbs": ["甘草", "甘遂"]})
    assert r.json()["is_safe"] is False


def test_query_offline_mode(client):
    r = client.post("/api/query", json={"question": "什么是阴虚", "mode": "offline"})
    assert r.status_code == 200 and len(r.json()["results"]) > 0


def test_restore_session_from_record(client):
    """服务重启后：同会话id自动从档案还原辨证上下文。"""
    # 预置档案：患者信息 + 对话 + 可信证据 + 报告
    srv._records.update_patient("r9", sex="女", age=35)
    srv._records.add_message("r9", "user", "心慌睡不着、吃不下饭")
    srv._records.add_message("r9", "assistant", "再补充些信息")
    srv._records.update_evidence("r9", ["心悸", "失眠", "食少", "疲乏"], tongue="舌淡")
    srv._records.add_final("r9", {"primary_syndrome": "心脾两虚证",
                                  "treatment_principle": "补益心脾",
                                  "symptoms_canonical": ["心悸", "失眠", "食少", "疲乏"]})

    session = srv._restore_session(srv._records, "r9")
    assert session.case.sex == "女" and session.case.age == 35
    assert set(session.case.symptoms) >= {"心悸", "失眠", "食少"}
    assert session.case.tongue == "舌淡"
    # 红旗/维度上下文回放
    assert any("心慌" in t for t in session.evidence_texts)
    # LLM历史回顾消息已注入
    assert len(session.history) == 2
    # 无档案的会话id走空白新建路径
    assert srv._has_record(srv._records, "never-exist-xyz") is False
    blank = srv._get_or_create(
        srv._records, "guest:never-exist-xyz",
        type("R", (), {"session": "never-exist-xyz", "reset": False,
                       "sex": None, "age": None, "pregnant": None})())
    assert blank.case.symptoms == []


def test_chat_records_evidence_for_resume(client):
    """每轮问诊后可信证据被持久化（供还原）。"""
    client.post("/api/chat", json={"session": "r10", "message": "我头痛失眠心慌",
                                   "sex": "男", "age": 40})
    rec = srv._records.get("r10")
    # 离线模式无LLM提取，但患者信息与消息已记录
    assert rec["patient"]["sex"] == "男"
    assert any(m["text"] == "我头痛失眠心慌" for m in rec["messages"])
