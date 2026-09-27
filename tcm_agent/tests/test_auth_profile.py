"""账号系统与用户画像测试。"""

import pytest
from fastapi.testclient import TestClient

from tcm_agent import auth, profile
from tcm_agent import server as srv
from tcm_agent.records import RecordStore


# ---------------------------------------------------------------- auth单元

def test_register_login_token(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    monkeypatch.setattr(auth, "SECRET_FILE", tmp_path / "secret.key")

    with pytest.raises(auth.AuthError):
        auth.register("a", "123456")          # 用户名过短
    with pytest.raises(auth.AuthError):
        auth.register("张三", "123")          # 密码过短
    info = auth.register("张三", "pass123456", nickname="三三")
    assert info["nickname"] == "三三"
    with pytest.raises(auth.AuthError):
        auth.register("张三", "another789")   # 重复注册

    assert auth.verify("张三", "pass123456")
    assert not auth.verify("张三", "wrong")
    assert not auth.verify("不存在", "pass123456")

    token = auth.issue_token("张三")
    assert auth.parse_token(token) == "张三"
    assert auth.parse_token(token + "x") is None          # 签名被改
    assert auth.parse_token("bad:token") is None          # 格式非法
    # 过期令牌
    expired = auth.issue_token("张三", days=-1)
    assert auth.parse_token(expired) is None


def test_password_hashed_not_plaintext(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    monkeypatch.setattr(auth, "SECRET_FILE", tmp_path / "secret.key")
    auth.register("李四", "secret99")
    import json
    data = json.load(open(tmp_path / "accounts.json", encoding="utf-8"))
    assert "secret99" not in str(data)
    assert "salt" in data["李四"] and "hash" in data["李四"]


# ---------------------------------------------------------------- 画像

def _record(sid, syndrome, symptoms, t=None):
    return {
        "id": sid, "created_at": t or 1.0, "updated_at": t or 1.0,
        "patient": {"sex": "女", "age": 30, "pregnant": False},
        "messages": [{"role": "user", "text": "x", "time": 1}],
        "finals": [{"time": 1, "result": {
            "primary_syndrome": syndrome,
            "treatment_principle": "",
            "symptoms_canonical": symptoms,
            "syndromes": [{"name": syndrome}, {"name": syndrome}],
        }}],
        "evidence": {"symptoms": symptoms, "tongue": "", "pulse": ""},
    }


def test_profile_constitution_inference():
    records = [
        _record("s1", "肝肾阴虚证", ["失眠", "盗汗", "腰膝酸软"]),
        _record("s2", "肾阴虚证", ["失眠", "五心烦热"]),
        _record("s3", "心脾两虚证", ["心悸", "失眠", "食少"]),
    ]
    prof = profile.compute_profile("测试用户", records)
    assert prof.sessions == 3
    assert prof.dominant_constitution == "阴虚质"
    assert prof.constitution_scores["阴虚质"] > prof.constitution_scores["气虚质"]
    assert prof.constitution_confident
    assert "失眠" in prof.chronic_symptoms  # 跨3次出现
    assert len(prof.syndrome_history) == 3
    assert any("熬夜" in a for a in prof.lifestyle_advice)


def test_profile_acute_not_counted():
    records = [
        _record("s1", "风寒束表证（表寒）", ["恶寒", "无汗"]),
        _record("s2", "肾阴虚证", ["腰膝酸软"]),
    ]
    prof = profile.compute_profile("u", records)
    # 急性外感证不计入体质
    assert "dominant_constitution" in prof.model_fields
    assert prof.dominant_constitution == "阴虚质"
    assert prof.constitution_scores.get("阴虚质") == 1.0


def test_profile_empty():
    prof = profile.compute_profile("u", [])
    assert prof.sessions == 0 and prof.dominant_constitution == ""
    assert prof.lifestyle_advice == []


def test_lifestyle_table_covers_constitutions():
    for c in ("阴虚质", "阳虚质", "气虚质", "气郁质", "痰湿质", "湿热质", "血瘀质", "特禀质", "平和质"):
        advice = profile.lifestyle_advice(c)
        assert len(advice) >= 3, f"{c}建议不足"
    assert profile.lifestyle_advice("") == []
    assert profile.lifestyle_advice("未知体质") == []


# ---------------------------------------------------------------- 服务端集成

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    monkeypatch.setattr(auth, "SECRET_FILE", tmp_path / "secret.key")
    monkeypatch.setattr(srv.llm, "llm_available", lambda: False)
    monkeypatch.setattr(srv, "_records", RecordStore(tmp_path))  # 与生产一致：游客存储=RECORDS_DIR根
    monkeypatch.setattr(srv, "_SESSIONS", {})
    monkeypatch.setattr(srv, "RECORDS_DIR", tmp_path)
    import tcm_agent.records as records_mod
    monkeypatch.setattr(records_mod, "RECORDS_DIR", tmp_path)
    return TestClient(srv.app)


def test_auth_flow_and_user_scoping(client):
    # 注册→登录
    r = client.post("/api/auth/register", json={"username": "王五", "password": "pass1234"})
    assert r.json()["token"]
    token = r.json()["token"]
    assert client.post("/api/auth/login",
                       json={"username": "王五", "password": "wrong"}).status_code == 401
    token2 = client.post("/api/auth/login",
                         json={"username": "王五", "password": "pass1234"}).json()["token"]
    assert auth.parse_token(token2) == "王五"
    assert client.get("/api/auth/me", params={"token": token}).json()["username"] == "王五"

    # 用户问诊 → 档案进入用户空间
    client.post("/api/chat", json={"session": "u1", "message": "我头痛",
                                   "token": token, "sex": "女", "age": 30})
    user_recs = client.get("/api/records", params={"token": token}).json()["records"]
    guest_recs = client.get("/api/records").json()["records"]
    assert any(x["id"] == "u1" for x in user_recs)
    assert all(x["id"] != "u1" for x in guest_recs)  # 游客看不到用户档案

    # 画像接口
    prof = client.get("/api/profile", params={"token": token}).json()
    assert prof["username"] == "王五" and prof["sessions"] >= 1
    assert client.get("/api/profile").status_code == 401


def test_import_guest_records(client):
    # 游客产生档案
    client.post("/api/chat", json={"session": "g1", "message": "我头晕"})
    token = client.post("/api/auth/register",
                        json={"username": "赵六", "password": "pass1234"}).json()["token"]
    r = client.post("/api/auth/import-guest", json={"token": token})
    assert r.json()["imported"] >= 1
    recs = client.get("/api/records", params={"token": token}).json()["records"]
    assert any(x["id"] == "g1" for x in recs)
    # 再导一次不重复
    assert client.post("/api/auth/import-guest", json={"token": token}).json()["imported"] == 0


def test_consult_with_constitution_lifestyle():
    from tcm_agent.models import PatientCase
    from tcm_agent import reasoning
    case = PatientCase(symptoms=["腰膝酸软", "五心烦热", "盗汗", "失眠"])
    r1 = reasoning.consult(case)
    # 无画像时从主证推导体质（肾阴虚→阴虚质）也给建议
    assert any("熬夜" in a for a in r1.lifestyle_advice), r1.lifestyle_advice
    # 显式体质优先
    r2 = reasoning.consult(case, constitution="气郁质")
    assert any("疏肝" in a or "运动" in a for a in r2.lifestyle_advice)
    # 急性证不推导体质建议
    r3 = reasoning.consult(PatientCase(symptoms=["恶寒", "发热", "无汗", "头痛"]))
    assert r3.lifestyle_advice == []
