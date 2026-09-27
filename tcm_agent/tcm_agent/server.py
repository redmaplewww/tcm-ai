"""Web测试服务：FastAPI后端 + 单页前端。

启动：
  python -m tcm_agent.server            # 默认 http://127.0.0.1:8000
  python -m tcm_agent.server --port 8080

API：
  GET  /                     前端页面
  POST /api/chat             多轮问诊（LLM对话+证据取证+动态复辨）
  POST /api/query            知识查询（离线检索 / LLM问答）
  POST /api/herb-safety      中药配伍安全检查
  GET  /api/info             知识库与数据统计
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import auth, llm, profile as profile_mod, reasoning, tools
from .chat_session import ConsultSession, make_agent_llm_fn
from .kb import get_kb
from .models import PatientCase
from .records import RECORDS_DIR, RecordStore

app = FastAPI(title="中医AI问诊与查询", version="0.1.0")

#: 会话池（单机测试用途；不再自动销毁——辨证后转入咨询模式，直到用户开新会话）
_SESSIONS: dict[str, ConsultSession] = {}
_SESSION_CAP = 60

#: 游客档案存储；登录用户档案在 tcm_records/_u_<用户名>/
_records = RecordStore()


def _user_store(token: Optional[str]) -> tuple[Optional[str], RecordStore]:
    """解析令牌 -> (用户名|None, 该用户的档案存储)。未登录返回游客存储。"""
    username = auth.parse_token(token) if token else None
    if username:
        return username, RecordStore(RECORDS_DIR / f"_u_{username}")
    return None, _records


def _user_profile_summary(username: str, store: RecordStore) -> tuple[str, str]:
    """返回 (体质, 画像摘要文本) 供会话个性化。"""
    prof = profile_mod.compute_profile(username, store.all())
    if not prof.sessions:
        return "", ""
    parts = [f"用户{prof.username}，累计问诊{prof.sessions}次。"]
    if prof.dominant_constitution:
        conf = "较明确" if prof.constitution_confident else "初步倾向"
        parts.append(f"体质{conf}：{prof.dominant_constitution}"
                     + (f"（其他倾向：{','.join(list(prof.constitution_scores)[1:3])}）"
                        if len(prof.constitution_scores) > 1 else ""))
    if prof.chronic_symptoms:
        parts.append("跨次反复出现的症状：" + "、".join(prof.chronic_symptoms[:6]))
    elif prof.top_symptoms:
        parts.append("常见症状：" + "、".join(prof.top_symptoms[:6]))
    if prof.syndrome_history:
        lasts = [h.syndrome for h in prof.syndrome_history[-3:] if h.syndrome]
        if lasts:
            parts.append("近期辨证：" + "、".join(lasts))
    return prof.dominant_constitution, "；".join(parts)


# ---------------------------------------------------------------- 请求模型

class ChatRequest(BaseModel):
    session: str = ""
    message: str = ""
    force: bool = False
    reset: bool = False
    sex: Optional[str] = None
    age: Optional[int] = None
    pregnant: Optional[bool] = None
    token: str = ""  # 登录令牌（未登录为空=游客模式）


class AuthRequest(BaseModel):
    username: str = ""
    password: str = ""
    nickname: str = ""
    token: str = ""


class QueryRequest(BaseModel):
    question: str
    mode: str = "auto"  # auto / offline / llm


class HerbSafetyRequest(BaseModel):
    herbs: list[str]
    pregnant: bool = False


# ---------------------------------------------------------------- 工具

def _assessment_summary(result) -> dict[str, Any]:
    if result is None:
        return None
    return {
        "primary_syndrome": result.primary_syndrome,
        "treatment_principle": result.treatment_principle,
        "top": [
            {"name": c.name, "system": c.system, "confidence": c.confidence}
            for c in result.syndromes
        ],
        "symptoms_canonical": result.symptoms_canonical,
    }


def _has_record(store: RecordStore, session_id: str) -> bool:
    try:
        rec = store.get(session_id)
    except ValueError:
        return False
    return bool(rec.get("messages") or rec.get("finals"))


def _restore_session(store: RecordStore, session_id: str,
                     constitution: str = "", profile_summary: str = "") -> ConsultSession:
    """从档案还原会话：患者信息 + 可信证据 + 红旗/维度上下文 + LLM历史回顾 + 画像。"""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    rec = store.get(session_id)
    patient = rec.get("patient") or {}
    evidence = rec.get("evidence") or {}
    final = (rec.get("finals") or [{}])[-1].get("result") or {}
    symptoms = evidence.get("symptoms") or final.get("symptoms_canonical") or []

    case = PatientCase(
        sex=patient.get("sex"),
        age=patient.get("age"),
        pregnant=bool(patient.get("pregnant")),
        symptoms=list(symptoms),
        tongue=evidence.get("tongue") or "",
        pulse=evidence.get("pulse") or "",
    )
    session = ConsultSession(
        make_agent_llm_fn() if llm.llm_available() else None, case=case,
        constitution=constitution, profile_summary=profile_summary,
    )
    # 红旗检测与问诊维度的上下文回放（历史用户陈述）
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            session.evidence_texts.append(str(m.get("text", "")))
    # LLM对话上下文回放：合成历史回顾消息（追加在画像消息之后）
    summary_parts = [
        str(m.get("text", "")) for m in rec.get("messages", []) if m.get("role") == "user"
    ]
    summary = "；".join(summary_parts)[-1500:]
    if final.get("primary_syndrome"):
        summary += f"（此前辨证结论：{final.get('primary_syndrome')}，治法：{final.get('treatment_principle', '')}）"
    if summary:
        session.history.append(
            ModelRequest(parts=[UserPromptPart(content="[历史会话回顾] " + summary)])
        )
        session.history.append(
            ModelResponse(parts=[TextPart(content="好的，我已了解此前的问诊情况，将在此基础上继续。")])
        )
    return session


def _get_or_create(store: RecordStore, session_key: str, req: ChatRequest,
                   constitution: str = "", profile_summary: str = "") -> ConsultSession:
    if req.reset or session_key not in _SESSIONS:
        sid = req.session or session_key
        if not req.reset and _has_record(store, sid):
            # 档案存在（服务重启/换端后）：还原会话继续，而不是从零开始
            session = _restore_session(store, sid, constitution, profile_summary)
        else:
            case = PatientCase(
                sex=(req.sex if req.sex in ("男", "女") else None),
                age=req.age,
                pregnant=bool(req.pregnant),
            )
            session = ConsultSession(
                make_agent_llm_fn() if llm.llm_available() else None, case=case,
                constitution=constitution, profile_summary=profile_summary,
            )
        _SESSIONS[session_key] = session
        if len(_SESSIONS) > _SESSION_CAP:  # 淘汰最旧会话（档案已持久化，不丢失）
            oldest = next(iter(_SESSIONS))
            _SESSIONS.pop(oldest, None)
    else:
        session = _SESSIONS[session_key]
        # 允许中途补充人口学信息
        if req.sex in ("男", "女") and not session.case.sex:
            session.case.sex = req.sex
        if req.age is not None and session.case.age is None:
            session.case.age = req.age
        if req.pregnant:
            session.case.pregnant = True
    return session


# ---------------------------------------------------------------- API

#: 前端页面（每次请求重新构造，避免文件更新后Content-Length过期导致响应损坏）
_INDEX_HTML = Path(__file__).parent / "web" / "index.html"


@app.get("/")
def index():
    return FileResponse(str(_INDEX_HTML), media_type="text/html")


@app.get("/api/info")
def api_info():
    from . import data
    kb = get_kb()
    stats = kb.stats()
    return {
        "kb": stats,
        "llm_available": llm.llm_available(),
        "model": llm.llm_credentials()[2] if llm.llm_available() else "离线模式",
        "data": {
            "证型": len(data.syndromes()),
            "中药": len(data.herbs()),
            "方剂": len(data.formulas()),
            "穴位": len(data.acupoints()),
            "疾病": len(data.diseases()),
            "红旗规则": len(data.red_flags()),
            "术语": len(data.glossary()),
        },
    }


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    session_id = req.session or "default"
    username, store = _user_store(req.token)
    constitution, profile_summary = ("", "")
    if username:
        constitution, profile_summary = _user_profile_summary(username, store)
    session_key = f"{username or 'guest'}:{session_id}"
    session = _get_or_create(store, session_key, req, constitution, profile_summary)
    message = (req.message or "").strip()
    force = req.force or message in ("/诊断", "/分析", "/辨证")
    if not message and not force:
        return JSONResponse({"error": "message不能为空"}, status_code=400)

    result = await session.step(message, force_analysis=force)

    # 问诊档案记录（按用户隔离；失败不阻断问诊）
    try:
        store.update_patient(session_id, sex=req.sex, age=req.age, pregnant=req.pregnant)
        if message:
            store.add_message(session_id, "user", message)
        meta: dict[str, Any] = {}
        if result.challenges:
            meta["challenges"] = [c.question for c in result.challenges]
        if result.syndrome_change:
            meta["syndrome_change"] = result.syndrome_change
        if result.excluded_evidence:
            meta["excluded_evidence"] = result.excluded_evidence
        store.add_message(session_id, "assistant", result.reply, meta=meta or None)
        if result.final is not None:
            store.add_final(session_id, result.final.model_dump())
        # 持久化当前可信证据（供重启/换端后还原会话）
        store.update_evidence(
            session_id,
            session.engine.credible_symptoms(),
            session.engine.credible_tongue(),
            session.engine.credible_pulse(),
        )
    except Exception:  # noqa: BLE001
        pass

    resp: dict[str, Any] = {
        "session": session_id,
        "user": username,
        "reply": result.reply,
        "challenges": [c.model_dump() for c in result.challenges],
        "excluded_evidence": result.excluded_evidence,
        "assessment": _assessment_summary(result.assessment),
        "syndrome_change": result.syndrome_change,
        "is_emergency": result.is_emergency,
        "emergency_advice": result.emergency_advice,
        "final": result.final.model_dump() if result.final else None,
        "turns": result.turns,
        "done": result.final is not None,
        "continued": True,  # 报告后仍可继续对话（咨询模式）
    }
    return resp


@app.post("/api/query")
async def api_query(req: QueryRequest):
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "question不能为空"}, status_code=400)
    use_llm = (req.mode == "llm") or (req.mode == "auto" and llm.llm_available())
    if use_llm and llm.llm_available():
        from .agents import build_query_agent
        agent = build_query_agent()
        try:
            out = await agent.run(question)
            return {
                "mode": "llm",
                "answer": out.answer,
                "key_points": out.key_points,
                "sources": [s.model_dump() for s in out.sources],
                "disclaimer": out.disclaimer,
            }
        except Exception as e:  # noqa: BLE001 - 前端需要降级提示
            return {
                "mode": "llm_failed",
                "error": f"LLM调用失败：{e}",
                "offline": _offline_search(question),
            }
    return {"mode": "offline", **_offline_search(question)}


def _offline_search(question: str) -> dict[str, Any]:
    chunks = get_kb().search(question, k=5)
    return {
        "results": [
            {
                "title": c.title,
                "section": c.section,
                "file": c.file,
                "excerpt": c.excerpt(300),
            }
            for c in chunks
        ],
        "disclaimer": "以上为知识库检索结果，供参考；诊疗请咨询执业中医师。",
    }


@app.post("/api/herb-safety")
def api_herb_safety(req: HerbSafetyRequest):
    result = reasoning.check_herb_combination(req.herbs, pregnant=req.pregnant)
    return result.model_dump()


# ---------------------------------------------------------------- 账号

@app.post("/api/auth/register")
def api_register(req: AuthRequest):
    try:
        info = auth.register(req.username, req.password, req.nickname)
    except auth.AuthError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "username": info["username"],
            "nickname": info["nickname"], "token": auth.issue_token(info["username"])}


@app.post("/api/auth/login")
def api_login(req: AuthRequest):
    if not auth.verify(req.username, req.password):
        return JSONResponse({"error": "用户名或密码错误"}, status_code=401)
    info = auth.get_account(req.username) or {}
    return {"ok": True, "username": req.username,
            "nickname": info.get("nickname", req.username),
            "token": auth.issue_token(req.username)}


@app.get("/api/auth/me")
def api_me(token: str = ""):
    username = auth.parse_token(token)
    if not username:
        return JSONResponse({"error": "未登录或令牌失效"}, status_code=401)
    info = auth.get_account(username) or {}
    return {"username": username, "nickname": info.get("nickname", username)}


# ---------------------------------------------------------------- 用户画像

@app.get("/api/profile")
def api_profile(token: str = ""):
    username = auth.parse_token(token)
    if not username:
        return JSONResponse({"error": "未登录或令牌失效"}, status_code=401)
    _, store = _user_store(token)
    prof = profile_mod.compute_profile(username, store.all())
    return prof.model_dump()


# ---------------------------------------------------------------- 健康档案

@app.get("/api/records")
def api_records(token: str = ""):
    _, store = _user_store(token)
    return {"records": store.list()}


@app.get("/api/records/{session_id}")
def api_record_detail(session_id: str, token: str = ""):
    _, store = _user_store(token)
    try:
        rec = store.get(session_id)
    except ValueError:
        return JSONResponse({"error": "档案不存在"}, status_code=404)
    if not rec.get("messages") and not rec.get("finals"):
        return JSONResponse({"error": "档案不存在"}, status_code=404)
    return rec


@app.delete("/api/records/{session_id}")
def api_record_delete(session_id: str, token: str = ""):
    username, store = _user_store(token)
    try:
        ok = store.delete(session_id)
    except ValueError:
        return JSONResponse({"ok": False}, status_code=400)
    _SESSIONS.pop(f"{username or 'guest'}:{session_id}", None)
    return {"ok": ok}


@app.post("/api/auth/import-guest")
def api_import_guest(req: AuthRequest):
    """把游客档案导入当前账号（不覆盖已有档案）。"""
    username = auth.parse_token(req.token)
    if not username:
        return JSONResponse({"error": "未登录或令牌失效"}, status_code=401)
    user_dir = RECORDS_DIR / f"_u_{username}"
    user_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    for path in RECORDS_DIR.glob("*.json"):
        target = user_dir / path.name
        if target.exists():
            continue
        shutil.copy2(path, target)
        imported += 1
    return {"ok": True, "imported": imported}


# ---------------------------------------------------------------- 入口

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tcm-server", description="中医AI Web测试服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    import uvicorn
    print(f"中医AI Web服务：http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
