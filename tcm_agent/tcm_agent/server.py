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
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import llm, reasoning, tools
from .chat_session import ConsultSession, make_agent_llm_fn
from .kb import get_kb
from .models import PatientCase
from .records import RecordStore

app = FastAPI(title="中医AI问诊与查询", version="0.1.0")

#: 会话池（单机测试用途；不再自动销毁——辨证后转入咨询模式，直到用户开新会话）
_SESSIONS: dict[str, ConsultSession] = {}
_SESSION_CAP = 60

#: 问诊档案（每会话一JSON，持久化到 tcm_records/）
_records = RecordStore()

_INDEX_HTML = FileResponse(
    str(__import__("pathlib").Path(__file__).parent / "web" / "index.html"),
    media_type="text/html",
)


# ---------------------------------------------------------------- 请求模型

class ChatRequest(BaseModel):
    session: str = ""
    message: str = ""
    force: bool = False
    reset: bool = False
    sex: Optional[str] = None
    age: Optional[int] = None
    pregnant: Optional[bool] = None


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


def _get_or_create(session_id: str, req: ChatRequest) -> ConsultSession:
    if req.reset or session_id not in _SESSIONS:
        case = PatientCase(
            sex=(req.sex if req.sex in ("男", "女") else None),
            age=req.age,
            pregnant=bool(req.pregnant),
        )
        session = ConsultSession(
            make_agent_llm_fn() if llm.llm_available() else None, case=case
        )
        _SESSIONS[session_id] = session
        if len(_SESSIONS) > _SESSION_CAP:  # 淘汰最旧会话（档案已持久化，不丢失）
            oldest = next(iter(_SESSIONS))
            _SESSIONS.pop(oldest, None)
    else:
        session = _SESSIONS[session_id]
        # 允许中途补充人口学信息
        if req.sex in ("男", "女") and not session.case.sex:
            session.case.sex = req.sex
        if req.age is not None and session.case.age is None:
            session.case.age = req.age
        if req.pregnant:
            session.case.pregnant = True
    return session


# ---------------------------------------------------------------- API

@app.get("/")
def index():
    return _INDEX_HTML


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
    session = _get_or_create(session_id, req)
    message = (req.message or "").strip()
    force = req.force or message in ("/诊断", "/分析", "/辨证")
    if not message and not force:
        return JSONResponse({"error": "message不能为空"}, status_code=400)

    result = await session.step(message, force_analysis=force)

    # 问诊档案记录（失败不阻断问诊）
    try:
        _records.update_patient(session_id, sex=req.sex, age=req.age, pregnant=req.pregnant)
        if message:
            _records.add_message(session_id, "user", message)
        meta: dict[str, Any] = {}
        if result.challenges:
            meta["challenges"] = [c.question for c in result.challenges]
        if result.syndrome_change:
            meta["syndrome_change"] = result.syndrome_change
        if result.excluded_evidence:
            meta["excluded_evidence"] = result.excluded_evidence
        _records.add_message(session_id, "assistant", result.reply, meta=meta or None)
        if result.final is not None:
            _records.add_final(session_id, result.final.model_dump())
    except Exception:  # noqa: BLE001
        pass

    resp: dict[str, Any] = {
        "session": session_id,
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


# ---------------------------------------------------------------- 健康档案

@app.get("/api/records")
def api_records():
    return {"records": _records.list()}


@app.get("/api/records/{session_id}")
def api_record_detail(session_id: str):
    rec = _records.get(session_id)
    if not rec.get("messages") and not rec.get("finals"):
        return JSONResponse({"error": "档案不存在"}, status_code=404)
    return rec


@app.delete("/api/records/{session_id}")
def api_record_delete(session_id: str):
    ok = _records.delete(session_id)
    _SESSIONS.pop(session_id, None)
    return {"ok": ok}


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
