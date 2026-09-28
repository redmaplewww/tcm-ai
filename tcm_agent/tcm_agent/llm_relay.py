"""本地LLM继服务（LLM Relay）—— 部署架构的安全核心。

架构（服务器无HTTPS、不能挂明文API密钥的解决方案）：

    公网用户 ──HTTP──> red服务器(FastAPI前端+知识库+确定性引擎)
                          │ SSH加密反向隧道（服务器127.0.0.1:9000 → 本机8001）
                          ▼
                    本机LLM Relay（127.0.0.1:8001，仅回环）
                          │ 注入真实密钥，HTTPS外联
                          ▼
                    GLM（open.bigmodel.cn/api/anthropic）

- 真实API密钥只存在本机内存/本机ZCode配置，永不出现在服务器与网络明文中；
- Relay只绑定127.0.0.1，唯一入口是SSH隧道；
- 对请求做体积限制与路径白名单，透传Anthropic协议（/v1/messages）。

启动：python -m tcm_agent.llm_relay [--port 8001]
"""

from __future__ import annotations

import argparse
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import llm

app = FastAPI(title="TCM LLM Relay", version="1.0")

#: 允许转发的路径（Anthropic协议）
_ALLOWED_PATHS = {"/v1/messages"}

#: 请求体上限（结构化输出+工具定义足够）
_MAX_BODY = 2 * 1024 * 1024

#: 上游客户端（懒加载，持有真实密钥）
_client: Optional[httpx.AsyncClient] = None


def _upstream() -> tuple[str, str]:
    """返回（上游base_url, 真实api_key）。"""
    key, base, _model = llm.llm_credentials()
    return base.rstrip("/"), key


def _client_get() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
    return _client


@app.get("/healthz")
async def healthz():
    key, base, model = llm.llm_credentials()
    return {
        "ok": bool(key),
        "upstream": base,
        "model": model,
        "has_key": bool(key),
    }


@app.api_route("/{path:path}", methods=["POST", "GET"])
async def relay(path: str, request: Request):
    full_path = "/" + path
    if full_path not in _ALLOWED_PATHS and not full_path.startswith("/healthz"):
        return JSONResponse({"error": f"path not allowed: {full_path}"}, status_code=403)

    body = await request.body()
    if len(body) > _MAX_BODY:
        return JSONResponse({"error": "request too large"}, status_code=413)

    base, key = _upstream()
    if not key:
        return JSONResponse({"error": "relay has no upstream key"}, status_code=503)

    headers = {
        "content-type": request.headers.get("content-type", "application/json"),
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        # 真实密钥仅在此注入（本机→上游为HTTPS）
        "authorization": f"Bearer {key}",
        "x-api-key": key,
    }
    try:
        resp = await _client_get().post(base + full_path, content=body,
                                        headers=headers)
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"))
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream error: {e}"}, status_code=502)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tcm-llm-relay")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    key, base, model = llm.llm_credentials()
    print(f"LLM Relay（上游 {base}，模型 {model}，密钥{'已加载' if key else '缺失！'}）")
    print(f"监听 {args.host}:{args.port}（仅供SSH隧道调用，勿暴露公网）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
