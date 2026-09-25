"""路径与模型配置。"""

from __future__ import annotations

import os
from pathlib import Path

# tcm_agent/config.py -> tcm_agent/ -> tcm_agent/ -> 中医AI/
BASE_DIR = Path(__file__).resolve().parents[2]

#: 分级Markdown知识库根目录（可用环境变量覆盖）
KB_ROOT = Path(os.environ.get("TCM_KB_ROOT", str(BASE_DIR / "tcm_kb")))

#: 机器可读数据层目录
DATA_DIR = KB_ROOT / "data"

#: LLM模型标识，例如 'openai:gpt-4o-mini' / 'openai:deepseek-chat'（需相应API key）
#: 留空时Agent仍可构建，但只有离线推理引擎可用（不调用LLM）。
MODEL_SETTING = os.environ.get("TCM_MODEL", "openai:gpt-4o-mini")


def default_model() -> str:
    """返回默认模型标识（供pydantic-ai Agent构建）。"""
    return MODEL_SETTING


def llm_ready() -> bool:
    """粗略判断是否配置了常见LLM API key。"""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
        if os.environ.get(key):
            return True
    return False
