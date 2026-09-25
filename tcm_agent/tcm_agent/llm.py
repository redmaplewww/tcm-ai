"""真实LLM接入：GLM（Anthropic兼容端点，open.bigmodel.cn）。

凭据解析优先级：
1. 环境变量：TCM_LLM_KEY / TCM_LLM_BASE_URL / TCM_LLM_MODEL
2. 本机ZCode CLI配置（~/.zcode/v2/config.json）中 builtin:bigmodel-coding-plan
   的 apiKey 与 baseURL（复用本机已登录的GLM凭据，便于直接测试）

解析失败时 build_model() 返回 None，系统退回离线模式（确定性推理引擎+知识库
检索），所有离线功能不受影响。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

DEFAULT_MODEL_NAME = os.environ.get("TCM_LLM_MODEL", "GLM-5.3-Flash")
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/anthropic"

#: 长输出+思考+工具调用需要足够的token预算
LLM_MODEL_SETTINGS = AnthropicModelSettings(max_tokens=8192, timeout=240)


def _zcode_provider_creds() -> tuple[str, str]:
    """读取本机ZCode CLI配置中的bigmodel凭据。"""
    path = Path.home() / ".zcode" / "v2" / "config.json"
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        prov = cfg["provider"]["builtin:bigmodel-coding-plan"]["options"]
        return prov["apiKey"], prov.get("baseURL", DEFAULT_BASE_URL)
    except (OSError, KeyError, json.JSONDecodeError):
        return "", ""


def llm_credentials() -> tuple[str, str, str]:
    """返回 (api_key, base_url, model_name)；无key时api_key为空串。"""
    key = os.environ.get("TCM_LLM_KEY", "")
    base = os.environ.get("TCM_LLM_BASE_URL", "")
    if not key:
        key, zbase = _zcode_provider_creds()
        base = base or zbase
    return key, base or DEFAULT_BASE_URL, DEFAULT_MODEL_NAME


def llm_available() -> bool:
    return bool(llm_credentials()[0])


@lru_cache(maxsize=1)
def build_model() -> Optional[AnthropicModel]:
    """构建真实LLM模型实例；凭据不可用时返回None。"""
    key, base, name = llm_credentials()
    if not key:
        return None
    from pydantic_ai.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key=key, base_url=base)
    return AnthropicModel(name, provider=provider, settings=LLM_MODEL_SETTINGS)
