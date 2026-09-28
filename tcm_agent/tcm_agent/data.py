"""结构化数据层加载：tcm_kb/data/*.json。

所有数据文件顶层为JSON数组（UTF-8）。加载带缓存；文件缺失时返回空列表
（便于知识库尚未生成时系统仍可启动）。
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

from .config import DATA_DIR


@lru_cache(maxsize=None)
def _load(filename: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return data
    return []


def syndromes() -> list[dict[str, Any]]:
    """证型库（syndromes.json）。"""
    return _load("syndromes.json")


def herbs() -> list[dict[str, Any]]:
    """中药库（herbs.json）。"""
    return _load("herbs.json")


def formulas() -> list[dict[str, Any]]:
    """方剂库（formulas.json）。"""
    return _load("formulas.json")


def acupoints() -> list[dict[str, Any]]:
    """穴位库（acupoints.json）。"""
    return _load("acupoints.json")


def diseases() -> list[dict[str, Any]]:
    """疾病库（diseases.json）。"""
    return _load("diseases.json")


def red_flags() -> list[dict[str, Any]]:
    """红旗规则（red_flags.json）。"""
    return _load("red_flags.json")


def incompatibilities() -> list[dict[str, Any]]:
    """配伍禁忌规则（incompatibilities.json：十八反/十九畏）。"""
    return _load("incompatibilities.json")


def glossary() -> list[dict[str, Any]]:
    """术语表（glossary.json）。"""
    return _load("glossary.json")


def pulses() -> list[dict[str, Any]]:
    """脉象判别规则库（pulses.json：28脉+平脉的量化规则与出处）。"""
    return _load("pulses.json")


def get_pulse(name: str) -> Optional[dict[str, Any]]:
    """按脉名查规则条目（支持"数脉"->"数"）。"""
    if not name:
        return None
    n = name.strip().rstrip("脉")
    return _find(pulses(), n, "name") or _find(pulses(), name.strip(), "name")


@lru_cache(maxsize=None)
def symptom_maps() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """返回 (同义词映射, 口语化映射)：术语 -> 规范症状列表。"""
    syn = {k: v for k, v in _load_map("symptom_synonyms.json").items() if isinstance(v, list)}
    col = {k: v for k, v in _load_map("colloquial_map.json").items() if isinstance(v, list)}
    return syn, col


def _load_map(filename: str) -> dict[str, Any]:
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------- 精确查找

def _find(items: list[dict[str, Any]], name: str, *keys: str) -> Optional[dict[str, Any]]:
    """先精确匹配后包含匹配的通用查找。"""
    if not name:
        return None
    keys = keys or ("name",)
    for it in items:
        if any(str(it.get(k, "")).strip() == name.strip() for k in keys):
            return it
    for it in items:
        if any(name.strip() in str(it.get(k, "")) for k in keys if it.get(k)):
            return it
    return None


def get_herb(name: str) -> Optional[dict[str, Any]]:
    return _find(herbs(), name, "name")


def get_formula(name: str) -> Optional[dict[str, Any]]:
    return _find(formulas(), name, "name")


def get_acupoint(name: str) -> Optional[dict[str, Any]]:
    return _find(acupoints(), name, "name", "code")


def get_syndrome(name: str) -> Optional[dict[str, Any]]:
    if not name:
        return None
    s = _find(syndromes(), name, "name", "id")
    if s:
        return s
    # 去掉"证"字再试（"肝气郁结" -> "肝气郁结证"）
    stripped = name.rstrip("证")
    return _find(syndromes(), stripped, "name", "id")


def get_disease(name: str) -> Optional[dict[str, Any]]:
    return _find(diseases(), name, "name")


def get_glossary_term(term: str) -> Optional[dict[str, Any]]:
    return _find(glossary(), term, "term")
