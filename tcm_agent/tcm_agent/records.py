"""问诊档案存储：每个会话持久化为一个JSON文件（默认 tcm_records/<session_id>.json）。

记录内容：患者信息（性别/年龄/怀孕）、完整对话、历次辨证报告。
用途：用户开新会话后历史不丢失，可随时回看档案。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .config import BASE_DIR

RECORDS_DIR = Path(os.environ.get("TCM_RECORDS_DIR") or (BASE_DIR / "tcm_records"))

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now() -> float:
    return time.time()


class RecordStore:
    """线程安全的档案存储（一个会话一个文件）。"""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else RECORDS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 内部

    def _path(self, sid: str) -> Path:
        if not _SAFE_ID.match(sid or ""):
            raise ValueError(f"非法会话id: {sid!r}")
        return self.root / f"{sid}.json"

    def _save(self, rec: dict[str, Any]) -> None:
        rec["updated_at"] = _now()
        with open(self._path(rec["id"]), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------ 读写

    def get(self, sid: str) -> dict[str, Any]:
        """读取档案；不存在则返回空白档案骨架。"""
        path = self._path(sid)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "id": sid,
            "created_at": _now(),
            "updated_at": _now(),
            "patient": {"sex": None, "age": None, "pregnant": False},
            "messages": [],
            "finals": [],
        }

    def update_patient(self, sid: str, sex: Optional[str] = None,
                       age: Optional[int] = None, pregnant: Optional[bool] = None) -> None:
        with self._lock:
            rec = self.get(sid)
            p = rec["patient"]
            if sex in ("男", "女"):
                p["sex"] = sex
            if age is not None:
                p["age"] = age
            if pregnant:
                p["pregnant"] = True
            self._save(rec)

    def add_message(self, sid: str, role: str, text: str,
                    meta: Optional[dict] = None) -> None:
        with self._lock:
            rec = self.get(sid)
            msg: dict[str, Any] = {"role": role, "text": text, "time": _now()}
            if meta:
                msg["meta"] = meta
            rec["messages"].append(msg)
            self._save(rec)

    def add_final(self, sid: str, final: dict[str, Any]) -> None:
        with self._lock:
            rec = self.get(sid)
            rec["finals"].append({"time": _now(), "result": final})
            self._save(rec)

    def update_evidence(self, sid: str, symptoms: list[str],
                        tongue: str = "", pulse: str = "") -> None:
        """持久化当前可信证据（用于服务重启/换端后会话还原）。"""
        with self._lock:
            rec = self.get(sid)
            rec["evidence"] = {
                "symptoms": [s for s in symptoms if s],
                "tongue": tongue or "",
                "pulse": pulse or "",
            }
            self._save(rec)

    # ------------------------------------------------------------ 列表/删除

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """档案摘要列表（按最近更新降序）。"""
        items: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                finals = rec.get("finals") or []
                last = finals[-1]["result"] if finals else None
                user_msgs = [m["text"] for m in rec.get("messages", [])
                             if m.get("role") == "user"]
                items.append({
                    "id": rec.get("id", path.stem),
                    "updated_at": rec.get("updated_at", 0),
                    "created_at": rec.get("created_at", 0),
                    "patient": rec.get("patient", {}),
                    "turns": len(rec.get("messages", [])),
                    "primary_syndrome": (last or {}).get("primary_syndrome", ""),
                    "symptoms": (last or {}).get("symptoms_canonical", [])
                    or user_msgs[:1],
                    "has_final": bool(finals),
                })
            except (json.JSONDecodeError, OSError, KeyError, IndexError):
                continue
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items[:limit]

    def all(self) -> list[dict[str, Any]]:
        """全部档案的完整内容（供画像计算）。"""
        records: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                if rec.get("messages") or rec.get("finals"):
                    records.append(rec)
            except (json.JSONDecodeError, OSError):
                continue
        records.sort(key=lambda r: r.get("updated_at", 0))
        return records

    def delete(self, sid: str) -> bool:
        path = self._path(sid)
        if path.exists():
            path.unlink()
            return True
        return False
