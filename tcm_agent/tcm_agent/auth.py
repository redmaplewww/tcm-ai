"""账号系统：注册/登录（pbkdf2密码哈希 + HMAC签名令牌）。

存储：
- tcm_data/accounts.json  {username: {salt, hash, nickname, created}}
- tcm_data/secret.key     令牌签名密钥（首次自动生成）

令牌：base64(username|过期时间戳):hmac签名，30天有效，无状态可跨重启。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from pathlib import Path
from typing import Optional

from .config import BASE_DIR

DATA_DIR = Path(BASE_DIR / "tcm_data")
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
SECRET_FILE = DATA_DIR / "secret.key"

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5]{2,20}$")
_TOKEN_DAYS = 30


class AuthError(Exception):
    """账号操作错误（用户名非法/已存在/密码错误等）。"""


def _load_accounts() -> dict:
    if ACCOUNTS_FILE.exists():
        try:
            with open(ACCOUNTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_accounts(accounts: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(accounts, f, ensure_ascii=False, indent=1)


def _secret() -> bytes:
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes().strip()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(32).encode()
    SECRET_FILE.write_bytes(secret)
    return secret


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                               salt.encode("utf-8"), 120_000).hex()


def register(username: str, password: str, nickname: str = "") -> dict:
    """注册新账号，返回账号信息。"""
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        raise AuthError("用户名需2-20位，可含中文、字母、数字、下划线")
    if len(password or "") < 6:
        raise AuthError("密码至少6位")
    accounts = _load_accounts()
    if username in accounts:
        raise AuthError("用户名已存在")
    salt = secrets.token_hex(8)
    accounts[username] = {
        "salt": salt,
        "hash": _hash_password(password, salt),
        "nickname": (nickname or "").strip() or username,
        "created": time.time(),
    }
    _save_accounts(accounts)
    return {"username": username, "nickname": accounts[username]["nickname"]}


def verify(username: str, password: str) -> bool:
    accounts = _load_accounts()
    info = accounts.get((username or "").strip())
    if not info:
        return False
    return hmac.compare_digest(info["hash"], _hash_password(password, info["salt"]))


def get_account(username: str) -> Optional[dict]:
    return _load_accounts().get((username or "").strip())


def issue_token(username: str, days: int = _TOKEN_DAYS) -> str:
    exp = int(time.time()) + days * 86400
    payload = f"{username}|{exp}".encode("utf-8")
    sig = hmac.new(_secret(), payload, hashlib.sha256).hexdigest()[:32]
    return base64.urlsafe_b64encode(payload).decode() + ":" + sig


def parse_token(token: str) -> Optional[str]:
    """校验令牌，有效返回用户名，否则None。"""
    try:
        b64, sig = (token or "").rsplit(":", 1)
        payload = base64.urlsafe_b64decode(b64.encode()).decode("utf-8")
        username, exp_str = payload.split("|")
        expect = hmac.new(_secret(), payload.encode("utf-8"),
                          hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, expect):
            return None
        if int(exp_str) < time.time():
            return None  # 过期
        if get_account(username) is None:
            return None
        return username
    except (ValueError, IndexError):
        return None
