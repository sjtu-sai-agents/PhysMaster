"""
Feishu Bot — FastAPI webhook server

Receives messages from Feishu, immediately replies "正在求解…",
then runs the PhysMaster pipeline in a background thread and pushes
the result back via the Feishu Open API.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict

import requests
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from worker import solve

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("feishu-bot")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_feishu_config() -> Dict[str, Any]:
    if not _CFG_PATH.exists():
        raise FileNotFoundError(
            f"Feishu config not found at {_CFG_PATH}. "
            "Copy config_example.yaml → config.yaml and fill in your credentials."
        )
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_CFG = _load_feishu_config()
FEISHU_CFG = _CFG["feishu"]
SERVER_CFG = _CFG.get("server", {})
PHYSMASTER_CFG = _CFG.get("physmaster", {})

APP_ID: str = FEISHU_CFG["app_id"]
APP_SECRET: str = FEISHU_CFG["app_secret"]
VERIFICATION_TOKEN: str = FEISHU_CFG.get("verification_token", "")
ENCRYPT_KEY: str = FEISHU_CFG.get("encrypt_key", "")

MAX_WORKERS: int = SERVER_CFG.get("max_workers", 2)
PM_CONFIG_PATH: str = PHYSMASTER_CFG.get("config_path", "config.yaml")

FEISHU_HOST = "https://open.feishu.cn"

# Maximum text length Feishu allows in a single message (approx 30 KB for
# rich-text; we use a conservative limit for plain text content blocks).
MAX_MSG_LEN = 25_000

# ---------------------------------------------------------------------------
# Tenant access-token cache
# ---------------------------------------------------------------------------
_token_cache: Dict[str, Any] = {"token": "", "expires_at": 0.0}
_token_lock = Lock()


def _get_tenant_access_token() -> str:
    """Return a valid tenant_access_token, refreshing if necessary."""
    now = time.time()
    with _token_lock:
        if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
            return _token_cache["token"]

    url = f"{FEISHU_HOST}/open-apis/auth/v3/tenant_access_token/internal"
    resp = requests.post(url, json={"app_id": APP_ID, "app_secret": APP_SECRET}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Failed to get tenant_access_token: {data}")

    token = data["tenant_access_token"]
    expire = data.get("expire", 7200)

    with _token_lock:
        _token_cache["token"] = token
        _token_cache["expires_at"] = time.time() + expire

    return token


def _feishu_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_tenant_access_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }


# ---------------------------------------------------------------------------
# Feishu messaging helpers
# ---------------------------------------------------------------------------

def _reply_text(message_id: str, text: str) -> None:
    """Reply to a specific message with plain text."""
    url = f"{FEISHU_HOST}/open-apis/im/v1/messages/{message_id}/reply"
    body = {
        "content": json.dumps({"text": text}),
        "msg_type": "text",
    }
    resp = requests.post(url, headers=_feishu_headers(), json=body, timeout=10)
    if resp.status_code != 200 or resp.json().get("code") != 0:
        logger.error("reply failed: %s", resp.text)


def _send_text(chat_id: str, text: str) -> None:
    """Proactively send a text message to a chat."""
    url = f"{FEISHU_HOST}/open-apis/im/v1/messages"
    body = {
        "receive_id": chat_id,
        "content": json.dumps({"text": text}),
        "msg_type": "text",
    }
    params = {"receive_id_type": "chat_id"}
    resp = requests.post(url, headers=_feishu_headers(), json=body, params=params, timeout=10)
    if resp.status_code != 200 or resp.json().get("code") != 0:
        logger.error("send failed: %s", resp.text)


def _truncate(text: str, limit: int = MAX_MSG_LEN) -> str:
    """Truncate text to *limit* characters, appending a notice if trimmed."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n…（结果过长，已截断。完整内容请查看服务器上的 summary.md 文件）"


# ---------------------------------------------------------------------------
# Message de-duplication (LRU cache of recent message IDs)
# ---------------------------------------------------------------------------
_DEDUP_MAX = 512
_seen_ids: OrderedDict[str, float] = OrderedDict()
_seen_lock = Lock()


def _is_duplicate(message_id: str) -> bool:
    with _seen_lock:
        if message_id in _seen_ids:
            return True
        _seen_ids[message_id] = time.time()
        while len(_seen_ids) > _DEDUP_MAX:
            _seen_ids.popitem(last=False)
        return False


# ---------------------------------------------------------------------------
# Background solver
# ---------------------------------------------------------------------------
_executor: ThreadPoolExecutor | None = None


def _background_solve(message_id: str, chat_id: str, query_text: str) -> None:
    """Run in a thread-pool thread.  Pushes result back to Feishu."""
    try:
        logger.info("Start solving for message %s", message_id)
        result = solve(query_text, config_path=PM_CONFIG_PATH)
        summary = result.get("summary", "（未生成 summary）")
        task_name = result.get("task_name", "unknown")
        header = f"[{task_name}] 求解完成：\n\n"
        _send_text(chat_id, _truncate(header + summary))
        logger.info("Result sent for message %s (task=%s)", message_id, task_name)
    except Exception:
        logger.exception("Solve failed for message %s", message_id)
        _send_text(chat_id, "求解过程出错，请查看服务端日志。")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor
    _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    logger.info("Bot started  (max_workers=%d)", MAX_WORKERS)
    yield
    _executor.shutdown(wait=False)
    logger.info("Bot stopped")


app = FastAPI(title="PhysMaster Feishu Bot", lifespan=lifespan)


@app.post("/webhook/event")
async def handle_event(request: Request) -> JSONResponse:
    body: Dict[str, Any] = await request.json()

    # --- URL verification challenge ---
    if body.get("type") == "url_verification":
        return JSONResponse({"challenge": body.get("challenge", "")})

    # --- v2 event schema ---
    schema_ver = body.get("schema")
    header = body.get("header", {})
    event = body.get("event", {})

    # Token verification (v2 schema)
    if schema_ver == "2.0":
        token = header.get("token", "")
        if VERIFICATION_TOKEN and token != VERIFICATION_TOKEN:
            logger.warning("Token mismatch, ignoring event")
            return JSONResponse({"code": 0})

    # Only handle message events
    event_type = header.get("event_type", "") or body.get("event", {}).get("type", "")
    if event_type != "im.message.receive_v1":
        return JSONResponse({"code": 0})

    message = event.get("message", {})
    message_id = message.get("message_id", "")
    chat_id = message.get("chat_id", "")
    msg_type = message.get("message_type", "")

    # We only handle plain-text messages
    if msg_type != "text":
        _reply_text(message_id, "目前仅支持文本消息，请直接发送物理题文字。")
        return JSONResponse({"code": 0})

    # Extract text content
    try:
        content = json.loads(message.get("content", "{}"))
        query_text: str = content.get("text", "").strip()
    except (json.JSONDecodeError, AttributeError):
        query_text = ""

    if not query_text:
        return JSONResponse({"code": 0})

    # Strip @mentions (Feishu wraps them as @_user_X)
    import re
    query_text = re.sub(r"@_user_\d+\s*", "", query_text).strip()
    if not query_text:
        return JSONResponse({"code": 0})

    # De-duplicate (Feishu may retry the callback)
    if _is_duplicate(message_id):
        logger.info("Duplicate message %s, skipping", message_id)
        return JSONResponse({"code": 0})

    # Acknowledge immediately
    _reply_text(message_id, "收到！正在求解，请稍候（通常需要 5-30 分钟）…")

    # Submit to background worker
    _executor.submit(_background_solve, message_id, chat_id, query_text)

    return JSONResponse({"code": 0})


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    host = SERVER_CFG.get("host", "0.0.0.0")
    port = SERVER_CFG.get("port", 9000)
    logger.info("Starting Feishu bot on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
