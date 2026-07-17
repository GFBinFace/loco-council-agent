"""Streamlit 会话状态管理。

集中定义所有 session_state 键名和初始化逻辑，
避免散落各处的字符串字面量。
"""

from collections import deque

import streamlit as st


# ── 状态键常量 ──────────────────────────────────────────

class Keys:
    """session_state 键名常量。"""
    # ── 通用 ──
    CURRENT_SESSION_ID = "current_session_id"      # str | None
    SESSION_TOKENS = "session_tokens"               # dict

    # ── 搜索面板 ──
    SEARCH_STATE = "search_state"                   # "idle"|"searching"|"needs_choice"|"done"
    SEARCH_RESULT = "search_result"                 # SearchResult | None
    SEARCH_START_TIME = "search_start_time"         # float | None
    PENDING_QUERY = "pending_query"                 # str
    CHAT_MESSAGES = "chat_messages"                 # list[dict]
    HISTORY_ENABLED = "history_enabled"             # bool
    DO_SEARCH = "do_search"                         # bool，触发搜索
    DO_CONTINUE = "do_continue"                     # bool，触发继续搜索
    PENDING_CHOICE = "pending_choice"               # ContinueChoice | None
    SEARCH_STATUS_TEXT = "search_status_text"       # str
    PROGRESS_LOG_SEARCH = "progress_log_search"     # deque[str]

    # ── 文档面板 ──
    PROGRESS_LOG_DOC = "progress_log_doc"           # deque[str]
    PDF_PATH_INPUT = "pdf_path_input"               # str，上传路径输入框
    PDF_PICKER = "pdf_picker"                       # UploadedFile，文件选择器

    # ── 操作历史 ──
    OP_FROM_ID = "op_from_id"                       # int
    OP_TO_ID = "op_to_id"                           # int
    OP_RANGE_ERROR = "op_range_error"               # str，范围校验错误信息（空串=无错误）


# ── 初始化 ──────────────────────────────────────────────

def init_session_state() -> None:
    """初始化所有 session_state 默认值（按需，不覆盖已有值）。"""
    defaults = {
        Keys.CURRENT_SESSION_ID: None,
        Keys.SESSION_TOKENS: {"input": 0, "output": 0, "index_input": 0, "index_output": 0},
        Keys.SEARCH_STATE: "idle",
        Keys.SEARCH_RESULT: None,
        Keys.SEARCH_START_TIME: None,
        Keys.PENDING_QUERY: "",
        Keys.CHAT_MESSAGES: [],
        Keys.HISTORY_ENABLED: False,
        Keys.SEARCH_STATUS_TEXT: "✅ 就绪，输入查询开始检索",
        Keys.PROGRESS_LOG_SEARCH: deque(maxlen=1000),
        Keys.PROGRESS_LOG_DOC: deque(maxlen=1000),
        Keys.PDF_PATH_INPUT: "",
        Keys.OP_FROM_ID: 1,
        Keys.OP_TO_ID: 1,
        Keys.OP_RANGE_ERROR: "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ── 辅助函数 ──────────────────────────────────────────────

def add_chat_message(query: str, answer: str, sources: list, token_usage: dict | None) -> None:
    """向当前对话的 chat_messages 追加一条 Q&A 记录。"""
    import time
    entry = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "token_usage": token_usage or {"input": 0, "output": 0},
        "timestamp": time.time(),
    }
    st.session_state[Keys.CHAT_MESSAGES].append(entry)


def accumulate_session_tokens(token_usage: dict | None, category: str = "search") -> None:
    """累加 token 到浏览器会话。category: 'search' | 'index'。"""
    if not token_usage:
        return
    tokens = st.session_state[Keys.SESSION_TOKENS]
    if category == "index":
        tokens["index_input"] += token_usage.get("input", 0)
        tokens["index_output"] += token_usage.get("output", 0)
    else:
        tokens["input"] += token_usage.get("input", 0)
        tokens["output"] += token_usage.get("output", 0)


def add_progress_log(message: str, key: str = Keys.PROGRESS_LOG_SEARCH) -> None:
    """向指定进度日志区追加一条消息（自动附带时间戳）。

    日志仅存于前端内存（session_state 的 deque），
    浏览器关闭即丢弃。容量上限 1000 条，超出时自动丢弃最旧记录。
    详细的运行日志由后端 logging 模块持久化到磁盘。

    Args:
        message: 日志消息文本
        key: 日志区键名，默认搜索区
    """
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state[key].append(f"[{ts}] {message}")
