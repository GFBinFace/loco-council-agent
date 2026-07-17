"""查询历史持久化存储。

SQLite 表 history_sessions：一条记录 = 一次完整对话（session），
内含 messages_json 嵌套多轮 Q&A。
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from config import Config

from utils import get_file_logger
logger = get_file_logger(__file__)


class HistoryStore:
    """查询历史存储——会话级嵌套存储，SQLite 后端。"""

    def __init__(self, config: Config = Config()):
        self._db_path = os.path.join(config.sqlite_dir, "history.sqlite")
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    # ── 建表 ──────────────────────────────────────────────

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS history_sessions (
                session_id          TEXT PRIMARY KEY,
                title               TEXT NOT NULL DEFAULT '',
                messages_json       TEXT NOT NULL DEFAULT '[]',
                turn_count          INTEGER NOT NULL DEFAULT 0,
                total_input_tokens  INTEGER NOT NULL DEFAULT 0,
                total_output_tokens INTEGER NOT NULL DEFAULT 0,
                total_elapsed_ms    INTEGER NOT NULL DEFAULT 0,
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_history_updated
                ON history_sessions(updated_at DESC);
        """)
        self._conn.commit()

    # ── 写入 ──────────────────────────────────────────────

    def save_turn(self, session_id: Optional[str], turn: dict) -> str:
        """追加一轮对话到 session。session_id 为 None 时自动创建新 session。

        Args:
            session_id: 现有 session 的 ID，None 表示新建
            turn: 单轮 Q&A 数据，含 query/answer/status/sources/input_tokens/
                  output_tokens/elapsed_ms/created_at

        Returns:
            session_id（新建时返回生成的 UUID）
        """
        now = datetime.now().isoformat()
        turn.setdefault("created_at", now)

        if session_id is None:
            # 新建 session
            return self._create_session(turn, now)

        # 追加到已有 session
        existing = self._get_raw(session_id)
        if existing is None:
            # session_id 无效，降级为新建
            logger.warning(
                "session_id %s 不存在，降级为新建会话", session_id[:12],
            )
            return self._create_session(turn, now)

        return self._append_turn(session_id, turn, now)

    def _create_session(self, turn: dict, now: str) -> str:
        """创建新 session 并写入首轮对话。"""
        session_id = uuid.uuid4().hex
        title = turn.get("query", "")[:40]
        messages_json = json.dumps([turn], ensure_ascii=False)
        input_tokens = turn.get("input_tokens", 0)
        output_tokens = turn.get("output_tokens", 0)
        elapsed_ms = turn.get("elapsed_ms", 0)

        try:
            self._conn.execute(
                "INSERT INTO history_sessions "
                "(session_id, title, messages_json, turn_count, "
                "total_input_tokens, total_output_tokens, total_elapsed_ms, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (
                    session_id, title, messages_json,
                    input_tokens, output_tokens, elapsed_ms,
                    now, now,
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("创建历史会话失败")
            raise

        logger.info(
            "历史会话已创建 id=%s title=%s tokens_in=%d tokens_out=%d",
            session_id[:12], title, input_tokens, output_tokens,
        )
        return session_id

    def _append_turn(self, session_id: str, turn: dict, now: str) -> str:
        """向已有 session 追加一轮对话。"""
        row = self._conn.execute(
            "SELECT messages_json FROM history_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        messages = json.loads(row["messages_json"])
        messages.append(turn)
        messages_json = json.dumps(messages, ensure_ascii=False)

        input_tokens = turn.get("input_tokens", 0)
        output_tokens = turn.get("output_tokens", 0)
        elapsed_ms = turn.get("elapsed_ms", 0)

        try:
            self._conn.execute(
                "UPDATE history_sessions SET "
                "messages_json = ?, "
                "turn_count = turn_count + 1, "
                "total_input_tokens = total_input_tokens + ?, "
                "total_output_tokens = total_output_tokens + ?, "
                "total_elapsed_ms = total_elapsed_ms + ?, "
                "updated_at = ? "
                "WHERE session_id = ?",
                (
                    messages_json,
                    input_tokens, output_tokens, elapsed_ms,
                    now, session_id,
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("追加历史记录失败: %s", session_id[:12])
            raise

        logger.info(
            "历史记录已追加 id=%s turn=%d tokens_in=%d tokens_out=%d",
            session_id[:12], len(messages), input_tokens, output_tokens,
        )
        return session_id

    # ── 查询 ──────────────────────────────────────────────

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取单次完整对话。不存在时返回 None。

        messages_json 已自动反序列化为 messages 列表。
        """
        row = self._get_raw(session_id)
        if row is None:
            return None
        result = dict(row)
        try:
            result["messages"] = json.loads(result.get("messages_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            result["messages"] = []
        return result

    def _get_raw(self, session_id: str):
        """获取原始行（不反序列化 JSON），供内部使用。"""
        return self._conn.execute(
            "SELECT * FROM history_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    def list_sessions(self, limit: int = 30) -> List[Dict]:
        """获取最近 N 次对话（按 updated_at 倒序）。

        每条返回 session_id / title / turn_count / total_* / created_at / updated_at。
        messages_json 不在列表中（UI 仅在点击恢复时加载完整内容）。
        """
        rows = self._conn.execute(
            "SELECT session_id, title, turn_count, "
            "total_input_tokens, total_output_tokens, total_elapsed_ms, "
            "created_at, updated_at "
            "FROM history_sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 删除 ──────────────────────────────────────────────

    def delete_session(self, session_id: str) -> bool:
        """删除整个对话 session。返回是否成功删除。"""
        try:
            cursor = self._conn.execute(
                "DELETE FROM history_sessions WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("历史会话已删除 id=%s", session_id[:12])
            return deleted
        except sqlite3.Error:
            logger.exception("删除历史会话失败 id=%s", session_id[:12])
            raise

    def close(self) -> None:
        """关闭数据库连接。"""
        # ⚠️ NOTICE: 当前 HistoryStore 作为 @st.cache_resource 单例全程存活至进程退出，
        # 操作系统会自动回收 SQLite 连接，暂无调用方。如果未来增加了销毁/重建 HistoryStore
        # 实例的业务逻辑（如动态切换数据源），必须在实例销毁前调用 close()。
        if self._conn:
            self._conn.close()
            self._conn = None
