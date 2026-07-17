"""测试 HistoryStore — 会话级嵌套存储（save_turn / get_session / list_sessions / delete_session）。"""

import tempfile

import pytest

from config import Config


@pytest.fixture
def hs():
    """构造临时 SQLite 目录的 HistoryStore 实例。"""
    cfg = Config()
    with tempfile.TemporaryDirectory() as td:
        cfg.sqlite_dir = td
        from storage.history_store import HistoryStore
        store = HistoryStore(cfg)
        yield store
        store.close()


def _make_turn(query="测试查询", answer="测试回答", **kwargs):
    """构造单轮 turn 数据。"""
    turn = {
        "query": query,
        "answer": answer,
        "status": "success",
        "sources": [{"doc_name": "test.pdf", "page_nums": [1]}],
        "input_tokens": 100,
        "output_tokens": 50,
        "elapsed_ms": 2000,
    }
    turn.update(kwargs)
    return turn


class TestSaveTurn:
    """测试 save_turn 创建和追加。"""

    def test_create_new_session_returns_session_id(self, hs):
        """save_turn(None) 创建新 session 并返回 UUID。"""
        sid = hs.save_turn(None, _make_turn(query="新对话"))
        assert sid is not None
        assert len(sid) == 32  # uuid4().hex

    def test_append_to_existing_session(self, hs):
        """save_turn(existing_id) 追加到已有 session。"""
        sid = hs.save_turn(None, _make_turn(query="第1轮"))
        hs.save_turn(sid, _make_turn(query="第2轮", input_tokens=50))
        session = hs.get_session(sid)
        assert session["turn_count"] == 2
        assert len(session["messages"]) == 2
        assert session["messages"][0]["query"] == "第1轮"
        assert session["messages"][1]["query"] == "第2轮"

    def test_append_aggregates_tokens(self, hs):
        """追加时正确累加 session 级 token 和耗时。"""
        sid = hs.save_turn(None, _make_turn(input_tokens=100, output_tokens=50, elapsed_ms=1000))
        hs.save_turn(sid, _make_turn(input_tokens=200, output_tokens=80, elapsed_ms=1500))
        session = hs.get_session(sid)
        assert session["total_input_tokens"] == 300
        assert session["total_output_tokens"] == 130
        assert session["total_elapsed_ms"] == 2500

    def test_nonexistent_session_id_falls_back_to_create(self, hs):
        """无效 session_id 降级为新建。"""
        sid = hs.save_turn("nonexistent_id_1234567890abcdef", _make_turn())
        assert sid != "nonexistent_id_1234567890abcdef"
        session = hs.get_session(sid)
        assert session is not None
        assert session["turn_count"] == 1

    def test_title_extracted_from_first_query(self, hs):
        """title 取首条 query 前 40 字符。"""
        long_query = "A" * 50
        sid = hs.save_turn(None, _make_turn(query=long_query))
        session = hs.get_session(sid)
        assert session["title"] == "A" * 40
        assert len(session["title"]) == 40

    def test_title_not_updated_on_append(self, hs):
        """追加时不更新 title。"""
        sid = hs.save_turn(None, _make_turn(query="第一个问题"))
        hs.save_turn(sid, _make_turn(query="第二个问题"))
        session = hs.get_session(sid)
        assert session["title"] == "第一个问题"


class TestGetSession:
    """测试 get_session。"""

    def test_get_returns_full_session(self, hs):
        """get_session 返回完整 session 含反序列化的 messages。"""
        sid = hs.save_turn(None, _make_turn(query="Q1", answer="A1"))
        session = hs.get_session(sid)
        assert session is not None
        assert session["session_id"] == sid
        assert session["turn_count"] == 1
        assert len(session["messages"]) == 1
        assert session["messages"][0]["query"] == "Q1"
        assert session["messages"][0]["sources"] == [{"doc_name": "test.pdf", "page_nums": [1]}]

    def test_get_nonexistent_returns_none(self, hs):
        """不存在的 session 返回 None。"""
        assert hs.get_session("nonexistent_1234567890abcdef") is None

    def test_get_handles_malformed_json(self, hs):
        """messages_json 损坏时退回空列表。"""
        sid = hs.save_turn(None, _make_turn())
        # 直接破坏 JSON
        hs._conn.execute(
            "UPDATE history_sessions SET messages_json = 'not-json' WHERE session_id = ?",
            (sid,),
        )
        hs._conn.commit()
        session = hs.get_session(sid)
        assert session["messages"] == []


class TestListSessions:
    """测试 list_sessions。"""

    def test_list_returns_recent_first(self, hs):
        """按 updated_at 倒序排列。"""
        sid1 = hs.save_turn(None, _make_turn(query="旧"))
        sid2 = hs.save_turn(None, _make_turn(query="新"))
        sessions = hs.list_sessions(limit=30)
        # 最新的在前
        assert sessions[0]["session_id"] == sid2

    def test_list_respects_limit(self, hs):
        """limit 参数生效。"""
        for i in range(5):
            hs.save_turn(None, _make_turn(query=f"Q{i}"))
        assert len(hs.list_sessions(limit=2)) == 2

    def test_list_does_not_include_messages(self, hs):
        """list_sessions 不返回 messages_json（性能优化）。"""
        hs.save_turn(None, _make_turn())
        sessions = hs.list_sessions()
        assert "messages_json" not in sessions[0]
        assert "messages" not in sessions[0]

    def test_list_empty_returns_empty(self, hs):
        """无记录时返回空列表。"""
        assert hs.list_sessions() == []


class TestDeleteSession:
    """测试 delete_session。"""

    def test_delete_removes_session(self, hs):
        """delete_session 后 get_session 返回 None。"""
        sid = hs.save_turn(None, _make_turn())
        assert hs.delete_session(sid) is True
        assert hs.get_session(sid) is None

    def test_delete_nonexistent_returns_false(self, hs):
        """删除不存在记录返回 False。"""
        assert hs.delete_session("nonexistent_1234567890abcdef") is False
