"""测试 SearchController — 搜索管线编排与历史存储协调。"""

from unittest.mock import MagicMock, patch

import pytest

from _types.retrieval_types import ChunkCandidate, ContinueChoice, SearchResult


# ── 辅助函数 ──────────────────────────────────────────────


def _make_chunk(
    chunk_id: str = "doc_chunk_0",
    text: str = "测试文本",
    doc_id: str = "abc123",
    chunk_index: int = 0,
    llm_score: int = 8,
) -> ChunkCandidate:
    """构造测试用 ChunkCandidate。"""
    return ChunkCandidate(
        id=chunk_id, text=text, doc_id=doc_id, doc_name="test.pdf",
        page_nums=[1], chunk_index=chunk_index, length=len(text),
        type="mixed", has_financial_keywords=[],
        hybrid_score=0.85, llm_score=llm_score,
    )


def _make_success_result(**overrides) -> SearchResult:
    """构造 status="success" 的 SearchResult。"""
    defaults = {
        "status": "success",
        "query": "测试查询",
        "answer": "根据文档，测试回答。",
        "sources": [_make_chunk()],
        "source_count": 1,
        "token_usage": {"input": 500, "output": 200},
    }
    defaults.update(overrides)
    return SearchResult(**defaults)


def _make_needs_choice_result(
    pending_decision: str = "zero_results",
    top_score: int = None,
    **overrides,
) -> SearchResult:
    """构造 status="needs_user_choice" 的 SearchResult。"""
    defaults = {
        "status": "needs_user_choice",
        "query": "测试查询",
        "pending_decision": pending_decision,
        "token_usage": {"input": 300, "output": 0},
    }
    if top_score is not None:
        defaults["top_score"] = top_score
    defaults.update(overrides)
    return SearchResult(**defaults)


@pytest.fixture
def mock_pipeline():
    """构造 mock pipeline。"""
    mock = MagicMock()
    mock.search.return_value = _make_success_result()
    mock.continue_search.return_value = _make_success_result()
    return mock


@pytest.fixture
def mock_history_store():
    """构造 mock HistoryStore。"""
    mock = MagicMock()
    mock.save_turn.return_value = "new_session_id_1234567890abcdef"
    mock.get_session.return_value = {
        "session_id": "sess_001",
        "title": "测试对话",
        "turn_count": 1,
        "messages": [
            {
                "query": "测试查询",
                "answer": "测试回答",
                "status": "success",
                "sources": [],
                "input_tokens": 500,
                "output_tokens": 200,
                "elapsed_ms": 3000,
                "created_at": "2026-07-07T12:00:00+00:00",
            }
        ],
    }
    mock.list_sessions.return_value = [
        {
            "session_id": "sess_001",
            "title": "测试对话",
            "turn_count": 1,
            "total_input_tokens": 500,
            "total_output_tokens": 200,
            "total_elapsed_ms": 3000,
            "created_at": "2026-07-07T12:00:00+00:00",
            "updated_at": "2026-07-07T12:00:00+00:00",
        }
    ]
    mock.delete_session.return_value = True
    return mock


@pytest.fixture
def controller(mock_pipeline, mock_history_store):
    """注入 mock pipeline 和 HistoryStore 的 SearchController。"""
    from controllers.search_controller import SearchController
    yield SearchController(mock_pipeline, mock_history_store)


# ── execute_search ───────────────────────────────────────


class TestExecuteSearch:
    """SearchController.execute_search()。"""

    def test_success_saves_history(self, controller, mock_pipeline, mock_history_store):
        """success 结果自动保存到 HistoryStore。"""
        result, sid = controller.execute_search("测试查询", None)
        assert result.status == "success"
        mock_pipeline.search.assert_called_once()
        # 保存时 session_id 为 None，由 HistoryStore 生成新 ID
        mock_history_store.save_turn.assert_called_once()
        call_kwargs = mock_history_store.save_turn.call_args
        assert call_kwargs[0][0] is None  # session_id = None，新建
        assert call_kwargs[0][1]["query"] == "测试查询"
        assert call_kwargs[0][1]["status"] == "success"
        assert sid == "new_session_id_1234567890abcdef"

    def test_success_returns_new_session_id(self, controller):
        """新建 session 时返回 HistoryStore 生成的 session_id。"""
        _, sid = controller.execute_search("测试查询", None)
        assert sid == "new_session_id_1234567890abcdef"

    def test_success_with_existing_session_id_preserves_it(self, controller, mock_history_store):
        """已有 session_id 时回传原值。"""
        existing_id = "existing_session_abcd12345678"
        mock_history_store.save_turn.return_value = existing_id
        _, sid = controller.execute_search("追问", existing_id)
        mock_history_store.save_turn.assert_called_once()
        call_kwargs = mock_history_store.save_turn.call_args
        assert call_kwargs[0][0] == existing_id
        assert sid == existing_id

    def test_needs_user_choice_does_not_save_history(
        self, controller, mock_pipeline, mock_history_store,
    ):
        """needs_user_choice 结果不保存到 HistoryStore。"""
        mock_pipeline.search.return_value = _make_needs_choice_result()
        result, sid = controller.execute_search("测试查询", None)
        assert result.status == "needs_user_choice"
        # 不应调用 save_turn
        mock_history_store.save_turn.assert_not_called()

    def test_needs_user_choice_returns_session_id_none(self, controller, mock_pipeline):
        """needs_user_choice 时 session_id 为 None（首次搜索未创建记录）。"""
        mock_pipeline.search.return_value = _make_needs_choice_result()
        _, sid = controller.execute_search("测试查询", None)
        assert sid is None

    def test_pipeline_exception_saves_error_history(
        self, controller, mock_pipeline, mock_history_store,
    ):
        """pipeline 抛出异常时保存 status="error" 的历史。"""
        mock_pipeline.search.side_effect = RuntimeError("LLM 超时")
        result, sid = controller.execute_search("测试查询", None)
        assert result.status == "error"
        assert "LLM 超时" in result.error_message
        mock_history_store.save_turn.assert_called_once()
        call_kwargs = mock_history_store.save_turn.call_args
        assert call_kwargs[0][1]["status"] == "error"
        assert call_kwargs[0][1]["query"] == "测试查询"

    def test_history_context_passed_to_pipeline(self, controller, mock_pipeline):
        """history_context 透传给 pipeline.search()。"""
        ctx = [{"query": "上一轮", "answer": "上一回答", "sources": []}]
        controller.execute_search("当前查询", "sid_001", history_context=ctx)
        _, kwargs = mock_pipeline.search.call_args
        assert kwargs["history_context"] == ctx

    def test_on_progress_passed_to_pipeline(self, controller, mock_pipeline):
        """on_progress 回调透传给 pipeline.search()。"""
        cb = MagicMock()
        controller.execute_search("查询", None, on_progress=cb)
        _, kwargs = mock_pipeline.search.call_args
        assert kwargs["on_progress"] == cb

    def test_sources_serialized_for_history(self, controller, mock_history_store):
        """ChunkCandidate 列表被正确序列化存储。"""
        controller.execute_search("查询", None)
        call_kwargs = mock_history_store.save_turn.call_args
        sources = call_kwargs[0][1]["sources"]
        assert len(sources) == 1
        assert sources[0]["doc_name"] == "test.pdf"
        assert sources[0]["page_nums"] == [1]
        assert sources[0]["chunk_index"] == 0
        assert sources[0]["llm_score"] == 8


# ── execute_continue ─────────────────────────────────────


class TestExecuteContinue:
    """SearchController.execute_continue()。"""

    def test_abandon_saves_history(self, controller, mock_pipeline, mock_history_store):
        """ABANDON 选择保存 status="abandoned" 的历史。"""
        result, sid = controller.execute_continue(
            ContinueChoice.ABANDON, "sid_001",
        )
        mock_pipeline.continue_search.assert_called_once_with(
            ContinueChoice.ABANDON,
            history_context=None,
            on_progress=None,
        )
        mock_history_store.save_turn.assert_called_once()
        call_kwargs = mock_history_store.save_turn.call_args
        assert call_kwargs[0][0] == "sid_001"
        assert call_kwargs[0][1]["status"] == "abandoned"

    def test_direct_llm_saves_history(self, controller, mock_history_store):
        """DIRECT_LLM 选择保存 status="direct_llm" 的历史。"""
        controller.execute_continue(ContinueChoice.DIRECT_LLM, "sid_001")
        call_kwargs = mock_history_store.save_turn.call_args
        assert call_kwargs[0][1]["status"] == "direct_llm"

    def test_rag_saves_history(self, controller, mock_history_store):
        """RAG 选择保存 status="low_confidence_rag" 的历史。"""
        controller.execute_continue(ContinueChoice.RAG, "sid_001")
        call_kwargs = mock_history_store.save_turn.call_args
        assert call_kwargs[0][1]["status"] == "low_confidence_rag"

    def test_pipeline_exception_saves_error_history(
        self, controller, mock_pipeline, mock_history_store,
    ):
        """continue_search 抛出异常时保存 error 历史，状态跟随 choice。"""
        mock_pipeline.continue_search.side_effect = RuntimeError("生成失败")
        result, sid = controller.execute_continue(
            ContinueChoice.RAG, "sid_001",
        )
        assert result.status == "error"
        mock_history_store.save_turn.assert_called_once()
        call_kwargs = mock_history_store.save_turn.call_args
        assert call_kwargs[0][1]["status"] == "low_confidence_rag"

    def test_history_context_passed_to_pipeline(self, controller, mock_pipeline):
        """history_context 透传给 pipeline.continue_search()。"""
        ctx = [{"query": "上一轮", "answer": "上一回答", "sources": []}]
        controller.execute_continue(
            ContinueChoice.RAG, "sid_001", history_context=ctx,
        )
        _, kwargs = mock_pipeline.continue_search.call_args
        assert kwargs["history_context"] == ctx

    def test_on_progress_passed_to_pipeline(self, controller, mock_pipeline):
        """on_progress 回调透传给 pipeline.continue_search()。"""
        cb = MagicMock()
        controller.execute_continue(
            ContinueChoice.DIRECT_LLM, "sid_001", on_progress=cb,
        )
        _, kwargs = mock_pipeline.continue_search.call_args
        assert kwargs["on_progress"] == cb


# ── 历史存储转发 ─────────────────────────────────────────


class TestHistoryDelegation:
    """SearchController 历史管理方法正确转发给 HistoryStore。"""

    def test_restore_session_delegates(self, controller, mock_history_store):
        """restore_session 转发给 HistoryStore.get_session()。"""
        session = controller.restore_session("sess_001")
        mock_history_store.get_session.assert_called_once_with("sess_001")
        assert session is not None
        assert session["session_id"] == "sess_001"

    def test_restore_session_nonexistent_returns_none(
        self, controller, mock_history_store,
    ):
        """不存在的 session 返回 None。"""
        mock_history_store.get_session.return_value = None
        session = controller.restore_session("nonexistent")
        assert session is None

    def test_list_sessions_delegates(self, controller, mock_history_store):
        """list_sessions 转发给 HistoryStore.list_sessions()。"""
        result = controller.list_sessions(limit=10)
        mock_history_store.list_sessions.assert_called_once_with(10)
        assert len(result) == 1
        assert result[0]["session_id"] == "sess_001"

    def test_delete_session_delegates(self, controller, mock_history_store):
        """delete_session 转发给 HistoryStore.delete_session()。"""
        result = controller.delete_session("sess_001")
        mock_history_store.delete_session.assert_called_once_with("sess_001")
        assert result is True
