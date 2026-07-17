"""测试 retrieval_types 数据契约。"""

import pytest
from _types.retrieval_types import ChunkCandidate, SearchResult


class TestChunkCandidate:
    """ChunkCandidate dataclass 构造与默认值"""

    def test_chunk_candidate_minimal_construction(self):
        """最小字段构造，可选 score 字段使用默认值"""
        c = ChunkCandidate(
            id="test_chunk_0",
            text="测试内容",
            doc_id="abc123",
            doc_name="test.pdf",
            page_nums=[1, 2],
            chunk_index=0,
            length=4,
            type="mixed",
            has_financial_keywords=["净利润"],
        )
        assert c.id == "test_chunk_0"
        assert c.text == "测试内容"
        assert c.page_nums == [1, 2]
        assert c.hybrid_score == 0.0
        assert c.cross_encoder_score is None
        assert c.llm_score is None
        assert c.llm_reason is None

    def test_chunk_candidate_full_construction(self):
        """全字段构造，各阶段 score 字段可显式赋值"""
        c = ChunkCandidate(
            id="x",
            text="y",
            doc_id="d",
            doc_name="n",
            page_nums=[3],
            chunk_index=5,
            length=10,
            type="table",
            has_financial_keywords=[],
            hybrid_score=0.85,
            cross_encoder_score=0.92,
            llm_score=8,
            llm_reason="高度相关",
        )
        assert c.hybrid_score == 0.85
        assert c.cross_encoder_score == 0.92
        assert c.llm_score == 8
        assert c.llm_reason == "高度相关"

    def test_chunk_candidate_score_fields_mutable(self):
        """各阶段可逐步写入 score 字段"""
        c = ChunkCandidate(
            id="c0", text="t", doc_id="d", doc_name="n",
            page_nums=[1], chunk_index=0, length=2,
            type="text", has_financial_keywords=[],
        )
        # 混合检索写入
        c.hybrid_score = 0.75
        # CrossEncoder二次排序写入
        c.cross_encoder_score = 0.88
        # LLM打分写入
        c.llm_score = 7
        c.llm_reason = "匹配"

        assert c.hybrid_score == 0.75
        assert c.cross_encoder_score == 0.88
        assert c.llm_score == 7


class TestSearchResult:
    """SearchResult dataclass 三种 status"""

    def test_search_result_success(self):
        """status=success 时 answer 和 sources 有值"""
        c = ChunkCandidate(
            id="c0", text="t", doc_id="d", doc_name="n",
            page_nums=[1], chunk_index=0, length=2,
            type="text", has_financial_keywords=[],
        )
        r = SearchResult(
            status="success",
            query="测试查询",
            answer="测试答案",
            sources=[c],
            source_count=1,
        )
        assert r.status == "success"
        assert r.answer == "测试答案"
        assert len(r.sources) == 1

    def test_search_result_needs_user_choice_zero_results(self):
        """status=needs_user_choice 时 pending_decision 指示场景"""
        r = SearchResult(
            status="needs_user_choice",
            query="q",
            pending_decision="zero_results",
        )
        assert r.status == "needs_user_choice"
        assert r.pending_decision == "zero_results"
        assert r.answer is None

    def test_search_result_needs_user_choice_low_confidence(self):
        """低置信度场景带 top_score"""
        r = SearchResult(
            status="needs_user_choice",
            query="q",
            pending_decision="low_confidence",
            top_score=4,
        )
        assert r.pending_decision == "low_confidence"
        assert r.top_score == 4

    def test_search_result_error(self):
        """status=error 时 error_message 有值"""
        r = SearchResult(
            status="error",
            query="q",
            error_message="出了点问题",
        )
        assert r.status == "error"
        assert r.error_message == "出了点问题"
        assert r.answer is None
