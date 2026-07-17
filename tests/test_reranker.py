"""测试 Reranker 重排序。"""

from unittest.mock import MagicMock, patch

import pytest

from _types.retrieval_types import ChunkCandidate


def _make_chunk(
    chunk_id: str = "doc_chunk_0",
    text: str = "测试文本",
    doc_id: str = "abc123",
    chunk_index: int = 0,
) -> ChunkCandidate:
    """构造测试用 ChunkCandidate。"""
    return ChunkCandidate(
        id=chunk_id,
        text=text,
        doc_id=doc_id,
        doc_name="test.pdf",
        page_nums=[1],
        chunk_index=chunk_index,
        length=len(text),
        type="mixed",
        has_financial_keywords=[],
    )


class TestCrossEncoderRerank:
    """CrossEncoder 二次排序"""

    @pytest.fixture
    def reranker(self):
        """创建带 mock CrossEncoder 的 Reranker。"""
        with patch("services.retrieval.reranker.CrossEncoder") as mock_ce:
            from services.retrieval.reranker import Reranker
            r = Reranker()
            r.model = mock_ce.return_value
            yield r

    def test_cross_encoder_rerank_scores_and_truncates(self, reranker):
        """20 个候选 → 返回 top_k=10 个，按分数降序"""
        candidates = [_make_chunk(f"c{i}", f"text {i}") for i in range(20)]

        # Mock predict 返回降序分数
        mock_scores = [0.9 - i * 0.04 for i in range(20)]  # 0.9, 0.86, 0.82, ...
        reranker.model.predict.return_value = mock_scores

        result = reranker.cross_encoder_rerank("test query", candidates, top_k=10)

        assert len(result) == 10
        # 每个 chunk 都有 cross_encoder_score
        for c in result:
            assert c.cross_encoder_score is not None
        # 按分数降序
        for i in range(len(result) - 1):
            assert (result[i].cross_encoder_score or 0.0) >= (result[i + 1].cross_encoder_score or 0.0)

    def test_cross_encoder_rerank_empty_candidates_returns_empty(self, reranker):
        """候选为空时返回空列表"""
        result = reranker.cross_encoder_rerank("query", [], top_k=10)
        assert result == []

    def test_cross_encoder_rerank_fewer_than_top_k_returns_all(self, reranker):
        """候选数 < top_k 时全返回"""
        candidates = [_make_chunk(f"c{i}") for i in range(3)]
        reranker.model.predict.return_value = [0.8, 0.6, 0.4]

        result = reranker.cross_encoder_rerank("query", candidates, top_k=10)

        assert len(result) == 3


class TestLLMScoreAndSieve:
    """LLM 打分 + 分级收网"""

    @pytest.fixture
    def reranker(self):
        """创建带 mock CrossEncoder 的 Reranker。"""
        with patch("services.retrieval.reranker.CrossEncoder"):
            from services.retrieval.reranker import Reranker
            yield Reranker()

    def _mock_llm_scores(self, candidates, scores):
        """构造 mock RerankLLM 的 score_chunks 返回值。"""
        mock_llm = MagicMock()
        mock_llm.score_chunks.return_value = [
            {"chunk_id": c.id, "score": s, "reason": f"得分{s}"}
            for c, s in zip(candidates, scores)
        ]
        return mock_llm

    def test_tiered_sieving_high_confidence_tier_8(self, reranker):
        """N=8, >=8 的有 5 个 (>4) → 取全部 >=8"""
        N = 8
        candidates = [_make_chunk(f"c{i}") for i in range(N)]
        # 5 个 >=8, 3 个 <8
        scores = [9, 8, 8, 9, 8, 4, 3, 2]
        mock_llm = self._mock_llm_scores(candidates, scores)

        result = reranker.llm_score_and_sieve("query", candidates, mock_llm)

        # 应取 5 个 >=8 的
        assert len(result) == 5
        for c in result:
            assert c.llm_score is not None and c.llm_score >= 8

    def test_tiered_sieving_mid_tier_7(self, reranker):
        """N=10, >=8 的只有 4 个 (<=5) → >=7 的有 7 个 (>5) → 取全部 >=7"""
        N = 10
        candidates = [_make_chunk(f"c{i}") for i in range(N)]
        # >=8: 4个, >=7: 7个(含前4个), >=6: 8个
        scores = [8, 8, 8, 9, 7, 7, 7, 6, 3, 2]
        mock_llm = self._mock_llm_scores(candidates, scores)

        result = reranker.llm_score_and_sieve("query", candidates, mock_llm)

        # >=8 不到半数，触发 >=7（7 个 > 5 = N/2）
        assert len(result) == 7
        for c in result:
            assert c.llm_score is not None and c.llm_score >= 7

    def test_tiered_sieving_fallback_to_5(self, reranker):
        """N=6, 无任何阈值超半数 → 兜底取 >=5"""
        N = 6
        candidates = [_make_chunk(f"c{i}") for i in range(N)]
        # 最高 6 分，都不超过半数
        scores = [6, 5, 5, 4, 4, 3]
        mock_llm = self._mock_llm_scores(candidates, scores)

        result = reranker.llm_score_and_sieve("query", candidates, mock_llm)

        # 兜底取 >=5
        assert len(result) == 3  # 6, 5, 5
        for c in result:
            assert c.llm_score is not None and c.llm_score >= 5

    def test_tiered_sieving_all_below_5_returns_empty(self, reranker):
        """所有 chunk 得分 < 5 → 返回空列表（低置信度）"""
        N = 6
        candidates = [_make_chunk(f"c{i}") for i in range(N)]
        scores = [4, 3, 2, 1, 0, 0]
        mock_llm = self._mock_llm_scores(candidates, scores)

        result = reranker.llm_score_and_sieve("query", candidates, mock_llm)

        assert result == []

    def test_tiered_sieving_empty_candidates_returns_empty(self, reranker):
        """候选为空时返回空列表"""
        mock_llm = MagicMock()
        result = reranker.llm_score_and_sieve("query", [], mock_llm)
        assert result == []

    def test_llm_score_and_sieve_writes_scores_to_candidates(self, reranker):
        """LLM 打分结果正确写回 ChunkCandidate"""
        candidates = [_make_chunk(f"c{i}") for i in range(5)]
        scores = [8, 7, 6, 5, 9]
        mock_llm = self._mock_llm_scores(candidates, scores)

        reranker.llm_score_and_sieve("query", candidates, mock_llm)

        assert candidates[0].llm_score == 8
        assert candidates[0].llm_reason == "得分8"
        assert candidates[4].llm_score == 9

    def test_llm_score_and_sieve_fills_missing_with_minus_one(self, reranker):
        """LLM 未返回某 chunk 评分 → 该 chunk 的 llm_score 置 -1"""
        candidates = [_make_chunk(f"c{i}") for i in range(3)]
        # 只返回 c0 和 c1 的分数，漏了 c2
        mock_llm = MagicMock()
        mock_llm.score_chunks.return_value = [
            {"chunk_id": "c0", "score": 8, "reason": "好"},
            {"chunk_id": "c1", "score": 5, "reason": "一般"},
        ]

        reranker.llm_score_and_sieve("query", candidates, mock_llm)

        assert candidates[0].llm_score == 8
        assert candidates[1].llm_score == 5
        assert candidates[2].llm_score == -1
        assert "LLM 未返回" in candidates[2].llm_reason

    def test_llm_score_and_sieve_empty_response_fills_all_minus_one(self, reranker):
        """LLM 完全无响应 → 所有候选 llm_score 置 -1"""
        candidates = [_make_chunk(f"c{i}") for i in range(3)]
        mock_llm = MagicMock()
        mock_llm.score_chunks.return_value = []

        reranker.llm_score_and_sieve("query", candidates, mock_llm)

        for c in candidates:
            assert c.llm_score == -1
