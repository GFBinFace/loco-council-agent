"""测试 RAGPipeline 检索管线编排。"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.pipeline import RAGPipeline
from _types.retrieval_types import ChunkCandidate, ContinueChoice, SearchResult


def _make_chunk(
    chunk_id: str = "doc_chunk_0",
    text: str = "测试文本",
    doc_id: str = "abc123",
    chunk_index: int = 0,
    hybrid_score: float = 0.85,
) -> ChunkCandidate:
    """构造测试用 ChunkCandidate。"""
    return ChunkCandidate(
        id=chunk_id, text=text, doc_id=doc_id, doc_name="test.pdf",
        page_nums=[1], chunk_index=chunk_index, length=len(text),
        type="mixed", has_financial_keywords=[],
        hybrid_score=hybrid_score,
    )


def _make_mock_pipeline():
    """构造所有子组件都被 mock 的 RAGPipeline。"""
    with (
        patch("services.pipeline.SecurePDFProcessor"),
        patch("services.pipeline.FinancialTableChunker"),
        patch("services.pipeline.Embedder"),
        patch("services.pipeline.LanceDBHybridRetriever"),
        patch("services.pipeline.Reranker"),
        patch("services.pipeline.RerankLLM"),
        patch("services.pipeline.AnswerLLM"),
        patch("services.pipeline.DocManager"),
    ):
        pipeline = RAGPipeline()
        # 默认所有文档启用（检索不过滤）
        pipeline.doc_manager.list_enabled_doc_ids.return_value = None
        return pipeline


class TestPipelineSearch:
    """RAGPipeline.search() 开心路径"""

    def test_search_happy_path(self):
        """开心路径：混合检索→二次排序→打分→间隙填充→生成，返回 success"""
        pipeline = _make_mock_pipeline()

        # Mock 混合检索：返回 3 个候选
        mock_hybrid_chunks = [_make_chunk(f"c{i}", f"文本{i}", chunk_index=i) for i in range(3)]
        pipeline.retriever.search.return_value = mock_hybrid_chunks

        # Mock CrossEncoder二次排序
        pipeline.reranker.cross_encoder_rerank.return_value = mock_hybrid_chunks

        # Mock LLM打分与分级收网
        pipeline.reranker.llm_score_and_sieve.return_value = mock_hybrid_chunks

        # Mock 间隙填充
        pipeline._fill_gaps = MagicMock(return_value=mock_hybrid_chunks)

        # Mock 生成
        pipeline._llm_answer.ask.return_value = "这是基于3个chunk的答案。"

        result = pipeline.search("资产负债表")

        assert isinstance(result, SearchResult)
        assert result.status == "success"
        assert "答案" in result.answer
        assert result.source_count == 3
        assert result.query == "资产负债表"

    def test_search_zero_results_triggers_user_choice(self):
        """混合检索返回空 → needs_user_choice (zero_results)"""
        pipeline = _make_mock_pipeline()
        pipeline.retriever.search.return_value = []

        result = pipeline.search("查询")

        assert result.status == "needs_user_choice"
        assert result.pending_decision == "zero_results"
        assert pipeline._pending_state is not None
        assert pipeline._pending_state["decision_type"] == "zero_results"

    def test_search_low_confidence_triggers_user_choice(self):
        """LLM打分后全 <5 → needs_user_choice (low_confidence)"""
        pipeline = _make_mock_pipeline()

        mock_cross_encoder_chunks = [
            _make_chunk("c0", "文本0", chunk_index=0),
            _make_chunk("c1", "文本1", chunk_index=1),
        ]
        pipeline.retriever.search.return_value = mock_cross_encoder_chunks
        pipeline.reranker.cross_encoder_rerank.return_value = mock_cross_encoder_chunks
        pipeline.reranker.llm_score_and_sieve.return_value = []  # 低置信度

        result = pipeline.search("查询")

        assert result.status == "needs_user_choice"
        assert result.pending_decision == "low_confidence"
        assert pipeline._pending_state is not None
        assert pipeline._pending_state["decision_type"] == "low_confidence"
        assert len(pipeline._pending_state["cross_encoder_top5"]) == 2

    def test_search_progress_callback_order(self):
        """进度回调按正确顺序调用——新协议 (status_line, log_line)。"""
        pipeline = _make_mock_pipeline()

        chunks = [_make_chunk("c0")]
        pipeline.retriever.search.return_value = chunks
        pipeline.reranker.cross_encoder_rerank.return_value = chunks
        pipeline.reranker.llm_score_and_sieve.return_value = chunks
        pipeline._llm_answer.ask.return_value = "答案"

        progress_calls = []

        def track(status_line, log_line):
            progress_calls.append((status_line, log_line))

        pipeline.search("查询", on_progress=track)

        # 验证：至少包含关键阶段的回调
        # 混合检索（仅 log）、CrossEncoder（status+log）、LLM打分（status+log）、
        # 间隙填充（仅 log）、生成回答（status+log）、任务完成（仅 status）
        status_lines = [s for s, _ in progress_calls if s is not None]
        log_lines = [l for _, l in progress_calls if l is not None]
        # 应有阶段开始的 status_line（CrossEncoder、LLM 打分、生成回答）+ 任务完成
        assert len(status_lines) >= 2
        # 每个阶段首尾都有 log
        assert len(log_lines) >= 8
        # 最后一个 status_line 是任务完成
        assert "✅" in status_lines[-1]


class TestPipelineContinueSearch:
    """RAGPipeline.continue_search() 测试"""

    def test_continue_search_direct_llm_zero_results(self):
        """零结果场景，用户选 direct_llm"""
        pipeline = _make_mock_pipeline()
        pipeline._pending_state = {
            "query": "测试查询",
            "decision_type": "zero_results",
        }
        pipeline._llm_answer.ask.return_value = "纯LLM回答"

        result = pipeline.continue_search(ContinueChoice.DIRECT_LLM)

        assert result.status == "success"
        assert "纯LLM回答" in result.answer
        assert result.source_count == 0
        pipeline._llm_answer.ask.assert_called_once()

    def test_continue_search_rag_low_confidence(self):
        """低置信度场景，用户选 rag → 用 CrossEncoder top-5"""
        pipeline = _make_mock_pipeline()
        top5 = [_make_chunk(f"c{i}", chunk_index=i) for i in range(5)]
        pipeline._pending_state = {
            "query": "测试查询",
            "decision_type": "low_confidence",
            "cross_encoder_top5": top5,
        }
        pipeline._adjacent_merge = MagicMock(return_value=top5)
        pipeline._llm_answer.ask.return_value = "低置信度回答"

        result = pipeline.continue_search(ContinueChoice.RAG)

        assert result.status == "success"
        assert result.source_count == 5
        pipeline._llm_answer.ask.assert_called_once()

    def test_continue_search_rag_wrong_decision_type(self):
        """零结果场景调用 rag 选项 → 返回 error"""
        pipeline = _make_mock_pipeline()
        pipeline._pending_state = {
            "query": "查询",
            "decision_type": "zero_results",
        }

        result = pipeline.continue_search(ContinueChoice.RAG)

        assert result.status == "error"
        assert "仅适用于低置信度场景" in result.error_message

    def test_continue_search_abandon(self):
        """用户放弃查询"""
        pipeline = _make_mock_pipeline()
        pipeline._pending_state = {
            "query": "查询",
            "decision_type": "zero_results",
        }

        result = pipeline.continue_search(ContinueChoice.ABANDON)

        assert result.status == "success"
        assert "取消" in result.answer
        assert result.source_count == 0

    def test_continue_search_no_pending_state(self):
        """未调用 search() 直接调用 continue_search → error"""
        pipeline = _make_mock_pipeline()
        pipeline._pending_state = None

        result = pipeline.continue_search(ContinueChoice.DIRECT_LLM)

        assert result.status == "error"
        assert "没有待处理" in result.error_message


class TestFillGaps:
    """_fill_gaps 单间隙填充测试"""

    def _make_pipeline_for_fill(self):
        """构造带 mock LanceDB 表的 pipeline，用于测试间隙填充。"""
        with (
            patch("services.pipeline.SecurePDFProcessor"),
            patch("services.pipeline.FinancialTableChunker"),
            patch("services.pipeline.Embedder"),
            patch("services.pipeline.LanceDBHybridRetriever"),
            patch("services.pipeline.Reranker"),
            patch("services.pipeline.RerankLLM"),
            patch("services.pipeline.AnswerLLM"),
        ):
            pipeline = RAGPipeline()
            pipeline.retriever.table = MagicMock()
            pipeline.retriever.db = MagicMock()
            return pipeline

    def _mock_gap_row(self, chunk_id="gap", text="间隙文本",
                      doc_id="abc123", chunk_index=1):
        """构造 LanceDB 行数据，模拟间隙 chunk。"""
        return {
            "id": chunk_id, "text": text, "doc_id": doc_id,
            "chunk_index": chunk_index, "doc_name": "test.pdf",
            "page_nums": "[1]", "length": len(text),
            "type": "mixed", "has_financial_keywords": "[]",
        }

    def test_fill_gaps_single_gap_inserts_one(self):
        """chunk_index 0 和 2（间隙=1）→ 插入 chunk_1，结果 3 个"""
        import pandas as pd

        pipeline = self._make_pipeline_for_fill()
        a = _make_chunk("a", "文本A", chunk_index=0)
        b = _make_chunk("b", "文本B", chunk_index=2)

        mock_df = pd.DataFrame([self._mock_gap_row("gap_1", "间隙文本", chunk_index=1)])
        pipeline.retriever.table.to_pandas.return_value = mock_df

        result = pipeline._fill_gaps([a, b])

        assert len(result) == 3
        assert result[0].id == "a"
        assert result[1].id == "gap_1"
        assert result[2].id == "b"
        # 元数据完整保留
        assert result[1].text == "间隙文本"

    def test_fill_gaps_cascade(self):
        """1,3,5 → 级联插入 2 和 4 → 1,2,3,4,5"""
        import pandas as pd

        pipeline = self._make_pipeline_for_fill()

        candidates = [
            _make_chunk("c1", "文本1", chunk_index=1),
            _make_chunk("c3", "文本3", chunk_index=3),
            _make_chunk("c5", "文本5", chunk_index=5),
        ]

        gap_rows = [
            self._mock_gap_row("c2", "间隙2", chunk_index=2),
            self._mock_gap_row("c4", "间隙4", chunk_index=4),
        ]
        mock_df = pd.DataFrame(gap_rows)
        pipeline.retriever.table.to_pandas.return_value = mock_df

        result = pipeline._fill_gaps(candidates)

        assert len(result) == 5
        assert [c.id for c in result] == ["c1", "c2", "c3", "c4", "c5"]

    def test_fill_gaps_respects_ratio_limit(self):
        """gap_fill_ratio=0.5，原始 4 个候选 → 最多填充 2 个"""
        import pandas as pd

        pipeline = self._make_pipeline_for_fill()
        pipeline.config.gap_fill_ratio = 0.5  # 4 * 0.5 = 2

        candidates = [
            _make_chunk("c1", chunk_index=1),
            _make_chunk("c3", chunk_index=3),
            _make_chunk("c5", chunk_index=5),
            _make_chunk("c7", chunk_index=7),
        ]

        gap_rows = [
            self._mock_gap_row("c2", chunk_index=2),
            self._mock_gap_row("c4", chunk_index=4),
            self._mock_gap_row("c6", chunk_index=6),
        ]
        mock_df = pd.DataFrame(gap_rows)
        pipeline.retriever.table.to_pandas.return_value = mock_df

        result = pipeline._fill_gaps(candidates)

        # 最多填充 2 个，填充到 c4 后停止
        assert len(result) == 6  # 4 + 2
        gap_ids = [c.id for c in result if c.id.startswith("c") and int(c.id[1:]) % 2 == 0]
        assert len(gap_ids) == 2

    def test_fill_gaps_no_gap_preserves_all(self):
        """chunk_index 连续（差 1）→ 不填充，全部保留"""
        pipeline = self._make_pipeline_for_fill()

        a = _make_chunk("a", chunk_index=0)
        b = _make_chunk("b", chunk_index=1)

        mock_df = MagicMock()
        pipeline.retriever.table.to_pandas.return_value = mock_df

        result = pipeline._fill_gaps([a, b])

        assert len(result) == 2

    def test_fill_gaps_different_docs_no_fill(self):
        """不同 doc_id → 不填充"""
        pipeline = self._make_pipeline_for_fill()

        a = _make_chunk("a", doc_id="abc", chunk_index=0)
        b = _make_chunk("b", doc_id="xyz", chunk_index=2)

        mock_df = MagicMock()
        pipeline.retriever.table.to_pandas.return_value = mock_df

        result = pipeline._fill_gaps([a, b])

        assert len(result) == 2

    def test_fill_gaps_empty_returns_empty(self):
        """空列表返回空"""
        pipeline = self._make_pipeline_for_fill()
        result = pipeline._fill_gaps([])
        assert result == []

    def test_fill_gaps_single_candidate_returns_self(self):
        """单个候选直接返回"""
        pipeline = self._make_pipeline_for_fill()
        c = _make_chunk("a")
        pipeline.retriever.table.to_pandas.return_value = MagicMock()

        result = pipeline._fill_gaps([c])

        assert len(result) == 1
        assert result[0].id == "a"

    def test_fill_gaps_gap_not_in_db_skipped(self):
        """间隙 chunk 在 LanceDB 中不存在 → 不插入"""
        import pandas as pd

        pipeline = self._make_pipeline_for_fill()
        a = _make_chunk("a", chunk_index=0)
        b = _make_chunk("b", chunk_index=2)

        # 空 DataFrame，查不到 gap
        pipeline.retriever.table.to_pandas.return_value = pd.DataFrame()

        result = pipeline._fill_gaps([a, b])

        assert len(result) == 2
