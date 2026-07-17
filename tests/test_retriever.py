"""测试 retriever 混合检索（适配强制 reopen 表）。"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from _types.retrieval_types import ChunkCandidate


def _make_mock_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_mock_row(
    idx: int = 0,
    chunk_id: str = "doc_chunk_0",
    text: str = "测试文本内容",
    doc_id: str = "abc123",
    doc_name: str = "test.pdf",
    page_nums: str = "[1, 2]",
    chunk_index: int = 0,
    length: int = 10,
    chunk_type: str = "mixed",
    keywords: str = "['净利润']",
    relevance_score: float = 0.85,
) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_nums": page_nums,
        "chunk_index": chunk_index,
        "length": length,
        "type": chunk_type,
        "has_financial_keywords": keywords,
        "_relevance_score": relevance_score,
    }


class TestRetrieverSearch:

    @pytest.fixture
    def retriever(self):
        with patch("services.retrieval.retriever.lancedb.connect") as mock_connect:
            from services.retrieval.retriever import LanceDBHybridRetriever
            r = LanceDBHybridRetriever()
            yield r

    # 所有测试共享的 mock 表构建器（适配强制 reopen：open_table 返回此表）
    @staticmethod
    def _setup_mock_table(retriever, mock_df):
        mock_table = MagicMock()
        retriever.db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.rerank.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_pandas.return_value = mock_df
        return mock_search

    def test_search_returns_candidates(self, retriever):
        mock_rows = [
            _make_mock_row(0, "doc_chunk_0", "资产负债表数据", chunk_index=0, relevance_score=0.92),
            _make_mock_row(1, "doc_chunk_1", "利润表数据", chunk_index=1, relevance_score=0.78),
        ]
        self._setup_mock_table(retriever, _make_mock_df(mock_rows))

        query_vector = np.random.randn(1024).astype(np.float32)
        result = retriever.search("查询", query_vector)

        assert len(result) == 2
        assert isinstance(result[0], ChunkCandidate)
        assert result[0].id == "doc_chunk_0"
        assert result[0].hybrid_score >= result[1].hybrid_score

    def test_search_empty_table_returns_empty_list(self, retriever):
        retriever.db.open_table.side_effect = Exception("Table not found")

        query_vector = np.random.randn(1024).astype(np.float32)
        result = retriever.search("查询", query_vector)

        assert result == []

    def test_search_parses_page_nums_from_string(self, retriever):
        mock_row = _make_mock_row(0, "chunk_0", "文本", page_nums="[3, 4, 5]", keywords="['资产', '负债']")
        self._setup_mock_table(retriever, _make_mock_df([mock_row]))

        query_vector = np.random.randn(1024).astype(np.float32)
        result = retriever.search("查询", query_vector)

        assert len(result) == 1
        assert result[0].page_nums == [3, 4, 5]
        assert result[0].has_financial_keywords == ["资产", "负债"]

    def test_search_handles_malformed_page_nums(self, retriever):
        mock_row = _make_mock_row(0, "chunk_0", "文本", page_nums="malformed")
        self._setup_mock_table(retriever, _make_mock_df([mock_row]))

        query_vector = np.random.randn(1024).astype(np.float32)
        result = retriever.search("查询", query_vector)

        assert result[0].page_nums == []
