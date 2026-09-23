"""测试 DocManager SQLite 文档管理。"""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config import Config
from storage.doc_manager import DocManager


@pytest.fixture
def dm():
    """创建临时数据库的 DocManager（_ChunkStore 需真实 disk 但不实际写入）。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config()
        cfg.sqlite_dir = tmp
        cfg.lance_db_dir = tmp
        m = DocManager(cfg)
        yield m
        m.close()


def _make_chunks(n=1):
    return [
        {"id": f"abc_chunk_{i}", "chunk_index": i, "type": "text",
         "page_nums": [i + 1], "text": f"内容{i}"}
        for i in range(n)
    ]


def _add_sample_doc(dm: DocManager, file_md5="abc123", groups="default",
                     chunks=None):
    dm.add_document_meta(
        file_md5=file_md5,
        file_name="test.pdf",
        file_path="/tmp/test.pdf",
        file_size=1024,
        chunks=chunks or _make_chunks(1),
        groups=groups,
    )


class TestAddDocument:
    def _first_file(self, dm, group="default"):
        return dm.list_files()[group][0]

    def test_add_document_meta_creates_record(self, dm):
        _add_sample_doc(dm)
        f = self._first_file(dm)
        assert f["file_md5"] == "abc123"
        assert f["groups"] == "default"
        assert f["is_enabled"] == 1
        assert f["total_chunks"] == 1

    def test_add_document_meta_duplicate_skipped(self, dm):
        _add_sample_doc(dm)
        _add_sample_doc(dm)
        assert len(dm.list_files()["default"]) == 1


class TestListFiles:
    def test_groups_default_first(self, dm):
        _add_sample_doc(dm, "a", "zzz")
        _add_sample_doc(dm, "b", "aaa")
        _add_sample_doc(dm, "c", "default")
        assert list(dm.list_files().keys())[0] == "default"

    def test_list_enabled_doc_ids(self, dm):
        _add_sample_doc(dm, "a")
        _add_sample_doc(dm, "b")
        dm.set_file_enabled("b", False)
        assert dm.list_enabled_doc_ids() == ["a"]


class TestDelete:
    def test_delete_removes_record(self, dm):
        _add_sample_doc(dm)
        deleted = dm.delete_document("abc123")
        assert deleted == 1
        assert dm.list_files() == {}

    def test_delete_nonexistent_returns_zero(self, dm):
        assert dm.delete_document("nonexistent") == 0


class TestUpdate:
    @staticmethod
    def _first_file(dm, group="default"):
        return dm.list_files()[group][0]

    def test_set_groups(self, dm):
        _add_sample_doc(dm)
        dm.set_groups("abc123", "年报|审计")
        assert list(dm.list_files().keys()) == ["年报", "审计"]

    def test_set_tags(self, dm):
        _add_sample_doc(dm)
        dm.set_tags("abc123", "净利润|关联交易")
        assert self._first_file(dm)["tags"] == "净利润|关联交易"

    def test_set_file_type(self, dm):
        _add_sample_doc(dm)
        dm.set_file_type("abc123", "年报")
        assert self._first_file(dm)["file_type"] == "年报"

    def test_set_file_enabled(self, dm):
        _add_sample_doc(dm)
        dm.set_file_enabled("abc123", False)
        assert self._first_file(dm)["is_enabled"] == 0

    def test_set_group_enabled(self, dm):
        _add_sample_doc(dm, "a", "GP_A")
        _add_sample_doc(dm, "b", "GP_A")
        _add_sample_doc(dm, "c", "GP_B")
        dm.set_group_enabled("GP_A", False)
        for docs in dm.list_files().values():
            for d in docs:
                if "GP_A" in d["groups"]:
                    assert d["is_enabled"] == 0

    def test_set_group_enabled_default(self, dm):
        _add_sample_doc(dm, "a")
        dm.set_group_enabled("default", False)
        assert self._first_file(dm)["is_enabled"] == 0


@pytest.fixture
def dm_small_history():
    """操作历史滚动上限设为 3 的 DocManager（测试 rolling 用）。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config()
        cfg.sqlite_dir = tmp
        cfg.lance_db_dir = tmp
        cfg.operation_history_max_rows = 3
        m = DocManager(cfg)
        yield m
        m.close()


class TestOperationHistory:
    def test_stats_empty_returns_zeros(self, dm):
        assert dm.get_operation_stats() == {"min_id": 0, "max_id": 0, "total": 0}

    def test_stats_reflects_min_max_total(self, dm):
        # 1 条 index 记录 + 1 条 update 记录
        _add_sample_doc(dm)
        dm.set_file_enabled("abc123", False)
        stats = dm.get_operation_stats()
        assert stats == {"min_id": 1, "max_id": 2, "total": 2}

    def test_rolling_deletes_oldest_beyond_cap(self, dm_small_history):
        # 1 条 index + 5 条 update = 6 条，上限 3 → 仅保留最新 3 条（ID 4-6）
        _add_sample_doc(dm_small_history)
        for i in range(5):
            dm_small_history.set_file_enabled("abc123", i % 2 == 0)
        stats = dm_small_history.get_operation_stats()
        assert stats == {"min_id": 4, "max_id": 6, "total": 3}

    def test_rolling_keeps_latest_records(self, dm_small_history):
        # 保留的必须是最新的记录：最后一条操作为"启用文档"
        _add_sample_doc(dm_small_history)
        for enabled in (False, True, False, True):
            dm_small_history.set_file_enabled("abc123", enabled)
        ops = dm_small_history.list_operations(limit=10)
        assert len(ops) == 3
        assert ops[0]["op_detail"] == "启用文档"      # 最新在前（倒序）
        assert all(op["id"] >= 3 for op in ops)

    def test_list_operations_range_query_after_rolling(self, dm_small_history):
        # rolling 后按已删除的 ID 范围查询应返回空，不报错
        _add_sample_doc(dm_small_history)
        for i in range(5):
            dm_small_history.set_file_enabled("abc123", i % 2 == 0)
        assert dm_small_history.list_operations(limit=10, from_id=1, to_id=3) == []


class TestChunkStoreFts:
    """_ChunkStore 的 FTS 索引策略与重试。

    这个模块长期零测试覆盖——Windows 上的 PermissionDenied 重建失败就是
    从这里漏出去的。此处锁住三件事：走哪条索引路径、失败后重试几次、
    追加数据时还碰不碰 FTS。
    """

    @staticmethod
    def _store_with_mock_table(dm: DocManager) -> object:
        """把 _ChunkStore 的表换成 mock，便于断言调用参数与次数。"""
        store = dm._chunks
        store._table = MagicMock()
        return store

    def test_create_fts_index_uses_native_inverted(self, dm):
        """必须走原生倒排且关掉词长截断，否则中文长句整段进不了索引。"""
        store = self._store_with_mock_table(dm)

        store._create_fts_index()

        store._table.create_fts_index.assert_called_once_with(
            "text", use_tantivy=False, max_token_length=None,
        )

    def test_create_fts_index_retries_once_then_succeeds(self, dm):
        """首次撞上瞬时拒绝时，退避重试一次即可恢复，不向上抛。"""
        store = self._store_with_mock_table(dm)
        store._table.create_fts_index.side_effect = [
            ValueError("Failed to open file for write: PermissionDenied"), None,
        ]

        with patch("storage.doc_manager.time.sleep") as mock_sleep:
            store._create_fts_index()

        assert store._table.create_fts_index.call_count == 2
        mock_sleep.assert_called_once_with(2.0)

    def test_create_fts_index_raises_when_both_attempts_fail(self, dm):
        """两次都失败才抛出——上层据此记录索引失败。"""
        store = self._store_with_mock_table(dm)
        store._table.create_fts_index.side_effect = ValueError("PermissionDenied")

        with patch("storage.doc_manager.time.sleep"):
            with pytest.raises(ValueError, match="PermissionDenied"):
                store._create_fts_index()

        assert store._table.create_fts_index.call_count == 2

    def test_add_chunks_first_document_builds_fts(self, dm):
        """首次建表必须建 FTS，否则 BM25 通路整条失效。"""
        store = dm._chunks
        store._db = MagicMock()
        store._db.open_table.side_effect = Exception("Table not found")

        result = store.add_chunks(_make_chunks(1), np.zeros((1, 4), dtype=np.float32))

        assert result == {"added": 1, "skipped": False}
        store._table.create_fts_index.assert_called_once_with(
            "text", use_tantivy=False, max_token_length=None,
        )

    def test_add_chunks_append_does_not_touch_fts(self, dm):
        """追加数据不再重建 FTS——原生倒排索引随 append 自动维护。

        旧实现每次都全量重建（347 条要 9 秒、几十次文件增删），既是性能
        瓶颈，也是 Windows PermissionDenied 的来源。
        """
        store = dm._chunks
        store._db = MagicMock()
        store._table = MagicMock()
        store._db.open_table.return_value = store._table
        # 去重查询返回空表，表示该文档尚未入库，走正常追加
        store._table.search.return_value.where.return_value \
            .limit.return_value.to_pandas.return_value = pd.DataFrame()
        chunks = _make_chunks(2)
        for chunk in chunks:
            chunk["doc_id"] = "abc123"

        result = store.add_chunks(chunks, np.zeros((2, 4), dtype=np.float32))

        assert result == {"added": 2, "skipped": False}
        store._table.add.assert_called_once()
        store._table.create_fts_index.assert_not_called()


# 18 个汉字 / 54 字节，超过 tantivy 默认 40 字节的词长截断——旧路径会整段丢弃
_LONG_CJK_RUN = "仅仅是能够通过下个月的学院的剑术考核"


def _make_doc_chunk(doc_id: str, text: str) -> list:
    """构造单个 chunk 的写入载荷。"""
    return [{
        "id": f"{doc_id}_chunk_0", "doc_id": doc_id, "doc_name": f"{doc_id}.txt",
        "text": text, "chunk_index": 0, "type": "text", "page_nums": [1],
        "length": len(text),
    }]


class TestChunkStoreLanceDB:
    """_ChunkStore 的真实 LanceDB 集成（跑在临时库上，不碰 data/）。

    锁住方案 A 的核心承诺：换成原生倒排索引后，后追加的文档无需重建索引
    即可被全文检索到，且超过 40 字节的中文长串能进索引。旧 Tantivy 路径
    这两条都不成立——那是中文检索长期失效的根因。
    """

    @staticmethod
    def _fts_doc_ids(store, query: str) -> set:
        """跑一次全文检索，返回命中的 doc_id 集合。"""
        df = store._table.search(query, query_type="fts").limit(10).to_pandas()
        return set(df["doc_id"]) if "doc_id" in df.columns else set()

    def test_appended_document_searchable_without_rebuild(self, dm):
        """第二份文档追加后立即可全文检索，中间没有任何重建动作。"""
        store = dm._chunks
        store.add_chunks(_make_doc_chunk("doc_a", "无关内容"), np.zeros((1, 4), dtype=np.float32))
        store.add_chunks(_make_doc_chunk("doc_b", _LONG_CJK_RUN), np.zeros((1, 4), dtype=np.float32))

        assert "doc_b" in self._fts_doc_ids(store, _LONG_CJK_RUN)

    def test_appended_document_searchable_after_index_replace(self, dm):
        """索引被替换过（即迁移后的真实状态）之后，追加仍能自动进索引。"""
        store = dm._chunks
        store.add_chunks(_make_doc_chunk("doc_a", "无关内容"), np.zeros((1, 4), dtype=np.float32))
        store._table.create_fts_index(
            "text", use_tantivy=False, max_token_length=None, replace=True,
        )
        store.add_chunks(_make_doc_chunk("doc_b", _LONG_CJK_RUN), np.zeros((1, 4), dtype=np.float32))

        assert "doc_b" in self._fts_doc_ids(store, _LONG_CJK_RUN)
