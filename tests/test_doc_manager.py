"""测试 DocManager SQLite 文档管理。"""

import tempfile

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
