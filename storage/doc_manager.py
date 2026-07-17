"""文档数据管理模块。

DocManager 是业务层操作文档数据的唯一入口，内部协调两个存储后端：
- _DocMetaStore（SQLite）：文档元数据 + chunk 索引
- _ChunkStore（LanceDB）：向量索引 + FTS

检索功能由 retriever.py 独立提供（纯读操作，不参与数据写入）。
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from _types.retrieval_types import ChunkCandidate

from utils import get_file_logger
logger = get_file_logger(__file__)
# 每个文档默认的分组
_DEFAULT_GROUP = "default"


# ═══════════════════════════════════════════════════════════════
# _ChunkStore — LanceDB 向量存储
# ═══════════════════════════════════════════════════════════════

class _ChunkStore:
    """LanceDB 向量存储——负责 chunk 的向量写入、FTS 索引和按文档删除。"""

    def __init__(self, config: Config):
        import lancedb
        from lancedb.rerankers import RRFReranker

        os.makedirs(config.lance_db_dir, exist_ok=True)
        self._db = lancedb.connect(config.lance_db_dir)
        self._table_name = "financial_docs"
        self._table = None
        self._reranker = RRFReranker()

    # ── 写入 ──────────────────────────────────────────────

    def add_chunks(
        self, chunks: List[Dict], embeddings: np.ndarray
    ) -> Dict:
        """写入 chunk 向量到 LanceDB。首次调用时创建表并建索引。"""
        df = self._chunks_to_df(chunks, embeddings)
        try:
            self._table = self._db.open_table(self._table_name)
            table_exists = True
        except Exception:
            table_exists = False

        # 如果表不存在，则创建表并建索引。
        if not table_exists:
            self._table = self._db.create_table(
                self._table_name, data=df, mode="overwrite",
            )
            self._table.create_fts_index("text")
            if len(df) >= 256:
                num_partitions = min(256, max(1, len(df) - 1))
                self._table.create_index(
                    metric="cosine",
                    num_partitions=num_partitions,
                    num_sub_vectors=64,
                    vector_column_name="vector",
                )
                logger.info("LanceDB 表已创建（含向量索引），%d 条 chunk", len(chunks))
            else:
                logger.info("LanceDB 表已创建（跳过向量索引），%d 条 chunk", len(chunks))
            return {'added': len(chunks), 'skipped': False}

        # 如果能查到这个文档的数据已经存在了，就不再添加这次的数据，而是直接返回。
        doc_id = chunks[0].get('doc_id', '') if chunks else ''
        if doc_id:
            existing = self._table.search().where(
                f"doc_id = '{doc_id}'"
            ).limit(1).to_pandas()
            if not existing.empty:
                logger.info("文档 %s 已存在 LanceDB 中，跳过", doc_id[:16])
                return {'added': 0, 'skipped': True}

        # 常规写入数据。
        self._table.add(df)
        logger.info("LanceDB 写入 %d 条 chunk", len(chunks))
        # 追加数据后重建 FTS 索引——LanceDB 的 Tantivy FTS 不会自动索引
        # 后续 ADD 的新行。若表已存在且此前已建过 FTS，重建覆盖旧索引，
        # 确保全文检索覆盖全部数据（曾致首文档外所有被追加文档的 BM25 通路
        # 静默返回空——全表有数据但搜不到）。
        try:
            self._table.create_fts_index("text", replace=True)
        except Exception:
            logger.exception(
                "FTS 索引重建失败——数据已写入 LanceDB 但全文检索不可用，"
                "请删除此文档后重新索引"
            )
            raise
        logger.info(
            "FTS 索引已重建，覆盖 %d 条 chunk", self._table.count_rows(),
        )
        return {'added': len(chunks), 'skipped': False}

    # ── 删除 ──────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        删除指定文档的所有 chunk 向量。表不存在时返回 0。

        安全前提：doc_id 为 MD5 hex 字符串（0-9a-f），不含 SQL 特殊字符，
        因此 f-string 拼接在此处是安全的。LanceDB 不支持参数化 delete。
        """
        try:
            self._table = self._db.open_table(self._table_name)
        except Exception:
            return 0
        before = self._table.count_rows()
        self._table.delete(f"doc_id = '{doc_id}'")
        deleted = before - self._table.count_rows()
        if deleted > 0:
            logger.info("LanceDB 删除文档 %s: %d 行", doc_id[:16], deleted)
        return deleted

    @staticmethod
    def _chunks_to_df(
        chunks: List[Dict], embeddings: np.ndarray
    ) -> pd.DataFrame:
        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "vector": embeddings[i].tolist(),
                "type": chunk.get("type", "mixed"),
                "doc_id": chunk.get("doc_id", ""),
                "doc_name": chunk.get("doc_name", ""),
                "page_nums": str(chunk.get("page_nums", [])),
                "chunk_index": chunk.get("chunk_index", 0),
                "length": chunk.get("length", 0),
                "chapter_title": str(chunk.get("chapter_title", "") or ""),
                "chapter_index": str(chunk.get("chapter_index", "") or ""),
                "has_financial_keywords": str(
                    chunk.get("has_financial_keywords", [])
                ),
            })
        return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════
# _DocMetaStore — SQLite 文档元数据
# ═══════════════════════════════════════════════════════════════

class _DocMetaStore:
    """SQLite 元数据存储——管理 files 表和 chunks 表的记录。"""

    def __init__(self, config: Config):
        self._db_path = os.path.join(config.sqlite_dir, "docs.sqlite")
        self._op_max_rows = config.operation_history_max_rows
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    # ── 建表 ──────────────────────────────────────────────

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                file_md5     TEXT PRIMARY KEY,
                file_name    TEXT NOT NULL,
                file_path    TEXT NOT NULL,
                file_size    INTEGER NOT NULL,
                file_type    TEXT NOT NULL DEFAULT '',
                groups       TEXT NOT NULL DEFAULT '""" + _DEFAULT_GROUP + """',
                indexed_at   TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'ready',
                tags         TEXT DEFAULT '',
                total_chunks INTEGER DEFAULT 0,
                is_enabled   INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                file_md5    TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_type  TEXT NOT NULL DEFAULT 'mixed',
                page_nums   TEXT NOT NULL DEFAULT '[]',
                text        TEXT NOT NULL,
                FOREIGN KEY (file_md5) REFERENCES files(file_md5)
            );

            CREATE TABLE IF NOT EXISTS operation_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                op_type      TEXT NOT NULL,
                op_detail    TEXT NOT NULL,
                file_name    TEXT NOT NULL,
                file_md5     TEXT NOT NULL,
                chunk_count  INTEGER DEFAULT 0,
                op_result    TEXT DEFAULT 'success',
                op_time      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_op_history_time
                ON operation_history(op_time DESC);

            CREATE INDEX IF NOT EXISTS idx_chunks_file_md5
                ON chunks(file_md5);
        """)
        self._conn.commit()

    # ── 写入 ──────────────────────────────────────────────

    def add_document_meta(
        self,
        file_md5: str,
        file_name: str,
        file_path: str,
        file_size: int,
        chunks: List[Dict],
        file_type: str = "",
        groups: str = _DEFAULT_GROUP,
        tags: str = "",
        lancedb_result: Optional[Dict] = None,
    ) -> None:
        """一次性写入 files 行和 chunks 行，并记录操作历史。"""
        lr = lancedb_result or {}
        skipped = lr.get("skipped", False)
        try:
            self._conn.execute("BEGIN")
            # 写入 files 表
            self._conn.execute(
                "INSERT OR IGNORE INTO files (file_md5, file_name, file_path, "
                "file_size, file_type, groups, indexed_at, tags, total_chunks) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_md5, file_name, file_path, file_size,
                    file_type, groups,
                    datetime.now().isoformat(),
                    tags, len(chunks),
                ),
            )
            # 写入 chunks 表
            if chunks:
                rows = [
                    (
                        c["id"], file_md5,
                        c.get("chunk_index", 0),
                        c.get("type", "mixed"),
                        str(c.get("page_nums", [])),
                        c["text"],
                    )
                    for c in chunks
                ]
                self._conn.executemany(
                    "INSERT OR IGNORE INTO chunks (chunk_id, file_md5, chunk_index, "
                    "chunk_type, page_nums, text) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
            # 记录操作历史（与业务数据同事务，原子落盘）
            if skipped:
                self._record_operation(
                    "index", "跳过索引 (MD5 重复)", file_md5, file_name,
                    len(chunks), "skipped",
                )
            else:
                self._record_operation(
                    "index", "新建索引", file_md5, file_name,
                    len(chunks), "success",
                )
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            logger.exception("写入文档元数据失败: %s", file_md5[:16])
            raise

        logger.info(
            "SQLite 写入文档元数据: %s (%d chunks)", file_name, len(chunks),
        )

    # ── 查询 ──────────────────────────────────────────────

    def list_files_grouped(self) -> Dict[str, List[Dict]]:
        rows = self._conn.execute(
            "SELECT file_md5, file_name, file_path, file_size, file_type, "
            "groups, indexed_at, status, tags, total_chunks, is_enabled "
            "FROM files ORDER BY groups, file_name"
        ).fetchall()

        result: Dict[str, List[Dict]] = {}
        for row in rows:
            d = dict(row)
            for g in d["groups"].split("|"):
                result.setdefault(g, []).append(d)

        if _DEFAULT_GROUP in result:
            ordered = {_DEFAULT_GROUP: result.pop(_DEFAULT_GROUP)}
            ordered.update(result)
            return ordered
        return result

    def list_enabled_doc_ids(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT file_md5 FROM files WHERE is_enabled = 1"
        ).fetchall()
        return [row["file_md5"] for row in rows]

    def has_document(self, file_md5: str) -> bool:
        """检查指定 MD5 的文档是否已在 SQLite 中（表示已完整索引）。"""
        row = self._conn.execute(
            "SELECT 1 FROM files WHERE file_md5 = ?", (file_md5,)
        ).fetchone()
        return row is not None

    def has_any_docs(self) -> bool:
        """SQLite 中是否有任何文件记录。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM files"
        ).fetchone()
        return row["cnt"] > 0

    def get_overview(self) -> dict:
        """
        返回知识库概览统计。

        Returns:
            {
                "total_docs": int,
                "enabled_docs": int,
                "total_chunks": int,
                "total_chars": int | None,
                "latest_index": str | None,
            }
        """
        total_docs = self._conn.execute(
            "SELECT COUNT(*) FROM files"
        ).fetchone()[0]
        enabled = self._conn.execute(
            "SELECT COUNT(*) FROM files WHERE is_enabled = 1"
        ).fetchone()[0]
        total_chunks = self._conn.execute(
            "SELECT COALESCE(SUM(total_chunks), 0) FROM files"
        ).fetchone()[0]
        try:
            total_chars = self._conn.execute(
                "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM chunks"
            ).fetchone()[0]
        except Exception:
            total_chars = -1
        row = self._conn.execute(
            "SELECT MAX(indexed_at) FROM files WHERE total_chunks > 0"
        ).fetchone()
        latest_index = row[0] if row and row[0] else None
        return {
            "total_docs": total_docs or 0,
            "enabled_docs": enabled or 0,
            "total_chunks": total_chunks or 0,
            "total_chars": total_chars if total_chars >= 0 else None,
            "latest_index": latest_index,
        }

    # ── 删除 ──────────────────────────────────────────────

    def delete_document(self, file_md5: str) -> int:
        # 删除前获取文档信息，供操作记录
        old_row = self._conn.execute(
            "SELECT file_name, total_chunks FROM files WHERE file_md5 = ?",
            (file_md5,),
        ).fetchone()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "DELETE FROM chunks WHERE file_md5 = ?", (file_md5,)
            )
            cursor = self._conn.execute(
                "DELETE FROM files WHERE file_md5 = ?", (file_md5,)
            )
            if old_row:
                self._record_operation(
                    "delete", "删除文档", file_md5,
                    old_row["file_name"], old_row["total_chunks"], "success",
                )
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            raise
        count = cursor.rowcount
        if count > 0:
            logger.info("SQLite 删除文档: %s (%d 行)", file_md5[:16], count)
        return count

    # ── 更新 ──────────────────────────────────────────────

    def set_groups(self, file_md5: str, groups: str) -> None:
        old_row = self._conn.execute(
            "SELECT file_name, total_chunks, groups FROM files WHERE file_md5 = ?",
            (file_md5,),
        ).fetchone()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE files SET groups = ? WHERE file_md5 = ?",
                (groups, file_md5),
            )
            if old_row:
                self._record_operation(
                    "update",
                    f"修改分组: {old_row['groups']} → {groups}",
                    file_md5, old_row["file_name"], old_row["total_chunks"], "success",
                )
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            logger.exception("设置 groups 失败: %s", file_md5[:16])
            raise

    def set_tags(self, file_md5: str, tags: str) -> None:
        old_row = self._conn.execute(
            "SELECT file_name, total_chunks, tags FROM files WHERE file_md5 = ?",
            (file_md5,),
        ).fetchone()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE files SET tags = ? WHERE file_md5 = ?",
                (tags, file_md5),
            )
            if old_row:
                self._record_operation(
                    "update",
                    f"修改标签: {old_row['tags'] or '空'} → {tags or '空'}",
                    file_md5, old_row["file_name"], old_row["total_chunks"], "success",
                )
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            logger.exception("设置 tags 失败: %s", file_md5[:16])
            raise

    def set_file_type(self, file_md5: str, file_type: str) -> None:
        """更新文档的 file_type 字段。"""
        old_row = self._conn.execute(
            "SELECT file_name, total_chunks, file_type FROM files WHERE file_md5 = ?",
            (file_md5,),
        ).fetchone()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE files SET file_type = ? WHERE file_md5 = ?",
                (file_type, file_md5),
            )
            if old_row:
                self._record_operation(
                    "update",
                    f"修改文件类型: {old_row['file_type']} → {file_type}",
                    file_md5, old_row["file_name"], old_row["total_chunks"], "success",
                )
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            logger.exception("设置 file_type 失败: %s", file_md5[:16])
            raise

    def set_file_enabled(self, file_md5: str, enabled: bool) -> None:
        old_row = self._conn.execute(
            "SELECT file_name, total_chunks FROM files WHERE file_md5 = ?",
            (file_md5,),
        ).fetchone()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE files SET is_enabled = ? WHERE file_md5 = ?",
                (1 if enabled else 0, file_md5),
            )
            if old_row:
                self._record_operation(
                    "update",
                    "启用文档" if enabled else "禁用文档",
                    file_md5, old_row["file_name"], old_row["total_chunks"], "success",
                )
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            logger.exception("设置 is_enabled 失败: %s", file_md5[:16])
            raise

    def set_group_enabled(self, group_name: str, enabled: bool) -> None:
        val = 1 if enabled else 0
        g = group_name.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
        rows = self._conn.execute(
            "SELECT file_md5, file_name, total_chunks FROM files WHERE "
            "groups = ?"
            " OR groups LIKE ? ESCAPE '\\'"
            " OR groups LIKE ? ESCAPE '\\'"
            " OR groups LIKE ? ESCAPE '\\'",
            (group_name, f"{g}|%", f"%|{g}|%", f"%|{g}"),
        ).fetchall()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE files SET is_enabled = ? WHERE "
                "groups = ?"
                " OR groups LIKE ? ESCAPE '\\'"
                " OR groups LIKE ? ESCAPE '\\'"
                " OR groups LIKE ? ESCAPE '\\'",
                (val, group_name, f"{g}|%", f"%|{g}|%", f"%|{g}"),
            )
            detail = f"启用分组: {group_name}" if enabled else f"禁用分组: {group_name}"
            for row in rows:
                self._record_operation(
                    "update", detail,
                    row["file_md5"], row["file_name"], row["total_chunks"], "success",
                )
            self._conn.commit()
        except sqlite3.Error:
            self._conn.rollback()
            logger.exception("设置 group enabled 失败: %s", group_name)
            raise

    # ── 操作历史 ──────────────────────────────────────────

    def _record_operation(
        self,
        op_type: str,
        op_detail: str,
        file_md5: str,
        file_name: str,
        chunk_count: int = 0,
        op_result: str = "success",
    ) -> None:
        """写入一条操作记录到 operation_history 表，并执行滚动清理。

        注意：不执行 commit——由调用方在其事务内统一提交，确保
        业务数据变更与审计记录原子落盘（滚动清理同样包含在该事务内）。
        """
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO operation_history "
            "(op_type, op_detail, file_name, file_md5, chunk_count, op_result, op_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (op_type, op_detail, file_name, file_md5, chunk_count, op_result, now),
        )
        # 滚动清理：精确保留最新 N 条（按 id 第 N 新的值为界，AUTOINCREMENT 有
        # 空洞时也不会多删）。正常状态每次删 0~1 行，PK B-tree 上开销可忽略。
        cur = self._conn.execute(
            "DELETE FROM operation_history WHERE id < ("
            "  SELECT MIN(id) FROM ("
            "    SELECT id FROM operation_history ORDER BY id DESC LIMIT ?))",
            (self._op_max_rows,),
        )
        if cur.rowcount > 0:
            logger.info(
                "操作历史滚动清理完成: 删除最早 %d 条 (上限 %d)",
                cur.rowcount, self._op_max_rows,
            )

    def list_operations(
        self,
        limit: int = 50,
        from_id: Optional[int] = None,
        to_id: Optional[int] = None,
    ) -> List[Dict]:
        """返回操作记录（按 op_time 倒序）。

        不传 ID 范围时返回最近 N 条；
        传入 from_id / to_id 时返回 ID 范围内的记录（利用 PK 索引）。
        """
        if from_id is not None and to_id is not None:
            rows = self._conn.execute(
                "SELECT id, op_type, op_detail, file_name, file_md5, "
                "chunk_count, op_result, op_time "
                "FROM operation_history "
                "WHERE id BETWEEN ? AND ? "
                "ORDER BY id DESC LIMIT ?",
                (from_id, to_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, op_type, op_detail, file_name, file_md5, "
                "chunk_count, op_result, op_time "
                "FROM operation_history ORDER BY op_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_operation_stats(self) -> Dict:
        """
        返回操作历史统计信息。

        Returns:
            {"min_id": int, "max_id": int, "total": int}，空表时三项均为 0。
            MIN/MAX 走 PK B-tree 端点查询；COUNT 为全表扫描，
            但行数被滚动上限封顶，开销可忽略。
        """
        row = self._conn.execute(
            "SELECT MIN(id), MAX(id), COUNT(*) FROM operation_history"
        ).fetchone()
        return {"min_id": row[0] or 0, "max_id": row[1] or 0, "total": row[2] or 0}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ═══════════════════════════════════════════════════════════════
# DocManager — 业务层统一入口
# ═══════════════════════════════════════════════════════════════

class DocManager:
    """文档数据管理器。业务层操作文档的唯一入口，内部协调 SQLite + LanceDB。"""

    def __init__(self, config: Config = Config()):
        self._meta = _DocMetaStore(config)
        self._chunks = _ChunkStore(config)

    # ── 写入 ──────────────────────────────────────────────

    def add_document_meta(
        self,
        file_md5: str,
        file_name: str,
        file_path: str,
        file_size: int,
        chunks: List[Dict],
        file_type: str = "",
        groups: str = _DEFAULT_GROUP,
        tags: str = "",
        lancedb_result: Optional[Dict] = None,
    ) -> None:
        """一次性写入 SQLite 的 files + chunks 记录，并记录操作历史。"""
        self._meta.add_document_meta(
            file_md5, file_name, file_path, file_size,
            chunks, file_type, groups, tags,
            lancedb_result=lancedb_result,
        )

    def add_chunks(
        self, chunks: List[Dict], embeddings: np.ndarray
    ) -> Dict:
        """写入向量到 LanceDB。"""
        return self._chunks.add_chunks(chunks, embeddings)

    # ── 删除 ──────────────────────────────────────────────

    def delete_document(self, file_md5: str) -> int:
        """删除文档的全部数据：SQLite 元数据 + LanceDB 向量。"""
        count = self._meta.delete_document(file_md5)
        self._chunks.delete_by_doc_id(file_md5)
        return count

    # ── 查询 ──────────────────────────────────────────────

    def list_files(self) -> Dict[str, List[Dict]]:
        return self._meta.list_files_grouped()

    def list_enabled_doc_ids(self) -> List[str]:
        return self._meta.list_enabled_doc_ids()

    def has_any_docs(self) -> bool:
        """SQLite 中是否有任何文件记录（用于判断是否需要启用文档过滤）。"""
        return self._meta.has_any_docs()

    def has_document(self, file_md5: str) -> bool:
        """检查指定 MD5 的文档是否已完整索引。"""
        return self._meta.has_document(file_md5)

    def get_overview(self) -> dict:
        """返回知识库概览统计。"""
        return self._meta.get_overview()

    def list_operations(
        self, limit: int = 50,
        from_id: Optional[int] = None, to_id: Optional[int] = None,
    ) -> List[Dict]:
        """返回操作记录。不传范围返回最近 N 条，传范围按 ID 过滤。"""
        return self._meta.list_operations(limit, from_id=from_id, to_id=to_id)

    def get_operation_stats(self) -> Dict:
        """
        返回操作历史统计信息。

        Returns:
            {"min_id": int, "max_id": int, "total": int}，空表时三项均为 0。
        """
        return self._meta.get_operation_stats()

    # ── 更新 ──────────────────────────────────────────────

    def set_groups(self, file_md5: str, groups: str) -> None:
        self._meta.set_groups(file_md5, groups)

    def set_tags(self, file_md5: str, tags: str) -> None:
        self._meta.set_tags(file_md5, tags)

    def set_file_type(self, file_md5: str, file_type: str) -> None:
        self._meta.set_file_type(file_md5, file_type)

    def set_file_enabled(self, file_md5: str, enabled: bool) -> None:
        self._meta.set_file_enabled(file_md5, enabled)

    def set_group_enabled(self, group_name: str, enabled: bool) -> None:
        self._meta.set_group_enabled(group_name, enabled)

    def close(self) -> None:
        # ⚠️ NOTICE: 当前 DocManager 作为 @st.cache_resource 单例全程存活至进程退出，
        # 操作系统会自动回收 SQLite 连接，暂无调用方。如果未来增加了销毁/重建 DocManager
        # 实例的业务逻辑，必须在实例销毁前调用 close()。
        self._meta.close()
