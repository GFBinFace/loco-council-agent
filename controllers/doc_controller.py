"""文档区服务调度层。

DocController 是前端左侧文档面板与 Pipeline 之间的编排器，
负责索引管线调度和文档元数据管理。
"""

import os
from typing import Callable, Optional

from config import Config


class DocController:
    """文档区编排器——索引管线 + 文档 CRUD + 分组管理。

    Pipeline 由启动层注入（依赖注入），Controller 不自行获取全局实例。
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline

    # ── 文档管理（转发给 Pipeline → DocManager）────────────

    def list_documents(self):
        """获取按分组建档的文档列表。"""
        return self._pipeline.list_documents()

    def set_document_file_type(self, file_md5: str, file_type: str) -> None:
        """更新文档的 file_type 字段。"""
        self._pipeline.set_document_file_type(file_md5, file_type)

    def set_document_groups(self, file_md5: str, groups: str) -> None:
        """更新文档的分组。"""
        self._pipeline.set_document_groups(file_md5, groups)

    def set_document_tags(self, file_md5: str, tags: str) -> None:
        """更新文档的标签。"""
        self._pipeline.set_document_tags(file_md5, tags)

    def set_document_enabled(self, file_md5: str, enabled: bool) -> None:
        """启用/禁用单个文档。"""
        self._pipeline.set_document_enabled(file_md5, enabled)

    def set_group_enabled(self, group_name: str, enabled: bool) -> None:
        """启用/禁用整个分组。"""
        self._pipeline.set_group_enabled(group_name, enabled)

    def delete_document(self, file_md5: str) -> int:
        """删除文档的元数据和向量数据。"""
        return self._pipeline.delete_document(file_md5)

    # ── 索引管线 ─────────────────────────────────────────

    def execute_index(
        self,
        file_path: str,
        on_progress: Optional[Callable] = None,
    ) -> dict:
        """执行索引管线。

        Args:
            file_path: PDF 文件的本地完整路径
            on_progress: 进度回调，签名为 (status_line: str | None, log_line: str | None)

        Returns:
            {
                "success": bool,
                "skipped": bool,
                "doc_id": str,
                "doc_name": str,
                "num_chunks": int,
                "token_usage": {"input": N, "output": N} | None,
                "error": str | None,
            }
        """
        error = self._validate(file_path)
        if error is not None:
            if on_progress:
                on_progress(None, f"文件校验失败: {error}")
            return {
                "success": False,
                "skipped": False,
                "doc_id": "",
                "doc_name": os.path.basename(file_path),
                "num_chunks": 0,
                "token_usage": None,
                "error": error,
            }

        try:
            result = self._pipeline.index_document(file_path, on_progress=on_progress)
        except Exception as exc:
            # 致命错误：Controller 统一汇报
            if on_progress:
                doc_name = os.path.basename(file_path)
                on_progress(f"❌ 索引失败：{doc_name}", f"管线异常: {exc}")
            return {
                "success": False,
                "skipped": False,
                "doc_id": "",
                "doc_name": os.path.basename(file_path),
                "num_chunks": 0,
                "token_usage": None,
                "error": str(exc),
            }

        return {
            "success": bool(result.get("success")),
            "skipped": bool(result.get("skipped")),
            "doc_id": result.get("doc_id", ""),
            "doc_name": result.get("doc_name", os.path.basename(file_path)),
            "num_chunks": result.get("num_chunks", 0),
            "token_usage": result.get("token_usage"),
            "error": result.get("error"),
        }

    def list_operations(
        self, limit: int = 50,
        from_id: Optional[int] = None, to_id: Optional[int] = None,
    ):
        """返回操作记录。不传范围返回最近 N 条，传范围按 ID 过滤。"""
        return self._pipeline.doc_manager.list_operations(
            limit, from_id=from_id, to_id=to_id,
        )

    def get_operation_stats(self) -> dict:
        """
        返回操作历史统计信息。

        Returns:
            {"min_id": int, "max_id": int, "total": int}，空表时三项均为 0。
        """
        return self._pipeline.doc_manager.get_operation_stats()

    def get_overview(self) -> dict:
        """
        返回知识库概览统计。

        Returns:
            {
                "total_docs": int,       # 已索引文档数
                "enabled_docs": int,      # 激活文档数
                "total_chunks": int,      # 知识块总数
                "total_chars": int,       # 文本总量（字符数），None 表示查询异常
                "latest_index": str,      # 最近索引时间，None 表示无
                "embedding_dim": int,     # 向量维度
                "storage_bytes": int,     # 存储占用（字节数），-1 表示计算失败
            }
        """
        overview = self._pipeline.doc_manager.get_overview()
        overview["embedding_dim"] = Config().embedding_dim
        overview["storage_bytes"] = self._get_dir_size(
            Config().sqlite_dir, Config().lance_db_dir,
        )
        return overview

    # ── 内部 ──────────────────────────────────────────────

    @staticmethod
    def _get_dir_size(*dirs: str) -> int:
        """递归计算一个或多个目录的总字节数。失败时返回 -1。"""
        total = 0
        for d in dirs:
            if not os.path.isdir(d):
                return -1
            for dirpath, _, filenames in os.walk(d):
                for f in filenames:
                    filepath = os.path.join(dirpath, f)
                    try:
                        total += os.path.getsize(filepath)
                    except OSError:
                        pass
        return total

    @staticmethod
    def _validate(file_path: str) -> Optional[str]:
        """校验输入文件，返回错误信息；合法时返回 None。"""
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            return f"文件不存在: {file_path}"
        
        # 检查文件格式
        ext = file_path.lower()
        if not (ext.endswith(".pdf") or ext.endswith(".txt")):
            return f"文件格式不支持: {os.path.basename(file_path)}（支持 .pdf 和 .txt）"

        # 检查文件大小
        try:
            size = os.path.getsize(file_path)
        except OSError as exc:
            return f"无法读取文件: {exc}"
        if size == 0:
            return "文件为空"
        max_bytes = Config().max_pdf_size_mb * 1024 * 1024
        if size > max_bytes:
            return (
                f"文件过大: {size / 1024 / 1024:.1f}MB"
                f"（上限 {Config().max_pdf_size_mb}MB）"
            )

        # PDF 文件检查文件头；TXT 不检查
        if file_path.lower().endswith(".pdf"):
            try:
                with open(file_path, "rb") as fh:
                    header = fh.read(5)
                if header != b"%PDF-":
                    return f"文件头不是有效的 PDF 格式: {os.path.basename(file_path)}"
            except OSError as exc:
                return f"无法读取文件: {exc}"

        return None
