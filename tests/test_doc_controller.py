"""测试 DocController — 索引管线编排与文档管理转发。"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from _types.retrieval_types import ContinueChoice


# ── 辅助函数 ──────────────────────────────────────────────


def _make_index_result(**overrides):
    """构造 pipeline.index_document() 返回的 dict。"""
    result = {
        "success": True,
        "skipped": False,
        "doc_id": "abc123",
        "doc_name": "test.pdf",
        "num_chunks": 15,
        "token_usage": {"input": 5000, "output": 2000},
        "error": None,
    }
    result.update(overrides)
    return result


def _make_mock_pipeline():
    """构造 mock pipeline。"""
    mock = MagicMock()
    mock.index_document.return_value = _make_index_result()
    mock.list_documents.return_value = {}
    mock.delete_document.return_value = 1
    return mock


@pytest.fixture
def mock_pipeline():
    """mock pipeline 实例。"""
    return _make_mock_pipeline()


@pytest.fixture
def controller(mock_pipeline):
    """注入 mock pipeline 的 DocController。"""
    from controllers.doc_controller import DocController
    yield DocController(mock_pipeline)


# ── _validate ────────────────────────────────────────────


class TestValidate:
    """DocController._validate() 各种边界情况。"""

    def test_file_not_found(self, mock_pipeline):
        """不存在的文件路径返回错误信息。"""
        from controllers.doc_controller import DocController
        ctrl = DocController(mock_pipeline)
        error = ctrl._validate("/nonexistent/path/file.pdf")
        assert error is not None
        assert "不存在" in error

    def test_unsupported_extension(self, mock_pipeline):
        """非 .pdf / .txt 后缀文件返回格式不支持错误。"""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"hello world")
            docx_path = f.name
        try:
            from controllers.doc_controller import DocController
            ctrl = DocController(mock_pipeline)
            error = ctrl._validate(docx_path)
            assert error is not None
            assert "格式不支持" in error
        finally:
            os.unlink(docx_path)

    def test_txt_passes_validation(self, mock_pipeline):
        """.txt 文件通过校验。"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"chapter one\nsome content")
            txt_path = f.name
        try:
            from controllers.doc_controller import DocController
            ctrl = DocController(mock_pipeline)
            error = ctrl._validate(txt_path)
            assert error is None
        finally:
            os.unlink(txt_path)

    def test_empty_pdf_file(self, mock_pipeline):
        """空文件（0 bytes）返回错误信息。"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name
        try:
            from controllers.doc_controller import DocController
            ctrl = DocController(mock_pipeline)
            error = ctrl._validate(pdf_path)
            assert error is not None
            assert "为空" in error
        finally:
            os.unlink(pdf_path)

    def test_file_too_large(self, mock_pipeline):
        """超过 max_pdf_size_mb 上限的文件返回错误信息。"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-x")
            pdf_path = f.name
        try:
            from controllers.doc_controller import DocController, Config
            ctrl = DocController(mock_pipeline)
            with patch("os.path.getsize", return_value=201 * 1024 * 1024):
                error = ctrl._validate(pdf_path)
            assert error is not None
            assert "过大" in error
        finally:
            os.unlink(pdf_path)

    def test_valid_pdf_passes_validation(self, mock_pipeline):
        """合法的 PDF 文件（%PDF- 头、非空、.pdf 后缀）返回 None。"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nsome content")
            pdf_path = f.name
        try:
            from controllers.doc_controller import DocController
            ctrl = DocController(mock_pipeline)
            error = ctrl._validate(pdf_path)
            assert error is None
        finally:
            os.unlink(pdf_path)

    def test_bad_pdf_header(self, mock_pipeline):
        """文件头不以 %PDF- 开头返回错误信息。"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"NOT A PDF HEADER")
            pdf_path = f.name
        try:
            from controllers.doc_controller import DocController
            ctrl = DocController(mock_pipeline)
            error = ctrl._validate(pdf_path)
            assert error is not None
            assert "PDF 格式" in error
        finally:
            os.unlink(pdf_path)

    def test_oserror_on_getsize_returns_error(self, mock_pipeline):
        """os.path.getsize 抛出 OSError 时返回错误。"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nsome content")
            pdf_path = f.name
        try:
            from controllers.doc_controller import DocController
            ctrl = DocController(mock_pipeline)
            with patch("os.path.getsize", side_effect=OSError("permission denied")):
                error = ctrl._validate(pdf_path)
            assert error is not None
            assert "无法读取文件" in error
        finally:
            os.unlink(pdf_path)


# ── execute_index ────────────────────────────────────────


class TestExecuteIndex:
    """DocController.execute_index()。"""

    def test_validation_failure_returns_error_dict(self, controller):
        """校验失败时不调用 pipeline，直接返回错误。"""
        result = controller.execute_index("/nonexistent/file.pdf")
        assert result["success"] is False
        assert result["skipped"] is False
        assert "不存在" in result["error"]
        assert result["doc_id"] == ""
        assert result["num_chunks"] == 0

    def test_validation_failure_extracts_doc_name(self, controller):
        """校验失败时仍返回 doc_name（从路径中提取）。"""
        result = controller.execute_index("/some/path/report.pdf")
        assert result["doc_name"] == "report.pdf"

    def test_success_returns_pipeline_result(self, controller, mock_pipeline):
        """成功索引时返回标准化结果。"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nfake content")
            pdf_path = f.name
        try:
            result = controller.execute_index(pdf_path)
            mock_pipeline.index_document.assert_called_once_with(pdf_path, on_progress=None)
            assert result["success"] is True
            assert result["doc_name"] == "test.pdf"
            assert result["num_chunks"] == 15
            assert result["token_usage"] == {"input": 5000, "output": 2000}
        finally:
            os.unlink(pdf_path)

    def test_skipped_document(self, controller, mock_pipeline):
        """MD5 已存在的文档返回 skipped。"""
        mock_pipeline.index_document.return_value = _make_index_result(
            success=False, skipped=True, doc_name="dup.pdf",
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nfake content")
            pdf_path = f.name
        try:
            result = controller.execute_index(pdf_path)
            assert result["success"] is False
            assert result["skipped"] is True
            assert result["doc_name"] == "dup.pdf"
        finally:
            os.unlink(pdf_path)

    def test_pipeline_exception_returns_error_dict(self, controller, mock_pipeline):
        """pipeline 抛出异常时返回错误，不向上传播。"""
        mock_pipeline.index_document.side_effect = RuntimeError("OCR 引擎崩溃")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nfake content")
            pdf_path = f.name
        try:
            result = controller.execute_index(pdf_path)
            assert result["success"] is False
            assert result["skipped"] is False
            assert "OCR 引擎崩溃" in result["error"]
            assert result["doc_id"] == ""
        finally:
            os.unlink(pdf_path)

    def test_pipeline_exception_extracts_doc_name(self, controller, mock_pipeline):
        """pipeline 异常时仍返回 doc_name。"""
        mock_pipeline.index_document.side_effect = RuntimeError("boom")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nfake content")
            pdf_path = f.name
        try:
            result = controller.execute_index(pdf_path)
            assert result["doc_name"] == os.path.basename(pdf_path)
        finally:
            os.unlink(pdf_path)


# ── 文档管理转发 ─────────────────────────────────────────


class TestDocManagementDelegation:
    """DocController 文档管理方法正确转发给 pipeline。"""

    def test_list_documents_delegates(self, controller, mock_pipeline):
        """list_documents 转发给 pipeline.list_documents()。"""
        controller.list_documents()
        mock_pipeline.list_documents.assert_called_once()

    def test_set_document_file_type_delegates(self, controller, mock_pipeline):
        """set_document_file_type 转发给 pipeline。"""
        controller.set_document_file_type("abc", "年报")
        mock_pipeline.set_document_file_type.assert_called_once_with("abc", "年报")

    def test_set_document_groups_delegates(self, controller, mock_pipeline):
        """set_document_groups 转发给 pipeline。"""
        controller.set_document_groups("abc", "财务|审计")
        mock_pipeline.set_document_groups.assert_called_once_with("abc", "财务|审计")

    def test_set_document_tags_delegates(self, controller, mock_pipeline):
        """set_document_tags 转发给 pipeline。"""
        controller.set_document_tags("abc", "重点|Q4")
        mock_pipeline.set_document_tags.assert_called_once_with("abc", "重点|Q4")

    def test_set_document_enabled_delegates(self, controller, mock_pipeline):
        """set_document_enabled 转发给 pipeline。"""
        controller.set_document_enabled("abc", False)
        mock_pipeline.set_document_enabled.assert_called_once_with("abc", False)

    def test_set_group_enabled_delegates(self, controller, mock_pipeline):
        """set_group_enabled 转发给 pipeline。"""
        controller.set_group_enabled("财务", True)
        mock_pipeline.set_group_enabled.assert_called_once_with("财务", True)

    def test_delete_document_delegates(self, controller, mock_pipeline):
        """delete_document 转发给 pipeline 并返回结果。"""
        result = controller.delete_document("abc")
        mock_pipeline.delete_document.assert_called_once_with("abc")
        assert result == 1
