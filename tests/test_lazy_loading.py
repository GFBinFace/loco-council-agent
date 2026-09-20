"""测试模型懒加载行为。

原则是"业务没启动就不要加载"——模型不该在对象构造时就被载入。
这里锁定两条最容易在后续改动中回退的保证：

1. Reranker 的 CrossEncoder：只在首次访问 .model 时才构造。
2. ocr_engine 对 paddle 生态的导入：必须推迟到真正用 OCR 时
   （该生态仅导入就占约 400MB 常驻内存，且只服务 OCR）。
"""

import os
import subprocess
import sys
from unittest.mock import patch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRerankerLazyLoading:
    """Reranker 的 CrossEncoder 只在首次访问时才构造。"""

    def test_init_does_not_construct_model(self):
        with patch("services.retrieval.reranker.CrossEncoder") as mock_ce:
            from services.retrieval.reranker import Reranker
            Reranker()
            assert mock_ce.call_count == 0

    def test_first_access_constructs_model_with_configured_name(self):
        with patch("services.retrieval.reranker.CrossEncoder") as mock_ce:
            from services.retrieval.reranker import Reranker
            reranker = Reranker()
            model = reranker.model
            assert mock_ce.call_count == 1
            assert model is mock_ce.return_value
            assert mock_ce.call_args[0][0] == reranker.config.rerank_model

    def test_second_access_reuses_the_same_instance(self):
        with patch("services.retrieval.reranker.CrossEncoder") as mock_ce:
            from services.retrieval.reranker import Reranker
            reranker = Reranker()
            assert reranker.model is reranker.model
            assert mock_ce.call_count == 1


class TestOcrEngineLazyImport:
    """导入 ocr_engine 不应连带导入 paddle 生态。

    这条断言必须在**新进程**里做：同进程内其他测试可能已经把它导入了，
    那样断言会永远为真，测不出回退。
    """

    def test_importing_ocr_engine_does_not_import_paddleocr(self):
        code = (
            "import sys;"
            "import services.indexing.ocr_engine;"
            "loaded = [m for m in ('paddleocr', 'paddlex', 'paddle')"
            " if m in sys.modules];"
            "assert not loaded, f'paddle 生态被提前导入: {loaded}'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, result.stderr
