"""前端冒烟测试用 Streamlit app——用 MagicMock Controller 渲染两个真实面板。

Controller 关键方法的每次调用都记录到 session_state["ctrl_calls"]，
供 AppTest 驱动侧断言 "UI 交互确实触发了正确的后端调用"。
本文件不以 test_ 开头，pytest 不会收集。
"""

import os
import sys
from typing import Any, Optional
from unittest.mock import MagicMock

# AppTest 以脚本方式执行本文件，需自行保证项目根在 sys.path 中
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from _types.retrieval_types import SearchResult
from ui.doc_mgmt_panel import render_doc_mgmt_panel
from ui.search_panel import render_search_panel
from ui.session_state import init_session_state

init_session_state()
# ctrl_calls 必须在 init_session_state 之后以 "not in" 方式初始化——
# st.rerun() 会触发完整脚本重跑（如 dialog 内恢复按钮），
# setdefault 在重跑时不会保留旧值。
if "ctrl_calls" not in st.session_state:
    st.session_state["ctrl_calls"] = []


def _recorder(name: str, result: Optional[Any] = None):
    """
    构造既记录调用又返回预设结果的 side_effect 函数。

    Args:
        name: 记录用的方法名
        result: 调用返回值

    Returns:
        可作为 MagicMock side_effect 的函数。
    """
    def _record(*args: Any, **kwargs: Any) -> Any:
        st.session_state["ctrl_calls"].append((name, args))
        return result
    return _record


# ── 文档面板 mock ──
doc_ctrl = MagicMock()
doc_ctrl.list_documents.return_value = {
    "default": [{
        "file_md5": "md5aaa", "file_name": "样例.pdf", "file_type": "pdf",
        "groups": "default", "tags": "", "file_size": 2048,
        "total_chunks": 6, "indexed_at": "2026-07-15T00:00:00", "is_enabled": 1,
    }],
}
doc_ctrl.list_operations.return_value = [
    {"id": 60, "op_type": "update", "op_detail": "禁用分组: default",
     "file_name": "样例.pdf", "file_md5": "md5aaa", "chunk_count": 6,
     "op_result": "success", "op_time": "2026-07-15T12:19:56"},
    {"id": 59, "op_type": "index", "op_detail": "新建索引",
     "file_name": "样例.pdf", "file_md5": "md5aaa", "chunk_count": 6,
     "op_result": "success", "op_time": "2026-07-15T12:10:00"},
]
doc_ctrl.get_operation_stats.return_value = {"min_id": 1, "max_id": 60, "total": 60}
doc_ctrl.get_overview.return_value = {
    "total_docs": 1, "enabled_docs": 1, "total_chunks": 6, "total_chars": 5000,
    "latest_index": "2026-07-15T00:00:00", "embedding_dim": 1024, "storage_bytes": 4096,
}
doc_ctrl.set_group_enabled.side_effect = _recorder("set_group_enabled")
doc_ctrl.set_document_enabled.side_effect = _recorder("set_document_enabled")
doc_ctrl.set_document_groups.side_effect = _recorder("set_document_groups")
doc_ctrl.set_document_tags.side_effect = _recorder("set_document_tags")
doc_ctrl.set_document_file_type.side_effect = _recorder("set_document_file_type")
doc_ctrl.delete_document.side_effect = _recorder("delete_document", 6)

# ── 搜索面板 mock ──
search_ctrl = MagicMock()
search_ctrl.list_sessions.return_value = [{
    "session_id": "sess-001", "updated_at": "2026-07-15T12:00:00",
    "title": "样例查询", "turn_count": 1,
    "total_input_tokens": 100, "total_output_tokens": 50,
}]
search_ctrl.restore_session.side_effect = _recorder(
    "restore_session",
    {
        "messages": [{
            "query": "历史问题", "answer": "历史回答", "sources": [],
            "input_tokens": 100, "output_tokens": 50,
        }],
    },
)
search_ctrl.execute_search.side_effect = _recorder(
    "execute_search",
    (
        SearchResult(status="success", query="q", answer="模拟回答",
                     sources=[], source_count=0,
                     token_usage={"input": 1234, "output": 567}),
        "sid-123",
    ),
)

render_doc_mgmt_panel(doc_ctrl)
render_search_panel(search_ctrl)
