"""Streamlit 前端面板冒烟测试。

用 AppTest 驱动真实面板（Controller 为 MagicMock，见 smoke_app.py），
覆盖回调化改造后的关键交互路径：组开关同步、编辑/保存/取消、
删除确认、搜索发送、历史面板开合、空查询边界。

运行方式（项目根目录）: python -m pytest tests/streamlit
"""

import os
from typing import List, Tuple

from streamlit.testing.v1 import AppTest

from ui.session_state import Keys

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_app.py")
_MD5 = "md5aaa"


# ── 辅助函数 ──────────────────────────────────────────────


def _run_app() -> AppTest:
    """启动冒烟 app 并完成首轮渲染。"""
    at = AppTest.from_file(_APP, default_timeout=15)
    at.run()
    assert not at.exception
    return at


def _calls(at: AppTest) -> List[Tuple]:
    """读取 Controller 调用记录。"""
    return list(at.session_state["ctrl_calls"])


# ── 文档面板 ──────────────────────────────────────────────


def test_doc_panel_group_disable_syncs_toggle_and_calls_controller() -> None:
    at = _run_app()
    at.button(key="group_disable_default").click().run()
    assert not at.exception
    assert at.toggle(key=f"enabled_{_MD5}").value is False
    assert ("set_group_enabled", ("default", False)) in _calls(at)


def test_doc_panel_group_enable_syncs_toggle_on() -> None:
    at = _run_app()
    at.button(key="group_disable_default").click().run()
    at.button(key="group_enable_default").click().run()
    assert at.toggle(key=f"enabled_{_MD5}").value is True
    assert ("set_group_enabled", ("default", True)) in _calls(at)


def test_doc_panel_toggle_flip_writes_db() -> None:
    at = _run_app()
    at.toggle(key=f"enabled_{_MD5}").set_value(False).run()
    assert ("set_document_enabled", (_MD5, False)) in _calls(at)


def test_doc_panel_edit_save_updates_groups_and_exits_edit_mode() -> None:
    at = _run_app()
    at.button(key=f"edit_btn_{_MD5}").click().run()
    at.text_input(key=f"edit_groups_{_MD5}").set_value("新分组").run()
    at.button(key=f"save_{_MD5}").click().run()
    assert at.session_state[f"edit_mode_{_MD5}"] is False
    assert ("set_document_groups", (_MD5, "新分组")) in _calls(at)
    # 未填写的字段不应写库（空字符串视为不修改）
    assert not [c for c in _calls(at) if c[0] == "set_document_tags"]


def test_doc_panel_edit_cancel_exits_without_write() -> None:
    at = _run_app()
    at.button(key=f"edit_btn_{_MD5}").click().run()
    at.button(key=f"cancel_{_MD5}").click().run()
    assert at.session_state[f"edit_mode_{_MD5}"] is False
    assert not [c for c in _calls(at) if c[0].startswith("set_document_")]


def test_doc_panel_delete_confirm_calls_controller_and_toasts() -> None:
    at = _run_app()
    at.button(key=f"yes_del_{_MD5}").click().run()
    assert ("delete_document", (_MD5,)) in _calls(at)
    assert any("已删除" in str(t.value) for t in at.toast)


# ── 操作历史弹窗 ──────────────────────────────────────────


def test_doc_panel_op_history_open_resets_range_to_latest() -> None:
    # 打开事件重置范围：max_id=60, page_size=50 → 默认范围 11–60
    at = _run_app()
    at.button(key="op_history_btn").click().run()
    assert not at.exception
    assert at.session_state[Keys.OP_FROM_ID] == 11
    assert at.session_state[Keys.OP_TO_ID] == 60
    # 弹窗内容已渲染：统计行含库存全貌，记录行带 #ID 前缀、&nbsp; 间距和全格式时间
    captions = [str(c.value) for c in at.caption]
    assert any("库中现存 60 条（ID 1–60）" in c for c in captions)
    assert any(c.startswith("#60&nbsp;") and "2026-07-15 12:19" in c for c in captions)


def test_doc_panel_op_range_query_over_page_size_sets_error() -> None:
    # 边界：范围跨度超过 page_size（50）→ 回调写入错误信息，不更新范围
    at = _run_app()
    at.button(key="op_history_btn").click().run()
    at.number_input(key="op_from_input").set_value(1)
    at.number_input(key="op_to_input").set_value(60)
    at.button(key="op_range_btn").click().run()
    assert at.session_state[Keys.OP_RANGE_ERROR] == "一次最多显示 50 条记录"
    assert at.session_state[Keys.OP_FROM_ID] == 11  # 范围保持打开时的默认值


def test_doc_panel_op_range_query_valid_locks_range() -> None:
    at = _run_app()
    at.button(key="op_history_btn").click().run()
    at.number_input(key="op_from_input").set_value(20)
    at.number_input(key="op_to_input").set_value(30)
    at.button(key="op_range_btn").click().run()
    assert at.session_state[Keys.OP_RANGE_ERROR] == ""
    assert at.session_state[Keys.OP_FROM_ID] == 20
    assert at.session_state[Keys.OP_TO_ID] == 30


# ── 搜索面板 ──────────────────────────────────────────────


def test_search_panel_send_appends_answer_and_clears_input() -> None:
    at = _run_app()
    at.text_input(key="search_q_0").set_value("测试问题").run()
    at.button(key="send_btn").click().run()
    assert not at.exception
    assert len(at.session_state[Keys.CHAT_MESSAGES]) == 1
    assert "回答完成" in at.session_state[Keys.SEARCH_STATUS_TEXT]
    # 回答气泡下渲染单轮 token 消耗
    captions = [str(c.value) for c in at.caption]
    assert any("🔢 tokens: 1,234 / 567" in c for c in captions)
    # 计数器递增 → 输入框换新 key 实现清空
    assert at.session_state["search_q_counter"] == 1
    assert any(
        c[0] == "execute_search" and c[1][0] == "测试问题" for c in _calls(at)
    )


def test_search_panel_empty_query_send_is_noop() -> None:
    # 边界：空查询——发送回调应直接返回，不触发检索
    at = _run_app()
    at.button(key="send_btn").click().run()
    assert len(at.session_state[Keys.CHAT_MESSAGES]) == 0
    assert not [c for c in _calls(at) if c[0] == "execute_search"]


def test_search_panel_history_dialog_lists_sessions() -> None:
    at = _run_app()
    at.button(key="history_btn").click().run()
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any("共 1 次对话" in c for c in captions)


def test_search_panel_history_restore_loads_conversation() -> None:
    at = _run_app()
    at.button(key="history_btn").click().run()
    # 点击恢复 → 回调设置 _history_dialog_close 标志 + 恢复会话
    at.button(key="hist_sess-001").click().run()
    # dialog 检测到标志 → st.rerun() 完整重跑 → 弹窗关闭，
    # 会话数据已在 dialog 重跑完成后落地
    assert len(at.session_state[Keys.CHAT_MESSAGES]) == 1
    assert at.session_state[Keys.CURRENT_SESSION_ID] == "sess-001"
    assert "已恢复历史对话" in at.session_state[Keys.SEARCH_STATUS_TEXT]


def test_search_panel_new_dialog_resets_state() -> None:
    at = _run_app()
    at.text_input(key="search_q_0").set_value("测试问题").run()
    at.button(key="send_btn").click().run()
    at.button(key="new_dialog_btn").click().run()
    assert at.session_state[Keys.CHAT_MESSAGES] == []
    assert at.session_state[Keys.CURRENT_SESSION_ID] is None
    assert at.session_state[Keys.SEARCH_STATE] == "idle"


# ── 对话导出 ──────────────────────────────────────────────


def test_format_source_locator_pdf_pages_only() -> None:
    from ui.components import format_source_locator
    s = {"page_nums": [3, 4], "chapter_title": "", "chunk_index": 5}
    assert format_source_locator(s) == "页码 [3, 4] — chunk #5"


def test_format_source_locator_txt_chapter_only() -> None:
    from ui.components import format_source_locator
    s = {"page_nums": [], "chapter_title": "天界卷 第三百二十九章 终之章(上)",
         "chunk_index": 189}
    assert format_source_locator(s) == "天界卷 第三百二十九章 终之章(上) — chunk #189"


def test_format_source_locator_neither_shows_chunk_only() -> None:
    # 等宽分块 TXT：既无页码也无章节 → 只显示 chunk #，不出现"页码 []"
    from ui.components import format_source_locator
    s = {"page_nums": [], "chapter_title": "", "chunk_index": 189}
    assert format_source_locator(s) == "chunk #189"
    assert "页码" not in format_source_locator(s)


def test_build_conversation_markdown_full_turn() -> None:
    from ui.components import build_conversation_markdown
    messages = [{
        "query": "净利润是多少？",
        "answer": "净利润为 100 万元。",
        "sources": [{"doc_name": "年报.pdf", "page_nums": [3],
                     "chunk_index": 5, "llm_score": 8}],
        "token_usage": {"input": 1000, "output": 200},
    }]
    md = build_conversation_markdown(messages, "sess-abc-123")
    assert "# 对话导出" in md
    assert "sess-abc-123" in md
    assert "净利润是多少？" in md
    assert "净利润为 100 万元。" in md
    assert "年报.pdf — 页码 [3] — chunk #5 — LLM得分=8" in md
    assert "1000/200" in md


def test_build_conversation_markdown_empty_returns_header_only() -> None:
    from ui.components import build_conversation_markdown
    md = build_conversation_markdown([], None)
    assert "轮数: 0" in md
    assert "（未保存）" in md
