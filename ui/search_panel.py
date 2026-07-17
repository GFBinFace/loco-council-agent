"""搜索问答面板 — 右侧区域。

状态机：idle → searching → needs_choice → done → idle
"""

import time
from typing import Optional

import streamlit as st
from streamlit.elements.lib.mutable_status_container import StatusContainer

from controllers.search_controller import SearchController
from _types.retrieval_types import ContinueChoice, SearchResult
from ui.components import (
    build_conversation_markdown,
    build_history_context,
    render_divider,
    render_sources,
)
from ui.layout import CONVERSATION_HEIGHT, LOG_HEIGHT
from ui.session_state import Keys, add_chat_message, accumulate_session_tokens, add_progress_log


# ── 按钮回调 ──────────────────────────────────────────────
#
# 统一用 on_click 回调替代 `if st.button(): ... st.rerun()`：
# 回调先于脚本重跑执行，重跑时天然读到新状态，无需 st.rerun()（也避免了
# 中途打断脚本导致后方未渲染控件的状态被清理，如输入框文字丢失）。

def _on_send() -> None:
    """发送按钮回调：锁定查询文本并触发搜索状态机。"""
    # 输入框用递增计数器作 key，发送后换新 key 实现自动清空
    counter = st.session_state.get("search_q_counter", 0)
    query = st.session_state.get(f"search_q_{counter}", "")
    if not query:
        return
    st.session_state[Keys.DO_SEARCH] = True
    st.session_state[Keys.PENDING_QUERY] = query
    st.session_state[Keys.SEARCH_STATE] = "searching"
    st.session_state[Keys.SEARCH_RESULT] = None
    st.session_state[Keys.SEARCH_START_TIME] = time.time()
    st.session_state[Keys.SEARCH_STATUS_TEXT] = "🔍 检索中…"
    st.session_state["search_q_counter"] = counter + 1


def _on_new_dialog() -> None:
    """新对话按钮回调：清空当前会话状态。"""
    st.session_state[Keys.CURRENT_SESSION_ID] = None
    st.session_state[Keys.CHAT_MESSAGES] = []
    st.session_state[Keys.SEARCH_STATE] = "idle"
    st.session_state[Keys.SEARCH_RESULT] = None
    st.session_state[Keys.SEARCH_STATUS_TEXT] = "✅ 新对话已开始"


def _on_continue_choice(choice: ContinueChoice) -> None:
    """分支选择回调：触发 continue_search，改写状态行避免旧文本截流。"""
    st.session_state[Keys.DO_CONTINUE] = True
    st.session_state[Keys.PENDING_CHOICE] = choice
    st.session_state[Keys.SEARCH_STATE] = "searching"
    st.session_state[Keys.SEARCH_RESULT] = None
    st.session_state[Keys.SEARCH_STATUS_TEXT] = "🔍 处理中…"


def _on_restore_session(controller: SearchController, session_id: str) -> None:
    """恢复按钮回调：加载会话 + 设置关闭标志。

    回调中不能调用 st.toast 等渲染元素的函数（dialog fragment 上下文限制），
    故 _restore_from_history 的 toast 已替换为状态行反馈。
    """
    _restore_from_history(controller, session_id)
    st.session_state["_history_dialog_close"] = True


# ── 主渲染函数 ────────────────────────────────────────────

def render_search_panel(controller: SearchController) -> None:
    """渲染右侧搜索问答面板。

    布局：
      状态行 → 双栏（对话区 | 功能键） → 输入行 → 进度日志

    注意：左侧索引是阻塞操作，索引期间本面板不会被渲染（前端整体
    置灰），"索引进行中"的可见信号只能是左栏进度条 + 右栏灰色本身，
    无法在本面板的状态行上做文章。
    """
    status_widget = _render_status_line()
    render_divider()

    col_main, col_sidebar = st.columns([3, 0.8])
    with col_main:
        with st.container(height=CONVERSATION_HEIGHT, border=True):
            _render_search_executor(controller, status_widget)
            _render_conversation_messages()
            _render_choice_area()
    with col_sidebar:
        _render_sidebar(controller)

    # 输入行——输入框 | 发送(宽) | 新对话(窄)
    col_input, col_send, col_new = st.columns([3, 0.5, 0.3])
    with col_input:
        query = st.text_input(
            "查询",
            key=f"search_q_{st.session_state.get('search_q_counter', 0)}",
            label_visibility="collapsed",
            placeholder="输入你的问题…",
        )
    with col_send:
        searching = st.session_state[Keys.SEARCH_STATE] in ("searching",)
        st.button(
            "🚀 发送", key="send_btn", disabled=searching,
            use_container_width=True, on_click=_on_send,
        )
    with col_new:
        st.button(
            "🆕 新对话", key="new_dialog_btn",
            use_container_width=True, on_click=_on_new_dialog,
        )

    _render_progress_log()


# ── 子区域 ────────────────────────────────────────────────

def _render_status_line() -> StatusContainer:
    """渲染状态行（st.status），返回控件供搜索执行器实时更新。

    用 st.status 而非 st.empty + 自绘 HTML：st.empty 的内容在其他面板
    触发新一轮运行时会被前端提前清除（曾致索引期间右栏状态栏消失、
    下方控件上移错位）；st.status 是普通元素，随面板整体置灰保留，
    且原生支持本轮内 update(label, state) 实时刷新。
    """
    text = st.session_state[Keys.SEARCH_STATUS_TEXT]
    if st.session_state[Keys.SEARCH_STATE] == "searching":
        state = "running"
    elif text.startswith("❌"):
        state = "error"
    else:
        state = "complete"
    # 带 key 的容器为 CSS 提供 st-key-search_status_line 类锚点（浅蓝底色见 app.py）
    with st.container(key="search_status_line"):
        return st.status(text, expanded=False, state=state)


def _render_sidebar(controller: SearchController) -> None:
    """侧边功能栏：历史记录弹窗按钮、附带历史上下文开关（上→下）。

    发送和新对话按钮在下方输入行中。
    """
    # dialog 惯用法：弹窗函数必须在渲染流中调用（回调无法渲染 UI），
    # 属于回调模式的合法例外。空库时 toast 提示，不弹空窗。
    if st.button("📋 历史记录", key="history_btn", use_container_width=True):
        if not controller.list_sessions(limit=1):
            st.toast("暂无历史记录", icon="📋")
        else:
            # 防御：上次弹窗若因异常未消费关闭标志，先清零，避免开窗即被关闭
            st.session_state["_history_dialog_close"] = False
            _show_history_dialog(controller)
    st.toggle(
        "📜 附带历史上下文",
        key=Keys.HISTORY_ENABLED,
        help="开启后，本次查询会携带之前问答的摘要信息作为上下文",
    )
    # 导出当前对话——数据源为内存中的 chat_messages，空对话时禁用
    messages = st.session_state[Keys.CHAT_MESSAGES]
    session_id = st.session_state[Keys.CURRENT_SESSION_ID]
    st.download_button(
        "📥 导出对话",
        data=build_conversation_markdown(messages, session_id),
        file_name=f"对话导出_{(session_id or 'new')[:8]}.md",
        mime="text/markdown",
        key="export_conv_btn",
        use_container_width=True,
        disabled=not messages,
    )


def _render_search_executor(
    controller: SearchController,
    status_widget: StatusContainer,
) -> None:
    """搜索执行器——执行检索，进度实时更新外部状态行（st.status）。

    状态行渲染在执行器之前，检索期间的进度必须直接 update 该控件
    才能即时可见；rerun 后的持久显示依赖 session_state 中的文本。
    """
    do_search = st.session_state.get(Keys.DO_SEARCH, False)
    do_continue = st.session_state.get(Keys.DO_CONTINUE, False)

    if not do_search and not do_continue:
        return

    st.session_state[Keys.DO_SEARCH] = False
    st.session_state[Keys.DO_CONTINUE] = False

    # 组装历史上下文
    history_context = None
    if st.session_state[Keys.HISTORY_ENABLED]:
        history_context = build_history_context(st.session_state[Keys.CHAT_MESSAGES])

    def on_progress(status_line: str | None, log_line: str | None):
        """管线回调：状态行同时写 session（rerun 后持久）和控件（本轮即时可见）。"""
        if status_line:
            st.session_state[Keys.SEARCH_STATUS_TEXT] = status_line
            status_widget.update(label=status_line, state="running")
        if log_line:
            add_progress_log(log_line, key=Keys.PROGRESS_LOG_SEARCH)

    session_id = st.session_state[Keys.CURRENT_SESSION_ID]

    try:
        if do_search:
            result, session_id = controller.execute_search(
                st.session_state[Keys.PENDING_QUERY],
                session_id,
                history_context=history_context,
                on_progress=on_progress,
            )
            _handle_search_result(result, session_id)
        else:
            choice = st.session_state.get(Keys.PENDING_CHOICE)
            if choice is None:
                st.session_state[Keys.SEARCH_STATE] = "idle"
                st.session_state[Keys.SEARCH_STATUS_TEXT] = "❌ 未找到用户选择"
            else:
                result, session_id = controller.execute_continue(
                    choice,
                    session_id,
                    history_context=history_context,
                    on_progress=on_progress,
                )
                _handle_search_result(result, session_id)
    except Exception as exc:
        st.session_state[Keys.SEARCH_STATE] = "idle"
        st.session_state[Keys.SEARCH_STATUS_TEXT] = f"❌ 检索失败: {exc}"
    # 状态行在执行器之前已渲染，最终结果（回答完成/需要决策/失败）
    # 必须回写控件才能在本轮立即可见，否则要等下次交互触发的 rerun
    final_text = st.session_state[Keys.SEARCH_STATUS_TEXT]
    status_widget.update(
        label=final_text,
        state="error" if final_text.startswith("❌") else "complete",
    )


def _render_choice_area() -> None:
    """选择区：needs_choice 状态下显示分支选项。"""
    if st.session_state[Keys.SEARCH_STATE] != "needs_choice":
        return
    result: SearchResult = st.session_state[Keys.SEARCH_RESULT]
    if result is None:
        return

    st.warning("🚨 检索需要你的决策")
    st.caption(
        "⚠️ 本次检索需要你做出选择才能完成。完成后的问答将被保存到历史。"
    )

    if result.pending_decision == "zero_results":
        st.write("知识库中未检索到与您问题相关的资料。请选择：")
        col_a, col_b = st.columns(2)
        with col_a:
            st.button("🤖 由 AI 直接回答（不使用知识库）", key="choice_direct_llm",
                      on_click=_on_continue_choice, args=(ContinueChoice.DIRECT_LLM,))
        with col_b:
            st.button("❌ 放弃本次查询", key="choice_abandon_zero",
                      on_click=_on_continue_choice, args=(ContinueChoice.ABANDON,))

    elif result.pending_decision == "low_confidence":
        top = result.top_score or 0
        st.write(
            f"知识库中未找到与您问题高度相关的资料（最高相关性评分 {top}/10）。请选择："
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.button("📄 基于低匹配度片段尝试回答（可能不够准确）", key="choice_rag",
                      on_click=_on_continue_choice, args=(ContinueChoice.RAG,))
        with col_b:
            st.button("🤖 由 AI 直接回答（不使用知识库）", key="choice_direct_llm_low",
                      on_click=_on_continue_choice, args=(ContinueChoice.DIRECT_LLM,))


def _render_conversation_messages() -> None:
    """对话消息——空时占位，有内容时用聊天气泡展示，每条附来源引用。"""
    messages = st.session_state[Keys.CHAT_MESSAGES]
    if not messages:
        st.caption("对话区 — 输入问题开始检索")
        return
    for msg in messages:
        with st.chat_message("user"):
            st.write(msg.get("query", ""))
        with st.chat_message("assistant"):
            st.write(msg.get("answer", ""))
            sources = msg.get("sources", [])
            if sources:
                render_sources(sources)
            # 单轮 token 消耗（输入/输出）；无数据的轮次不显示
            tokens = msg.get("token_usage") or {}
            t_in, t_out = tokens.get("input", 0), tokens.get("output", 0)
            if t_in or t_out:
                st.caption(f"🔢 tokens: {t_in:,} / {t_out:,}")


@st.dialog("📋 查询历史", width="medium")
def _show_history_dialog(controller: SearchController) -> None:
    """历史记录弹窗——会话列表，点击恢复加载对应对话到对话区。

    恢复按钮的 on_click 设置 _history_dialog_close 标志，dialog 检测到
    标志后调用 st.rerun() 触发完整页面重跑（on_click 未渲染元素，安全）。
    重跑后主面板按钮不为 True，dialog 函数不执行 → 弹窗关闭。
    """
    sessions = controller.list_sessions(limit=30)
    if not sessions:
        st.caption("暂无历史记录")
        return
    # 恢复回调设置了关闭标志 → st.rerun() 关闭弹窗
    # （回调中不渲染任何元素，安全；st.rerun 触发整页重跑，主面板按钮
    # 不为 True → dialog 函数不执行 → 弹窗关闭）
    if st.session_state.get("_history_dialog_close"):
        st.session_state["_history_dialog_close"] = False
        st.rerun()
    st.caption(f"共 {len(sessions)} 次对话")
    for sess in sessions:
        ts = sess.get("updated_at", "")[:16]
        title = sess.get("title", "无标题")[:40]
        # &nbsp; 实体撑开间距（st.caption 走 markdown，连续空格会被折叠）
        ts_gap = "&nbsp;" * 5
        line = (
            f"{ts}{ts_gap}—{ts_gap}{title}  "
            f"({sess['turn_count']}轮, "
            f"{sess['total_input_tokens']}/{sess['total_output_tokens']} tokens)"
        )
        # 左 caption + 右恢复按钮：caption 天然左对齐，根除居中按钮造成的参差不齐
        col_info, col_btn = st.columns([4, 0.7], vertical_alignment="center")
        with col_info:
            st.caption(line)
        with col_btn:
            st.button(
                "📜 恢复", key=f"hist_{sess['session_id']}", use_container_width=True,
                on_click=_on_restore_session, args=(controller, sess["session_id"]),
            )

    st.caption(
        "💡 当前对话历史暂存于 SQLite 的 JSON 字段中，"
        "建议每个 session 的问答不超过 30 轮。"
        "后续将迁移至 JSONL 存储以支持更大规模会话管理。"
    )


def _render_progress_log() -> None:
    """进度日志区——固定高度，内容滚动，仅存于前端内存。"""
    st.caption("📋 处理详情")
    with st.container(height=LOG_HEIGHT, border=True):
        log_entries = list(st.session_state[Keys.PROGRESS_LOG_SEARCH])
        if not log_entries:
            st.caption("暂无日志")
        else:
            for entry in log_entries:
                st.caption(entry)


# ── 结果处理 ──────────────────────────────────────────────

def _handle_search_result(
    result: SearchResult,
    session_id: Optional[str],
) -> None:
    """处理 Controller 返回的 SearchResult，更新 UI 状态。"""
    query = st.session_state[Keys.PENDING_QUERY]

    if result.status == "needs_user_choice":
        st.session_state[Keys.CURRENT_SESSION_ID] = session_id
        st.session_state[Keys.SEARCH_STATE] = "needs_choice"
        st.session_state[Keys.SEARCH_RESULT] = result
        st.session_state[Keys.SEARCH_STATUS_TEXT] = "🚨 需要用户决策"
        return

    if result.status == "error":
        st.session_state[Keys.CURRENT_SESSION_ID] = session_id
        st.session_state[Keys.SEARCH_STATE] = "idle"
        st.session_state[Keys.SEARCH_RESULT] = None
        st.session_state[Keys.SEARCH_STATUS_TEXT] = f"❌ 错误: {result.error_message}"
        st.error(result.error_message or "未知错误")
        return

    # status == "success"
    answer = result.answer or ""
    token_usage = result.token_usage or {}
    elapsed_ms = int(
        (time.time() - st.session_state.get(Keys.SEARCH_START_TIME, time.time())) * 1000
    )

    # 累加 token（供 HistoryStore 记录；前端状态行仅展示单次消耗）
    accumulate_session_tokens(token_usage, "search")

    st.session_state[Keys.CURRENT_SESSION_ID] = session_id
    st.session_state[Keys.SEARCH_STATE] = "done"
    st.session_state[Keys.SEARCH_RESULT] = result
    elapsed_s = elapsed_ms / 1000
    st.session_state[Keys.SEARCH_STATUS_TEXT] = (
        f"✅ 回答完成 — {token_usage.get('input', 0):,}/{token_usage.get('output', 0):,} tokens"
        f" (耗时 {elapsed_s:.1f}s)"
    )
    sources_data = [
        {
            "doc_name": s.doc_name or s.doc_id,
            "page_nums": s.page_nums,
            "chunk_index": s.chunk_index,
            "chapter_title": s.chapter_title,
            "chapter_index": s.chapter_index,
            "llm_score": s.llm_score,
        }
        for s in (result.sources or [])
    ]
    add_chat_message(query, answer, sources_data, token_usage)


def _restore_from_history(controller: SearchController, session_id: str) -> None:
    """从历史记录恢复整个对话到当前 UI 展示区。

    仅覆盖当前展示，不触发新的 pipeline 调用。
    不调用 st.toast ——此函数可能在 dialog 内的 on_click 回调中执行（fragment
    上下文），回调中渲染元素不被 Streamlit 官方支持，会导致空白 UI。
    反馈通过状态行（SEARCH_STATUS_TEXT）传达，持久可见。
    """
    session = controller.restore_session(session_id)
    if session is None:
        st.session_state[Keys.SEARCH_STATUS_TEXT] = "⚠️ 历史记录加载失败"
        return

    messages = session.get("messages", [])
    if not messages:
        st.session_state[Keys.SEARCH_STATUS_TEXT] = "⚠️ 该会话无对话记录"
        return

    # 恢复对话 ID，后续追问会追加到同一条记录
    st.session_state[Keys.CURRENT_SESSION_ID] = session_id
    st.session_state[Keys.CHAT_MESSAGES] = []

    # 将历史消息逐条恢复到 chat_messages
    last_answer = ""
    last_sources = []
    for msg in messages:
        sources = msg.get("sources", [])
        token_usage = {
            "input": msg.get("input_tokens", 0),
            "output": msg.get("output_tokens", 0),
        }
        add_chat_message(
            msg.get("query", ""),
            msg.get("answer", ""),
            sources,
            token_usage,
        )
        last_answer = msg.get("answer", "")
        last_sources = sources

    # 展示最后一轮的答案
    st.session_state[Keys.SEARCH_STATE] = "done"
    from _types.retrieval_types import SearchResult as SR
    st.session_state[Keys.SEARCH_RESULT] = SR(
        status="success",
        query=messages[-1].get("query", ""),
        answer=last_answer,
        sources=last_sources,
        source_count=len(last_sources),
        token_usage={
            "input": messages[-1].get("input_tokens", 0),
            "output": messages[-1].get("output_tokens", 0),
        },
    )
    st.session_state[Keys.PENDING_QUERY] = messages[-1].get("query", "")
    st.session_state[Keys.SEARCH_STATUS_TEXT] = (
        f"📜 已恢复历史对话 — {len(messages)} 轮 (id={session_id[:8]})"
    )
