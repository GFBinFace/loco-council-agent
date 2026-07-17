"""文档管理面板 — 左侧区域。

提供文档列表、属性编辑、上传、分组开关、占位按钮。
"""

from collections import deque
import time
from typing import Any, Dict, List

import streamlit as st

from config import StreamlitConfig
from controllers.doc_controller import DocController
from ui.components import confirm_delete_button, render_divider
from ui.layout import DOC_LIST_HEIGHT, LOG_HEIGHT
from ui.session_state import Keys, accumulate_session_tokens, add_progress_log


# ── 主渲染函数 ────────────────────────────────────────────

def render_doc_mgmt_panel(index_ctrl: DocController) -> None:
    """渲染左侧文档管理面板。"""
    # 确保进度 log 容器存在于任何路径（包括 rerun 后）——
    # st.rerun() 打断按钮分支的流执行后，分支内创建的新 session_state
    # key 可能被重置，此处提前创建以保证 defer 安全。
    st.session_state.setdefault(Keys.PROGRESS_LOG_DOC, deque(maxlen=1000))
    try:
        grouped = index_ctrl.list_documents()
    except Exception:
        grouped = None

    doc_status_widget = _render_status_line(grouped)
    render_divider()
    with st.container(height=DOC_LIST_HEIGHT):
        _render_document_list(grouped, index_ctrl)
    _render_placeholder_buttons(index_ctrl)
    render_divider()
    _render_upload_section(index_ctrl, grouped, doc_status_widget)
    _render_progress_log()


# ── 子区域 ────────────────────────────────────────────────

def _render_status_line(grouped: Any):
    """状态行（st.status）——显示文档统计。

    返回 st.status 控件供索引按钮分支的 on_progress 实时更新。
    与搜索区 _render_status_line() 同款模式：控件在本函数渲染，后续
    执行器通过返回对象 update(label, state) 改写。
    """
    # 非索引路径：按 grouped 组装统计文本
    if grouped is None:
        text = "📄 无法加载文档列表"
        state = "error"
    elif not grouped:
        text = "📄 知识库为空 — 请上传文档"
        state = "complete"
    else:
        total_docs = sum(len(files) for files in grouped.values())
        total_chunks = sum(
            f.get("total_chunks", 0) for files in grouped.values() for f in files
        )
        text = f"📄 知识库就绪 — {total_docs} 个文档, {total_chunks} chunks"
        state = "complete"
    # 带 key 的容器为 CSS 提供 st-key-doc_status_line 类锚点（浅蓝底色见 app.py）
    with st.container(key="doc_status_line"):
        return st.status(text, expanded=False, state=state)


def _render_document_list(grouped, index_ctrl: DocController) -> None:
    """文档列表：按分组折叠，每组内表格行展示文档属性。"""
    if grouped is None:
        st.warning("无法加载文档列表")
        return

    if not grouped:
        st.caption("暂无文档，请上传 PDF 或 TXT")
        return

    for group_name, files in grouped.items():
        _render_group(index_ctrl, group_name, files)


# ── 启用状态回调 ──────────────────────────────────────────
#
# 为什么用 on_click/on_change 回调而不是 `if st.button(): ... st.rerun()`：
# 回调先于脚本重跑执行，重跑时读到的已是新数据，无需 st.rerun()（也避免了
# 中途打断脚本导致未渲染控件状态被清理的副作用）。
# 为什么要显式改写 session_state：带 key 的 toggle 身份只由 key 决定
# （key_as_main_identity=True），改 value= 参数不会重置前端已存在的控件，
# 只有写 session_state 才会带强制更新标记下发到前端。

def _on_group_enabled_change(
    index_ctrl: DocController, group_name: str, files: List[Dict], enabled: bool,
) -> None:
    """组级启用/禁用回调：先写库，再同步组内各行 toggle 的 widget 状态。"""
    index_ctrl.set_group_enabled(group_name, enabled)
    for f in files:
        st.session_state[f"enabled_{f['file_md5']}"] = enabled


def _on_doc_enabled_change(index_ctrl: DocController, md5: str) -> None:
    """单文档启用 toggle 回调：将用户拨动后的 widget 状态写入 DB。"""
    index_ctrl.set_document_enabled(md5, st.session_state[f"enabled_{md5}"])


def _on_edit_mode(edit_key: str, editing: bool) -> None:
    """进入/退出文档行编辑模式回调。"""
    st.session_state[edit_key] = editing


def _on_edit_save(index_ctrl: DocController, doc: Dict, md5: str, edit_key: str) -> None:
    """保存按钮回调：仅当用户填写了新值时写库（空字符串视为不修改）。"""
    new_type = st.session_state.get(f"edit_type_{md5}", "")
    new_groups = st.session_state.get(f"edit_groups_{md5}", "")
    new_tags = st.session_state.get(f"edit_tags_{md5}", "")
    if new_type and new_type != doc.get("file_type", ""):
        index_ctrl.set_document_file_type(md5, new_type)
    if new_groups and new_groups != doc.get("groups", ""):
        index_ctrl.set_document_groups(md5, new_groups)
    if new_tags and new_tags != doc.get("tags", ""):
        index_ctrl.set_document_tags(md5, new_tags)
    st.session_state[edit_key] = False


def _on_delete_doc(index_ctrl: DocController, md5: str, file_name: str) -> None:
    """删除确认回调：删除文档元数据与向量数据，toast 反馈结果。"""
    index_ctrl.delete_document(md5)
    st.toast(f"已删除: {file_name}", icon="🗑️")


def _on_op_range_query() -> None:
    """操作历史范围查询回调：校验通过则锁定查询范围，否则记录错误信息。"""
    page_size = StreamlitConfig.operation_history_page_size
    from_id = st.session_state.get("op_from_input", 1)
    to_id = st.session_state.get("op_to_input", 1)
    if from_id > to_id:
        st.session_state[Keys.OP_RANGE_ERROR] = "起始 ID 不能大于结束 ID"
        return
    if to_id - from_id + 1 > page_size:
        st.session_state[Keys.OP_RANGE_ERROR] = f"一次最多显示 {page_size} 条记录"
        return
    st.session_state[Keys.OP_RANGE_ERROR] = ""
    st.session_state[Keys.OP_FROM_ID] = from_id
    st.session_state[Keys.OP_TO_ID] = to_id


def _render_group(index_ctrl: DocController, group_name: str, files: List[Dict]) -> None:
    """
    渲染单个文档分组。

    Args:
        index_ctrl: DocController 实例
        group_name: 分组名称
        files: 该分组下的文档列表
    """
    # 组头：名称 + 全激活/全禁用按钮
    col_title, col_btn1, col_btn2 = st.columns([3, 1, 1])
    with col_title:
        st.caption(f"📁 {group_name} ({len(files)} 个文档)")
    with col_btn1:
        st.button(
            "✅ 全部激活", key=f"group_enable_{group_name}", use_container_width=True,
            on_click=_on_group_enabled_change, args=(index_ctrl, group_name, files, True),
        )
    with col_btn2:
        st.button(
            "🚫 全部禁用", key=f"group_disable_{group_name}", use_container_width=True,
            on_click=_on_group_enabled_change, args=(index_ctrl, group_name, files, False),
        )

    with st.expander(f"📁 {group_name}", expanded=True):
        _render_document_header()
        for f in files:
            _render_document_row(index_ctrl, f)


def _render_document_header() -> None:
    """渲染文档列表表头。"""
    cols = st.columns([2.0, 1, 1, 1, 0.7, 0.7, 0.9, 0.5, 1.9])
    headers = ["文件名", "类型", "分组", "标签", "大小", "分块数", "索引时间", "启用", "数据操作"]
    aligns = ["L", "L", "L", "L", "R", "R", "L", "L", "C"]
    for col, h, a in zip(cols, headers, aligns):
        a_style = {"L": "left", "R": "right", "C": "center"}[a]
        col.html(
            f"<small style='display:block;text-align:{a_style};"
            f"line-height:1.8;padding-top:0.35rem;'>{h}</small>"
        )


def _render_document_row(index_ctrl: DocController, doc: Dict) -> None:
    """
    渲染单个文档的属性行。

    默认为只读视图，"✏️" 按钮进入编辑模式。
    编辑模式下显示 text_input + "💾 保存" 按钮。
    """
    md5 = doc["file_md5"]
    edit_key = f"edit_mode_{md5}"
    is_editing = st.session_state.get(edit_key, False)

    # ── 只读行 ──
    if not is_editing:
        cols = st.columns([2.0, 1, 1, 1, 0.7, 0.7, 0.9, 0.5, 1.9])
        _render_readonly_row(cols, doc, md5, edit_key, index_ctrl)
        return

    # ── 编辑行 ──
    _render_edit_row(index_ctrl, doc, md5, edit_key)


def _render_readonly_row(
    cols, doc: Dict, md5: str, edit_key: str, index_ctrl: DocController,
) -> None:
    """渲染只读状态下的文档行。"""
    _cell = lambda text, align="left": st.html(
        f"<small style='display:block;text-align:{align};"
        f"line-height:1.8;padding-top:0.35rem;'>{text}</small>"
    )
    with cols[0]:
        name = doc.get("file_name", "?")
        _cell(name[:30] + ("…" if len(name) > 30 else ""))
    with cols[1]:
        _cell(doc.get("file_type", "") or "—")
    with cols[2]:
        _cell(doc.get("groups", "") or "—")
    with cols[3]:
        _cell(doc.get("tags", "") or "—")
    with cols[4]:
        size_kb = doc.get("file_size", 0) / 1024
        size_text = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
        _cell(size_text, align="right")
    with cols[5]:
        _cell(str(doc.get("total_chunks", 0)), align="right")
    with cols[6]:
        _cell((doc.get("indexed_at", "") or "")[:10])
    with cols[7]:
        # 启用开关（受控模式）：显示值以 session_state 为唯一事实源，首次渲染以 DB 值播种，
        # 组按钮通过改写 session_state 同步显示；写库只发生在用户真实拨动时（on_change），
        # 避免"widget 状态 ≠ DB 状态"被误判为用户操作而静默回写。
        # 注意不传 value= ——播种后再传默认值会触发 Streamlit 的重复赋值告警。
        toggle_key = f"enabled_{md5}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = doc.get("is_enabled", 1) == 1
        st.toggle(
            "启用",
            key=toggle_key,
            label_visibility="collapsed",
            on_change=_on_doc_enabled_change,
            args=(index_ctrl, md5),
        )
    with cols[8]:
        col_edit, col_del = st.columns([0.9, 1.1])
        with col_edit:
            st.button(
                "编辑", key=f"edit_btn_{md5}", use_container_width=True,
                on_click=_on_edit_mode, args=(edit_key, True),
            )
        with col_del:
            confirm_delete_button(
                "删除", key=f"del_{md5}",
                on_confirm=_on_delete_doc,
                args=(index_ctrl, md5, doc.get("file_name", md5[:8])),
            )


def _render_edit_row(
    index_ctrl: DocController, doc: Dict, md5: str, edit_key: str,
) -> None:
    """渲染编辑状态下的文档行。"""
    st.caption(f"✏️ 编辑: {doc.get('file_name', '?')}")

    col_type, col_group, col_tag = st.columns([1, 1.2, 1.2])

    # 输入值不取返回值——保存回调（_on_edit_save）直接从 session_state 按 key 读取
    with col_type:
        st.text_input(
            "文件类型",
            key=f"edit_type_{md5}",
            placeholder=doc.get("file_type", "") or "类型",
        )
    with col_group:
        st.text_input(
            "分组",
            key=f"edit_groups_{md5}",
            placeholder=doc.get("groups", "") or "分组（| 分隔）",
        )
    with col_tag:
        st.text_input(
            "标签",
            key=f"edit_tags_{md5}",
            placeholder=doc.get("tags", "") or "标签（| 分隔）",
        )

    col_save, col_cancel = st.columns([1, 1])
    with col_save:
        st.button(
            "💾 保存", key=f"save_{md5}", use_container_width=True,
            on_click=_on_edit_save, args=(index_ctrl, doc, md5, edit_key),
        )
    with col_cancel:
        st.button(
            "❌ 取消", key=f"cancel_{md5}", use_container_width=True,
            on_click=_on_edit_mode, args=(edit_key, False),
        )


def _on_file_picked() -> None:
    """file_uploader 回调：将选中文件存为临时文件，路径填入输入框。"""
    import os
    import tempfile
    uploaded = st.session_state.get(Keys.PDF_PICKER)
    if uploaded is None:
        return
    # 根据原始文件名保留扩展名
    suffix = os.path.splitext(uploaded.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        st.session_state[Keys.PDF_PATH_INPUT] = tmp.name


def _render_upload_section(
    index_ctrl: DocController, grouped, doc_status_widget,
) -> None:
    """上传区：输入本地文档路径 → 索引。

    doc_status_widget 是顶部状态行控件——on_progress 实时 update 实现
    进度内联到状态栏。
    """
    st.markdown("#### 📤 索引文档")
    col_path, col_pick, col_btn = st.columns([3.5, 0.5, 1])
    with col_path:
        file_path = st.text_input(
            "文档路径",
            key=Keys.PDF_PATH_INPUT,
            label_visibility="collapsed",
            placeholder="输入路径，或点击 📂 选择文件…",
        )
    with col_pick:
        with st.popover("📂", use_container_width=True):
            st.file_uploader(
                "选择文件",
                type=["pdf", "txt"],
                key=Keys.PDF_PICKER,
                label_visibility="collapsed",
                on_change=_on_file_picked,
            )
    with col_btn:
        # ⚠️ 此按钮保留 `if st.button` 分支写法，不改造为 on_click 回调：
        # 索引是长耗时操作，必须在渲染流中持有 st.status 容器实时更新进度，
        # 而回调先于脚本执行、无法渲染 UI。结尾的 st.rerun() 属于任务收尾
        # 刷新文档列表，此时页面已展示完整结果，不属于被替换的问题模式。
        if st.button("🚀 开始索引", key="index_btn", use_container_width=True):
            doc_name = file_path.strip()
            add_progress_log(f"开始索引: {doc_name}", key=Keys.PROGRESS_LOG_DOC)

            # 组建文档统计文案供索引进度附加
            def _doc_stats_text():
                if grouped is None:
                    return "文档库无法加载"
                if not grouped:
                    return "文档库为空"
                total_docs = sum(len(files) for files in grouped.values())
                total_chunks = sum(
                    f.get("total_chunks", 0)
                    for files in grouped.values() for f in files
                )
                return f"文档库已有：{total_docs} 个文档, {total_chunks} chunks"

            def on_progress(status_line: str | None, log_line: str | None):
                if status_line:
                    doc_status_widget.update(label=(
                        f"索引进度：{status_line}。{_doc_stats_text()}"
                    ), state="running")
                if log_line:
                    add_progress_log(log_line, key=Keys.PROGRESS_LOG_DOC)

            result = index_ctrl.execute_index(doc_name, on_progress=on_progress)
            # 如果是从 📂 选择的临时文件，索引后清理
            _cleanup_picked_tmp(doc_name)

            token_usage = result.get("token_usage")
            doc_name_short = result.get("doc_name", doc_name)
            if result.get("success"):
                if token_usage:
                    accumulate_session_tokens(token_usage, "index")
                add_progress_log(
                    f"索引完成: {doc_name_short} — {result.get('num_chunks', 0)} chunks",
                    key=Keys.PROGRESS_LOG_DOC,
                )
                st.success(
                    f"✅ {doc_name_short} — {result.get('num_chunks', 0)} chunks 已索引"
                    + (f" (tokens: {token_usage['input']:,}/{token_usage['output']:,})"
                       if token_usage else "")
                )
            elif result.get("skipped"):
                add_progress_log(f"索引跳过: {doc_name_short} — 已存在", key=Keys.PROGRESS_LOG_DOC)
                st.info(f"⏭️ {doc_name_short} — 已存在，跳过索引")
            else:
                add_progress_log(
                    f"索引失败: {doc_name_short} — {result.get('error', '未知错误')}",
                    key=Keys.PROGRESS_LOG_DOC,
                )
                st.error(f"❌ 索引失败: {result.get('error', '未知错误')}")

            time.sleep(1)
            st.rerun()


def _render_placeholder_buttons(index_ctrl) -> None:
    """知识库概览 + 操作历史。"""
    col1, col2 = st.columns(2)
    with col1:
        with st.popover("📊 知识库概览", use_container_width=True):
            _render_overview_popover(index_ctrl)
    with col2:
        # dialog 惯用法：st.dialog 函数必须在渲染流中调用（回调无法渲染 UI），
        # 属于回调模式的合法例外。打开必经此按钮 → 此处即"打开事件"，
        # 重置查询范围为最新一页，实现"重开即见最新"。
        if st.button("📋 操作历史", key="op_history_btn", use_container_width=True):
            _reset_op_range_to_latest(index_ctrl)
            _show_operation_history_dialog(index_ctrl)


def _render_overview_popover(index_ctrl) -> None:
    """渲染知识库概览弹窗内容。"""
    try:
        ov = index_ctrl.get_overview()
    except Exception:
        st.caption("⚠️ 无法加载概览数据")
        return

    total_docs = ov["total_docs"]
    enabled_docs = ov["enabled_docs"]
    total_chars = ov["total_chars"]
    latest = ov["latest_index"]
    storage = ov["storage_bytes"]

    # ── 格式化各指标 ──
    # 已索引文档
    st.caption(f"📄 已索引文档：{total_docs}")

    # 激活文档
    if total_docs > 0:
        pct = round(enabled_docs / total_docs * 100)
        st.caption(f"✅ 激活文档：{enabled_docs}/{total_docs} ({pct}%)")
    else:
        st.caption("✅ 激活文档：—")

    # 知识块总数
    st.caption(f"🧩 知识块总数：{ov['total_chunks']:,}")

    # 文本总量
    if total_chars is not None and total_chars >= 0:
        if total_chars >= 10000:
            chars_display = f"{total_chars / 10000:.1f} 万字符"
        else:
            chars_display = f"{total_chars:,} 字符"
        st.caption(f"📝 文本总量：{chars_display}")
    else:
        st.caption("📝 文本总量：—")

    # 最近索引
    if latest:
        st.caption(f"🕐 最近索引：{latest[:16].replace('T', ' ')}")
    else:
        st.caption("🕐 最近索引：无")

    # 向量维度
    st.caption(f"🧬 向量维度：{ov['embedding_dim']}")

    # 存储占用
    if storage >= 0:
        if storage >= 1024 * 1024 * 1024:
            size_display = f"{storage / (1024**3):.2f} GB"
        elif storage >= 1024 * 1024:
            size_display = f"{storage / (1024**2):.1f} MB"
        elif storage >= 1024:
            size_display = f"{storage / 1024:.1f} KB"
        else:
            size_display = f"{storage} B"
        st.caption(f"💾 存储占用：{size_display}")
    else:
        st.caption("💾 存储占用：—")


def _reset_op_range_to_latest(index_ctrl: DocController) -> None:
    """将操作历史查询范围重置为最新一页——弹窗打开事件时调用。"""
    stats = index_ctrl.get_operation_stats()
    page_size = StreamlitConfig.operation_history_page_size
    st.session_state[Keys.OP_FROM_ID] = max(
        stats["min_id"], stats["max_id"] - page_size + 1
    )
    st.session_state[Keys.OP_TO_ID] = stats["max_id"]
    st.session_state[Keys.OP_RANGE_ERROR] = ""


@st.dialog("📋 操作历史", width="medium")
def _show_operation_history_dialog(index_ctrl: DocController) -> None:
    """操作历史弹窗——默认显示最新一页，支持按 ID 范围查询翻旧账。

    打开时由按钮分支重置范围；弹窗内查询走 fragment 局部重跑，
    锁定的自定义范围仅在弹窗存续期内有效。
    """
    # 加载统计信息，空库提前返回
    try:
        stats = index_ctrl.get_operation_stats()
    except Exception:
        st.caption("⚠️ 无法加载操作历史")
        return
    if stats["total"] == 0:
        st.caption("暂无操作记录")
        return

    page_size = StreamlitConfig.operation_history_page_size
    min_id, max_id = stats["min_id"], stats["max_id"]

    # 范围查询表单——输入值不取返回值，查询回调（_on_op_range_query）从 session_state 读取；
    # value 做 clamp 防御：滚动清理后 min_id 上移，旧范围值可能越界；
    # vertical_alignment="bottom" 让无 label 的按钮与带 label 的输入框底边对齐
    col_from, col_to, col_btn = st.columns([1, 1, 0.6], vertical_alignment="bottom")
    with col_from:
        st.number_input(
            "从 ID", min_value=min_id, max_value=max_id,
            value=min(max(st.session_state[Keys.OP_FROM_ID], min_id), max_id),
            key="op_from_input",
        )
    with col_to:
        st.number_input(
            "到 ID", min_value=min_id, max_value=max_id,
            value=min(max(st.session_state[Keys.OP_TO_ID], min_id), max_id),
            key="op_to_input",
        )
    with col_btn:
        st.button(
            "🔍 查询", key="op_range_btn", use_container_width=True,
            on_click=_on_op_range_query,
        )
    # 范围校验错误由回调写入 session_state，此处渲染
    if st.session_state.get(Keys.OP_RANGE_ERROR):
        st.error(st.session_state[Keys.OP_RANGE_ERROR])

    # 按当前范围查询（SQL 侧利用 PK 索引过滤）
    from_val = st.session_state[Keys.OP_FROM_ID]
    to_val = st.session_state[Keys.OP_TO_ID]
    try:
        ops_in_range = index_ctrl.list_operations(
            limit=page_size, from_id=from_val, to_id=to_val,
        )
    except Exception:
        st.caption("⚠️ 加载失败")
        return

    # op_type → 图标映射
    type_icons = {"index": "📄", "update": "✏️", "delete": "🗑️"}
    result_labels = {
        "success": "成功", "skipped": "跳过", "failed": "失败",
    }

    # 统计行：当前查询范围 + 库存全貌（给用户下次查询的基准）
    st.caption(
        f"ID {from_val}–{to_val}，共 {len(ops_in_range)} 条 ｜ "
        f"库中现存 {stats['total']:,} 条（ID {min_id}–{max_id}）"
    )
    with st.container(height=420, border=False):
        if not ops_in_range:
            st.caption("该范围内无记录")
        # caption 走 markdown 渲染，连续空格会被折叠，用 &nbsp; 实体撑开项间距
        gap = "&nbsp;" * 5
        for op in ops_in_range:
            icon = type_icons.get(op["op_type"], "❓")
            result = result_labels.get(op["op_result"], op["op_result"])
            # ISO 时间取到分钟："2026-07-15T12:19:56…" → "2026-07-15 12:19"
            time_str = op["op_time"][:16].replace("T", " ")
            st.caption(
                f"#{op['id']}{gap}{time_str}{gap}{icon}{gap}{op['file_name']} — "
                f"{op['op_detail']}  ({result}"
                + (f", {op['chunk_count']} chunks" if op["chunk_count"] else "")
                + ")"
            )


def _cleanup_picked_tmp(path: str) -> None:
    """清理通过 📂 选择的临时文件（不删除用户手动输入的路径）。"""
    import os
    import tempfile
    if path.startswith(tempfile.gettempdir()):
        try:
            os.unlink(path)
        except OSError:
            pass


def _render_progress_log() -> None:
    """进度日志区——固定高度，内容滚动，仅存于前端内存。"""
    st.caption("📋 处理详情")
    with st.container(height=LOG_HEIGHT, border=True):
        log_entries = list(st.session_state[Keys.PROGRESS_LOG_DOC])
        if not log_entries:
            st.caption("暂无日志")
        else:
            for entry in log_entries:
                st.caption(entry)
