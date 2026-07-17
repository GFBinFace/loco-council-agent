"""可复用 UI 组件。

提供来源引用展示、历史上下文组装、确认对话框等通用组件。
"""

from typing import Any, Callable, List, Tuple

import streamlit as st


# ── 分割线 ────────────────────────────────────────────────

def render_divider() -> None:
    """渲染区域间距（留白分离，替代传统分割线）。"""
    st.html("<div style='margin:8px 0;'></div>")


# ── 业务组件 ──────────────────────────────────────────────


def _source_field(s, name: str, default):
    """从 ChunkCandidate 对象或 dict 中取来源字段（两种形态并存于实时/恢复路径）。"""
    if isinstance(s, dict):
        return s.get(name, default)
    return getattr(s, name, default)


def format_source_locator(s) -> str:
    """
    组装单条来源的定位符文本。

    规则：有页码显示页码（PDF），有章节显示章节（TXT 章节分块），
    chunk # 恒显示——任何情况下不产出"页码 []"这类空定位。

    Args:
        s: ChunkCandidate 对象或 dict，含 page_nums/chapter_title/chunk_index

    Returns:
        形如 "页码 [3] — chunk #5" 或 "天界卷 第三百二十九章 — chunk #189"。
    """
    parts: list[str] = []
    pages = _source_field(s, "page_nums", [])
    if pages:
        parts.append(f"页码 {pages}")
    chapter = _source_field(s, "chapter_title", "")
    if chapter:
        parts.append(str(chapter))
    parts.append(f"chunk #{_source_field(s, 'chunk_index', '?')}")
    return " — ".join(parts)


def render_sources(sources: list) -> None:
    """
    展示来源引用列表。

    Args:
        sources: ChunkCandidate 列表或 dict 列表，
                 每项至少含 doc_name, page_nums, chunk_index。
    """
    if not sources:
        return
    with st.expander(f"📚 来源引用 ({len(sources)} 个 chunk)", expanded=False):
        for i, s in enumerate(sources):
            doc = _source_field(s, "doc_name", "?")
            score = _source_field(s, "llm_score", None)
            st.caption(
                f"{i + 1}. {doc} — {format_source_locator(s)}"
                + (f" — LLM得分={score}" if score is not None else "")
            )


def build_conversation_markdown(
    chat_messages: list, session_id: str | None = None,
) -> str:
    """
    将当前对话的 chat_messages 渲染为 Markdown 导出文本。

    Args:
        chat_messages: 当前会话的 Q&A 列表（含 query/answer/sources/token_usage）
        session_id: 会话 ID，None 表示尚未保存的新对话

    Returns:
        Markdown 格式的完整对话文本。
    """
    lines: list[str] = ["# 对话导出", ""]
    lines.append(f"- 会话 ID: {session_id or '（未保存）'}")
    lines.append(f"- 轮数: {len(chat_messages)}")
    lines.append("")
    for i, msg in enumerate(chat_messages, start=1):
        tokens = msg.get("token_usage") or {}
        lines.append(f"## 第 {i} 轮")
        lines.append("")
        lines.append(f"**问：** {msg.get('query', '')}")
        lines.append("")
        lines.append(f"**答：** {msg.get('answer', '')}")
        lines.append("")
        sources = msg.get("sources", [])
        if sources:
            lines.append("**来源引用：**")
            for s in sources:
                doc = _source_field(s, "doc_name", "?")
                score = _source_field(s, "llm_score", None)
                lines.append(
                    f"- {doc} — {format_source_locator(s)}"
                    + (f" — LLM得分={score}" if score is not None else "")
                )
            lines.append("")
        lines.append(
            f"**Tokens：** {tokens.get('input', 0)}/{tokens.get('output', 0)}"
        )
        lines.append("")
    return "\n".join(lines)


def build_history_context(chat_messages: list) -> list[dict]:
    """
    从当前会话的 chat_messages 组装 history_context。

    仅提取索引信息（doc_name, page_nums, chunk_index），不携带 chunk 全文。

    Args:
        chat_messages: 当前会话的 Q&A 列表

    Returns:
        history_context 列表，可直接传给 pipeline.search()。
    """
    context: list[dict] = []
    for msg in chat_messages:
        sources = msg.get("sources", [])
        source_refs: list[dict] = []
        for s in sources:
            source_refs.append({
                "doc_name": _source_field(s, "doc_name", ""),
                "page_nums": _source_field(s, "page_nums", []),
                "chunk_index": _source_field(s, "chunk_index", 0),
                "chapter_title": _source_field(s, "chapter_title", ""),
            })
        context.append({
            "query": msg.get("query", ""),
            "answer": msg.get("answer", ""),
            "sources": source_refs,
        })
    return context


def confirm_delete_button(
    label: str = "删除",
    key: str = "",
    on_confirm: Callable[..., None] | None = None,
    args: Tuple[Any, ...] = (),
) -> None:
    """
    删除确认按钮——点击弹出 popover 二次确认。

    点击 popover 内容外的任意位置即为取消，无需单独取消按钮。
    确认按钮走 on_click 回调（先于脚本重跑执行），调用方无需 st.rerun()。

    Args:
        label: 按钮文字
        key: Streamlit 组件唯一键
        on_confirm: 用户确认删除时执行的回调
        args: 传给 on_confirm 的位置参数

    Returns:
        None
    """
    with st.popover(label, use_container_width=True):
        st.caption("确认删除此文档？")
        st.button(
            "✅ 确认", key=f"yes_{key}", use_container_width=True,
            on_click=on_confirm, args=args,
        )
