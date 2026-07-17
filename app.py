"""Loco Council RAG — Streamlit 前端入口。

启动:
    streamlit run app.py

    或:
    python -m streamlit run app.py
"""

import os

# HF_HOME 必须在导入 pipeline 模块之前设置——
# pipeline 的 import 链会触发 sentence_transformers 导入，该库在导入时读取 HF_HOME。
from config import Config
if Config.huggingface_cache_dir:
    os.environ["HF_HOME"] = Config.huggingface_cache_dir

import streamlit as st

from controllers.search_controller import SearchController
from controllers.doc_controller import DocController
from services.pipeline import get_pipeline
from storage.history_store import HistoryStore
from ui.search_panel import render_search_panel
from ui.doc_mgmt_panel import render_doc_mgmt_panel
from ui.session_state import init_session_state

# ── 页面配置 ──────────────────────────────────────────────

st.set_page_config(
    page_title="Loco Council RAG",
    page_icon="🔎",
    layout="wide",
)

# 隐藏 st.text_input 右侧的 "Press Enter to apply" 提示——
# Streamlit 在输入值与 session_state 不一致时自动显示此文本，
# 代码层面无开关可关闭，需 CSS 全局屏蔽。
# 另：两侧状态栏（st.status）着浅蓝底色，通过带 key 容器的
# st-key-* 类精准命中，不影响其他 expander（如来源引用）。
st.html(
    "<style>"
    "div[data-testid='InputInstructions'] {display: none;}"
    ".st-key-search_status_line details,"
    ".st-key-doc_status_line details {"
    "  background: #e8f0fe; border: none; border-radius: 0.25rem;"
    "}"
    # 状态栏不承载可展开内容：禁点击 + 隐藏误导性的展开箭头
    ".st-key-search_status_line summary,"
    ".st-key-doc_status_line summary {pointer-events: none;}"
    ".st-key-search_status_line summary [data-testid='stExpanderToggleIcon'],"
    ".st-key-doc_status_line summary [data-testid='stExpanderToggleIcon'] {display: none;}"
    "</style>"
)

# ── Pipeline 单例 + Controller 注入 ─────────────────────

@st.cache_resource
def get_pipeline_singleton():
    """Pipeline 全局唯一实例。

    get_pipeline() 在进程内保证单例（Python 模块级全局变量），
    @st.cache_resource 在 Streamlit 层面兜底（避免 rerun 重复创建）。
    """
    return get_pipeline()

@st.cache_resource
def get_controllers():
    """创建 Controller 并注入 Pipeline 单例 + HistoryStore。"""
    pipeline = get_pipeline_singleton()
    doc_ctrl = DocController(pipeline)
    search_ctrl = SearchController(pipeline, HistoryStore())
    return doc_ctrl, search_ctrl


# ── 页面渲染 ──────────────────────────────────────────────

def main() -> None:
    init_session_state()

    # 标题行。用 st.html 避免 st.markdown 自动注入锚点链接图标
    st.html(
        "<h2 style='text-align:center; margin-bottom:48px;'>"
        "🔎 Loco Council — RAG 智能检索工作台</h2>"
    )

    doc_ctrl, search_ctrl = get_controllers()

    col_left, col_gap, col_right = st.columns([1, 0.066, 1.5])
    with col_left:
        render_doc_mgmt_panel(doc_ctrl)
    with col_gap:
        st.caption("")
    with col_right:
        render_search_panel(search_ctrl)


if __name__ == "__main__":
    main()
