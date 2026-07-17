"""搜索管线服务调度层。

SearchController 是前端与下属服务（Pipeline、HistoryStore）之间的编排器。
前端仅负责渲染，业务编排集中在此。
"""

import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from _types.retrieval_types import ContinueChoice, SearchResult


# continue_search 选择 → 历史记录 status 映射
CHOICE_STATUS_MAP = {
    ContinueChoice.ABANDON: "abandoned",
    ContinueChoice.DIRECT_LLM: "direct_llm",
    ContinueChoice.RAG: "low_confidence_rag",
}


class SearchController:
    """搜索管线编排器——协调 Pipeline 和 HistoryStore。

    Pipeline 和 HistoryStore 由启动层注入（依赖注入），
    Controller 不自行获取全局实例。
    """

    def __init__(self, pipeline, history_store):
        self._pipeline = pipeline
        self._history_store = history_store

    # ── 公开接口 ──────────────────────────────────────────

    def execute_search(
        self,
        query: str,
        session_id: Optional[str],
        history_context: Optional[List[dict]] = None,
        on_progress: Optional[Callable] = None,
    ) -> Tuple[SearchResult, Optional[str]]:
        """执行检索管线。

        needs_user_choice 时不保存 history；success/error 时自动保存。

        Returns:
            (SearchResult, session_id)
            session_id 在新建对话时由 HistoryStore 生成，其余情况回传原值。
        """
        # 执行检索
        t0 = time.time()
        try:
            result = self._pipeline.search(
                query,
                history_context=history_context,
                on_progress=on_progress,
            )
        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            # 致命错误：Controller 统一汇报
            if on_progress:
                on_progress(f"❌ 检索失败", f"管线异常: {exc}")
            result = SearchResult(
                status="error",
                query=query,
                error_message=str(exc),
            )
            session_id = self._save_turn(session_id, result, "error", elapsed_ms)
            return result, session_id

        elapsed_ms = int((time.time() - t0) * 1000)

        # 保存历史
        if result.status in ("success", "error"):
            status = "error" if result.status == "error" else "success"
            session_id = self._save_turn(session_id, result, status, elapsed_ms)

        return result, session_id

    def execute_continue(
        self,
        choice: ContinueChoice,
        session_id: Optional[str],
        history_context: Optional[List[dict]] = None,
        on_progress: Optional[Callable] = None,
    ) -> Tuple[SearchResult, Optional[str]]:
        """继续执行检索管线（用户做出选择后）。

        Returns:
            (SearchResult, session_id)
        """
        # 执行检索
        status_override = CHOICE_STATUS_MAP.get(choice)
        t0 = time.time()
        try:
            result = self._pipeline.continue_search(
                choice,
                history_context=history_context,
                on_progress=on_progress,
            )
        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            # 致命错误：Controller 统一汇报
            if on_progress:
                on_progress(f"❌ 检索失败", f"管线异常: {exc}")
            result = SearchResult(
                status="error",
                query="",
                error_message=str(exc),
            )
            status = status_override or "error"
            session_id = self._save_turn(session_id, result, status, elapsed_ms)
            return result, session_id

        elapsed_ms = int((time.time() - t0) * 1000)

        # 保存历史
        if result.status in ("success", "error"):
            status = status_override or (
                "error" if result.status == "error" else "success"
            )
            session_id = self._save_turn(session_id, result, status, elapsed_ms)

        return result, session_id

    def restore_session(self, session_id: str) -> Optional[dict]:
        """从 HistoryStore 加载完整对话数据供 UI 恢复。

        Returns:
            含 messages 列表的 session dict，不存在时返回 None。
        """
        return self._history_store.get_session(session_id)

    def list_sessions(self, limit: int = 30) -> list[dict]:
        """获取最近 N 次对话摘要列表（供历史面板渲染）。"""
        return self._history_store.list_sessions(limit)

    def delete_session(self, session_id: str) -> bool:
        """删除一次完整对话记录。"""
        return self._history_store.delete_session(session_id)

    # ── 内部 ──────────────────────────────────────────────

    def _save_turn(
        self,
        session_id: Optional[str],
        result: SearchResult,
        status: str,
        elapsed_ms: int,
    ) -> str:
        """组装 turn 数据并保存到 HistoryStore。返回 session_id。"""
        answer = result.answer or ""
        token_usage = result.token_usage or {}

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

        turn = {
            "query": result.query,
            "answer": answer,
            "status": status,
            "sources": sources_data,
            "input_tokens": token_usage.get("input", 0),
            "output_tokens": token_usage.get("output", 0),
            "elapsed_ms": elapsed_ms,
            "created_at": datetime.now().isoformat(),
        }

        return self._history_store.save_turn(session_id, turn)
