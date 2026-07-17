import ast
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import Config

# ── HuggingFace 离线配置（必须在首次 import transformers / sentence_transformers 之前设置）──
# huggingface_hub / transformers 在首次 import 时缓存配置，env var 必须在此时就生效。
# 后续所有模型加载（Embedder、Reranker）都走本地缓存，不联网。
if Config.huggingface_cache_dir:
    os.environ["HF_HOME"] = Config.huggingface_cache_dir
os.environ["HF_HUB_OFFLINE"] = "1"      # huggingface_hub
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # transformers

from storage.doc_manager import DocManager
from services.retrieval.embedder import Embedder
from services.indexing.chunkers.financial_table_chunker import FinancialTableChunker
from services.indexing.chunkers.text_chunker import TextChunker
from services.llm.client import AnswerLLM, RerankLLM
from services.indexing.ocr_engine import SecurePDFProcessor
from services.progress_reporter import ProgressReporter
from services.llm.prompts.answer import DIRECT_LLM_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from services.retrieval.reranker import Reranker
from services.retrieval.retriever import LanceDBHybridRetriever
from _types.retrieval_types import ChunkCandidate, ContinueChoice, SearchResult
from utils import compute_file_md5, extract_doc_name, read_text_file, tokens_to_chars

from utils import get_file_logger
logger = get_file_logger(__file__)


class RAGPipeline:
    """完整的RAG流程"""

    def __init__(self, config: Config = Config()):
        self.config = config
        # 校验 HuggingFace 缓存路径（env var 已在模块顶层设置）
        if config.huggingface_cache_dir:
            if not os.path.isdir(config.huggingface_cache_dir):
                try:
                    os.makedirs(config.huggingface_cache_dir, exist_ok=True)
                except OSError:
                    raise ValueError(
                        f"huggingface_cache_dir 路径无效或无法创建: "
                        f"{config.huggingface_cache_dir}"
                    ) from None
        # ⚠️ NOTICE: RAGPipeline 应在销毁时调用 self.doc_manager.close() 释放 SQLite 连接。
        # 当前 pipeline 全程存活至进程退出，操作系统会自动回收资源，暂不触发问题。
        # 如果未来增加了销毁/重建 pipeline 实例的业务逻辑（如前端热重载），必须补上。
        self.doc_manager = DocManager(config)
        self.ocr_processor = SecurePDFProcessor(config)
        self.table_chunker = FinancialTableChunker(config)
        self.text_chunker = TextChunker(config)
        self.embedder = Embedder(config)
        self.retriever = LanceDBHybridRetriever(config)
        self.reranker = Reranker(config)
        # 检索管线中使用的 LLM 客户端
        self._llm_rerank = RerankLLM(config)
        self._llm_answer = AnswerLLM(config)
        # 管线暂停时保存的内部状态，等待用户做决定。
        #   是 search 方法给 continue_search 方法缓存的数据。
        # 结构: 
        #   {"query", "decision_type", "cross_encoder_top5", "top_score"}
        self._pending_state: Optional[Dict] = None

    # ── 文档管理（转发给 DocManager）────────────────────────

    def list_documents(self):
        """获取按分组建档的文档列表。"""
        return self.doc_manager.list_files()

    def set_document_file_type(self, file_md5: str, file_type: str) -> None:
        """更新文档的 file_type 字段。"""
        self.doc_manager.set_file_type(file_md5, file_type)

    def set_document_groups(self, file_md5: str, groups: str) -> None:
        """更新文档的分组。"""
        self.doc_manager.set_groups(file_md5, groups)

    def set_document_tags(self, file_md5: str, tags: str) -> None:
        """更新文档的标签。"""
        self.doc_manager.set_tags(file_md5, tags)

    def set_document_enabled(self, file_md5: str, enabled: bool) -> None:
        """启用/禁用单个文档。"""
        self.doc_manager.set_file_enabled(file_md5, enabled)

    def set_group_enabled(self, group_name: str, enabled: bool) -> None:
        """启用/禁用整个分组。"""
        self.doc_manager.set_group_enabled(group_name, enabled)

    def delete_document(self, file_md5: str) -> int:
        """删除文档的元数据和向量数据。"""
        return self.doc_manager.delete_document(file_md5)

    # ── 索引管线 ─────────────────────────────────────────

    def index_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        on_progress: Optional[Callable] = None,
    ):
        """
        索引单个文档（目前支持 PDF 或 TXT）。

        Args:
            file_path:   文档文件路径
            doc_id:      文档唯一标识符（可选，默认取文件 MD5）
            on_progress: 进度回调，签名为 (status_line: str | None, log_line: str | None)
        """
        progress_reporter = ProgressReporter(on_progress, logger=logger)
        doc_name = extract_doc_name(file_path)
        doc_name = doc_name or os.path.basename(file_path)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                'doc_id': doc_id or doc_name,
                'doc_name': doc_name,
                'error': '文件不存在',
                'success': False,
            }

        # 确定文档 ID
        file_md5 = compute_file_md5(file_path)
        document_id = doc_id or file_md5

        # MD5 去重提前到分块之前——避免同名/改名的文档重跑 OCR+LLM
        # 后才在 add_chunks 里被拦截，浪费 API 调用和用户时间。
        if self.doc_manager.has_document(file_md5):
            if on_progress:
                on_progress(
                    f"⏭️ 跳过索引：{doc_name}（已存在）",
                    f"MD5 重复，跳过索引: {doc_name}",
                )
            return {
                'doc_id': document_id,
                'doc_name': doc_name,
                'num_chunks': 0,
                'skipped': True,
                'success': True,
                'token_usage': None,
            }

        # 根据文件类型选择分块策略
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            chunks, chunking_tokens = self._chunk_image_pdf(
                file_path, document_id, doc_name, on_progress,
            )
        elif ext == '.txt':
            chunks, chunking_tokens = self._chunk_txt(
                file_path, document_id, doc_name, on_progress,
            )
        else:
            return {
                'doc_id': document_id,
                'doc_name': doc_name,
                'error': f'不支持的文件格式: {ext}',
                'success': False,
            }

        if not chunks:
            progress_reporter.report_task_end(f"❌ 索引失败：{doc_name}")
            return {
                'doc_id': document_id,
                'doc_name': doc_name,
                'num_chunks': 0,
                'success': False,
                'token_usage': chunking_tokens,
            }

        # ── 向量嵌入 + 入库（共用）──
        progress_reporter.report_phase_start(
            "向量嵌入中…",
            f"开始向量嵌入与入库，共 {len(chunks)} 个 chunk",
        )
        texts = [chunk['text'] for chunk in chunks]
        # 分批嵌入：大文档一次性 encode 可能长时间静默（日志与 UI 均无输出），
        # 批间汇报进度便于事后从日志定位卡点
        batch_size = self.config.embedding_batch_size
        total_texts = len(texts)
        embedding_parts: List[np.ndarray] = []
        for start in range(0, total_texts, batch_size):
            end = min(start + batch_size, total_texts)
            # 进度前置汇报：批次卡死/崩溃时，日志最后一行直接点名在飞的批次；
            # 批耗时由相邻两行时间差测得，全部完成由 phase_end 兜底确认。
            # 不走 ProgressReporter phase 接口——批次是阶段内部进度，不重置计时
            if on_progress:
                on_progress(f"向量嵌入中… {start + 1}-{end}/{total_texts}", None)
            logger.info("正在嵌入批次: %d-%d/%d", start + 1, end, total_texts)
            embedding_parts.append(self.embedder.encode(texts[start:end]))
        embeddings = np.vstack(embedding_parts)
        # 先写入向量（LanceDB）——chunk 数据是核心资产，优先落盘
        lancedb_result = self.doc_manager.add_chunks(chunks, embeddings)
        # 再写入元数据（SQLite）+ 记录操作历史
        self.doc_manager.add_document_meta(
            file_md5=document_id,
            file_name=doc_name or "",
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            chunks=chunks,
            lancedb_result=lancedb_result,
        )
        progress_reporter.report_phase_end("向量嵌入与入库完成")
        if lancedb_result.get('skipped'):
            progress_reporter.report_task_end(
                f"索引结束：此文档已在数据库中存在，跳过索引。",
            )
            return {
                'doc_id': document_id,
                'doc_name': doc_name,
                'num_chunks': len(chunks),
                'skipped': True,
                'success': True,
                'token_usage': chunking_tokens,
            }

        progress_reporter.report_task_end(
            f"索引完成：{doc_name} — {len(chunks)} chunks",
        )
        return {
            'doc_id': document_id,
            'doc_name': doc_name,
            'num_chunks': len(chunks),
            'success': True,
            'token_usage': chunking_tokens,
        }

    # ── 分块策略 ─────────────────────────────────────────

    def _chunk_image_pdf(
        self,
        file_path: str,
        document_id: str,
        doc_name: str,
        on_progress: Optional[Callable] = None,
    ):
        """PDF 分块：OCR → LLM 切块。返回 (chunks, token_usage)。"""
        progress_reporter = ProgressReporter(on_progress, logger=logger)

        # ── OCR 识别（逐页回调，汇报进度）──
        ocr_result = self.ocr_processor.try_process(file_path, on_progress=on_progress)
        if not ocr_result:
            return None, {"input": 0, "output": 0}

        # ── LLM 分块 ──
        progress_reporter.report_phase_start(
            "LLM 分块中…",
            f"开始 LLM 分块，共 {len(ocr_result) - 1} 页",
        )
        chunks, chunking_tokens = self.table_chunker.chunk_pages(
            ocr_result, document_id, doc_name,
        )
        progress_reporter.report_phase_end(
            f"LLM 分块完成，生成 {len(chunks)} 个 chunk",
        )
        return chunks, chunking_tokens

    def _chunk_txt(
        self,
        file_path: str,
        document_id: str,
        doc_name: str,
        on_progress: Optional[Callable] = None,
    ):
        """
        TXT 分块：委托给 TextChunker。
        返回 (chunks, token_usage)。
        """
        progress_reporter = ProgressReporter(on_progress, logger=logger)

        text = read_text_file(file_path)

        progress_reporter.report_phase_start(
            "LLM 分块中…",
            f"开始分析 TXT 章节格式，共 {len(text)} 字符",
        )
        chunks, chunking_tokens = self.text_chunker.chunk_text(
            text, document_id, doc_name,
        )

        if any("chapter_title" in c for c in chunks):
            progress_reporter.report_phase_end(
                f"章节识别完成，生成 {len(chunks)} 个 chunk",
            )
        else:
            progress_reporter.report_phase_end(
                f"章节识别失败，降级为段落切分，生成 {len(chunks)} 个 chunk",
            )
        return chunks, chunking_tokens

    # ═══════════════════════════════════════════════════════════
    # 检索管线（混合检索 → CrossEncoder二次排序 → LLM打分 → 间隙填充 → 生成）
    # ═══════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        history_context: Optional[List[Dict]] = None,
        on_progress: Optional[Callable] = None,
    ) -> SearchResult:
        """
        执行完整检索管线。

        开心路径：混合检索→CrossEncoder二次排序→LLM打分→间隙填充→生成，一次性返回最终答案。
        需要用户选择时：返回 SearchResult(status="needs_user_choice")，
        调用方展示选择后调用 continue_search() 继续。

        Args:
            query: 用户查询文本
            history_context: 可选的历史 Q&A 列表，格式为 [{"query", "answer", "sources": [...]}]
            on_progress: 进度回调，签名为 (status_line: str | None, log_line: str | None)

        Returns:
            SearchResult，status 区分三种情况
        """
        # 重置状态
        self._pending_state = None
        # 重置 token 计数器（每次 search 独立统计）
        self._llm_rerank.get_token_usage_and_reset()
        self._llm_answer.get_token_usage_and_reset()
        progress_reporter = ProgressReporter(on_progress, logger=logger)

        # ── 混合检索 ──
        progress_reporter.report_phase_start(None, "混合检索中…")
        query_vector = self.embedder.encode(query)
        # 只有 SQLite 中存在文档记录时才启用过滤（空表 = 首次运行，兼容旧数据）
        enabled_ids = (
            self.doc_manager.list_enabled_doc_ids()
            if self.doc_manager.has_any_docs() else None
        )
        hybrid_candidates = self.retriever.search(
            query, query_vector,
            top_k=self.config.hybrid_search_top_k,
            allowed_doc_ids=enabled_ids,
        )
        progress_reporter.report_phase_end(f"混合检索完成，{len(hybrid_candidates)} 个候选")
        # 零结果处理
        if not hybrid_candidates:
            self._pending_state = {
                "query": query,
                "decision_type": "zero_results",
            }
            progress_reporter.report_task_end("混合检索结果:无候选。等待用户决定。")
            return SearchResult(
                status="needs_user_choice",
                query=query,
                pending_decision="zero_results",
                token_usage={"input": 0, "output": 0},
            )

        # ── CrossEncoder 二次排序 ──
        progress_reporter.report_phase_start(
            "CrossEncoder 排序中…",
            f"开始 CrossEncoder 二次排序，{len(hybrid_candidates)} 个候选",
        )
        cross_encoder_candidates = self.reranker.cross_encoder_rerank(
            query, hybrid_candidates, top_k=self.config.crossencoder_top_k,
        )
        progress_reporter.report_phase_end(
            f"CrossEncoder 二次排序完成，取前 {len(cross_encoder_candidates)}",
        )
        # 保存 top-5 供低置信度场景使用
        cross_encoder_top5 = cross_encoder_candidates[:5]

        # ── LLM 打分 + 分级收网 ──
        progress_reporter.report_phase_start(
            "LLM 打分中…",
            f"开始 LLM 打分，共 {len(cross_encoder_candidates)} 个候选",
        )
        final_candidates = self.reranker.llm_score_and_sieve(
            query, cross_encoder_candidates, self._llm_rerank,
        )
        progress_reporter.report_phase_end(f"LLM 打分完成，{len(final_candidates)} 个入围")
        # 零结果处理
        if not final_candidates:
            top_score = max(
                (c.llm_score or 0) for c in cross_encoder_candidates
            )
            rerank_tokens = self._llm_rerank.get_token_usage_and_reset()
            self._pending_state = {  # 给continue_search() 提供缓存数据
                "query": query,
                "decision_type": "low_confidence",
                "cross_encoder_top5": cross_encoder_top5,
                "top_score": top_score,
                "rerank_token_usage": rerank_tokens,
            }
            progress_reporter.report_task_end("候选数据关联性过低，等待用户决定。")
            return SearchResult(
                status="needs_user_choice",
                query=query,
                pending_decision="low_confidence",
                top_score=top_score,
                token_usage=dict(rerank_tokens),
            )

        # ── 后处理：间隙填充 ──
        progress_reporter.report_phase_start(None, "间隙填充中…")
        merged = self._fill_gaps(final_candidates)
        fill_count = len(merged) - len(final_candidates)
        progress_reporter.report_phase_end(f"间隙填充完成，补充 {fill_count} 个 chunk")

        # ── RAG 生成回答 ──
        progress_reporter.report_phase_start(
            "生成回答中…",
            f"开始 RAG 生成回答，基于 {len(merged)} 个 chunk",
        )
        answer = self._answer_with_rag(query, merged, history_context=history_context)
        progress_reporter.report_phase_end("回答生成完成")

        # 收集两个 LLM 的 token 消耗
        rerank_tokens = self._llm_rerank.get_token_usage_and_reset()
        answer_tokens = self._llm_answer.get_token_usage_and_reset()
        token_usage = {
            "input": rerank_tokens["input"] + answer_tokens["input"],
            "output": rerank_tokens["output"] + answer_tokens["output"],
        }

        progress_reporter.report_task_end(
            f"✅ 回答完成 — {int(token_usage['input']):,}/{int(token_usage['output']):,} tokens",
        )

        return SearchResult(
            status="success",
            query=query,
            answer=answer,
            sources=merged,
            source_count=len(merged),
            token_usage=token_usage,
        )

    def continue_search(
        self,
        choice: ContinueChoice,
        history_context: Optional[List[Dict]] = None,
        on_progress: Optional[Callable] = None,
    ) -> SearchResult:
        """
        用户做出选择后继续执行检索管线。

        调用前必须已有 search() 返回的 needs_user_choice 结果。

        Args:
            choice: 用户选择（见 ContinueChoice 枚举）
                - ContinueChoice.ABANDON:    放弃本次查询
                - ContinueChoice.DIRECT_LLM: 不使用知识库，纯 LLM 回答
                - ContinueChoice.RAG:        使用低置信度候选 chunk 做 RAG 回答
            history_context: 可选的历史 Q&A 列表
            on_progress: 进度回调，签名为 (status_line: str | None, log_line: str | None)

        Returns:
            SearchResult，通常 status="success" 含 answer
        """
        try:
            # 没有缓存数据，直接返回错误
            state = self._pending_state
            if state is None:
                return SearchResult(
                    status="error",
                    query="",
                    error_message="没有待处理的用户选择，请先调用 search()",
                )

            query: str = state["query"]
            if choice == ContinueChoice.ABANDON:
                # 合并 search 阶段的 RerankLLM token
                rerank_tokens = state.get("rerank_token_usage", {"input": 0, "output": 0})
                if on_progress:
                    on_progress("用户已主动取消查询。", "用户已主动取消查询。")
                # status="success"：系统成功执行了用户的取消意图，从系统角度不算错误。
                # 后续 SearchResult 状态模型扩展后可改为 "user_abort" 等专用状态。
                return SearchResult(
                    status="success",
                    query=query,
                    answer="已取消本次查询。",
                    source_count=0,
                    token_usage=dict(rerank_tokens),
                )

            # ── DIRECT_LLM / RAG 共用 ProgressReporter ──
            progress_reporter = ProgressReporter(on_progress, logger=logger)
            if choice == ContinueChoice.DIRECT_LLM:
                # 合并 search 阶段的 RerankLLM token + AnswerLLM token
                rerank_tokens = state.get("rerank_token_usage", {"input": 0, "output": 0})
                self._llm_answer.get_token_usage_and_reset()
                progress_reporter.report_phase_start(
                    "生成回答中…",
                    "开始纯 LLM 生成回答（不使用知识库）",
                )
                answer = self._answer_direct(query, history_context=history_context)
                progress_reporter.report_phase_end("回答生成完成")
                answer_tokens = self._llm_answer.get_token_usage_and_reset()
                token_usage = {
                    "input": rerank_tokens["input"] + answer_tokens["input"],
                    "output": rerank_tokens["output"] + answer_tokens["output"],
                }
                progress_reporter.report_task_end(
                    f"✅ 回答完成 — {int(token_usage['input']):,}/{int(token_usage['output']):,} tokens",
                )
                return SearchResult(
                    status="success",
                    query=query,
                    answer=answer,
                    source_count=0,
                    token_usage=token_usage,
                )

            # 用户选择低置信度 RAG 回答
            if choice == ContinueChoice.RAG:
                decision_type: str = state["decision_type"]
                if decision_type != "low_confidence":
                    return SearchResult(
                        status="error",
                        query=query,
                        error_message="rag 选项仅适用于低置信度场景（当前场景: {})".format(
                            decision_type
                        ),
                    )

                # 填充数据块缝隙
                candidates: List[ChunkCandidate] = state.get("cross_encoder_top5", [])
                progress_reporter.report_phase_start(None, f"间隙填充中…（{len(candidates)} 个候选）")
                merged = self._fill_gaps(candidates)
                fill_count = len(merged) - len(candidates)
                progress_reporter.report_phase_end(f"间隙填充完成，补充 {fill_count} 个 chunk")

                # 重置 AnswerLLM token 计数器
                self._llm_answer.get_token_usage_and_reset()
                # 使用低置信度RAG数据，获取答案
                progress_reporter.report_phase_start(
                    "生成回答中…",
                    f"开始 RAG 生成回答，基于 {len(merged)} 个 chunk（低置信度）",
                )
                answer = self._answer_with_rag(
                    query, merged, low_confidence=True,
                    history_context=history_context,
                )
                progress_reporter.report_phase_end("回答生成完成")
                answer_tokens = self._llm_answer.get_token_usage_and_reset()
                # 合并 search 阶段的 RerankLLM token
                rerank_tokens = state.get("rerank_token_usage", {"input": 0, "output": 0})
                token_usage = {
                    "input": rerank_tokens["input"] + answer_tokens["input"],
                    "output": rerank_tokens["output"] + answer_tokens["output"],
                }
                progress_reporter.report_task_end(
                    f"✅ 回答完成 — {int(token_usage['input']):,}/{int(token_usage['output']):,} tokens",
                )

                return SearchResult(
                    status="success",
                    query=query,
                    answer=answer,
                    sources=merged,
                    source_count=len(merged),
                    token_usage=token_usage,
                )

            progress_reporter.report_task_end("程序内部错误，未知选择选项")
            logger.error(
                "continue_search 收到未知选项: %s（可选: %s）",
                choice.value, [c.value for c in ContinueChoice],
            )
            return SearchResult(
                status="error",
                query=query,
                error_message=(
                    f"未知选项: {choice.value}，"
                    f"可选: {[c.value for c in ContinueChoice]}"
                ),
            )

        finally:
            self._pending_state = None  # 双保险：无论哪个分支退出，必定清理

    def _fill_gaps_width_1_old(
        self, candidates: List[ChunkCandidate],
    ) -> List[ChunkCandidate]:
        """
        后处理：相邻 chunk 单间隙填充（P0-1）。

        两个入围 chunk 在文档内 chunk_index 仅隔 1 chunk（中间恰好一个未命中）
        → 从 LanceDB 读取间隙 chunk，追加到结果列表。
        通过 gap_fill_ratio 控制额外引入的 chunk 数量上限。

        本方法是旧版策略（固定填充宽度为1的gap），现已弃用。
        """
        if len(candidates) < 2:
            return candidates

        # 按 doc_id 再按 chunk_index 排序
        candidates.sort(key=lambda c: (c.doc_id, c.chunk_index))

        # 获取 LanceDB 表用于查询间隙 chunk
        try:
            table = self.retriever.table
            if table is None:
                table = self.retriever.db.open_table(self.retriever.table_name)
        except Exception:
            logger.warning("无法打开 LanceDB 表，跳过间隙填充")
            return candidates

        # 获取pandas DataFrame数据
        df = table.to_pandas()
        if df is None or df.empty:
            return candidates

        # 计算间隙填充数量上限
        original_count = len(candidates)
        max_fill = int(
            original_count * self.config.gap_fill_ratio + 0.9999
        )

        # 填充结果列表里的gap
        filled_count = 0
        result: List[ChunkCandidate] = []
        for i in range(len(candidates)):
            result.append(candidates[i])

            # 上限已达，后续不再填充（但仍追加原始候选）
            if filled_count >= max_fill:
                continue

            # 确定相邻两个 chunk
            a = candidates[i]
            if i + 1 >= len(candidates):
                continue  # 最后一个候选，没有后继
            b = candidates[i + 1]

            # 同文档且间隔恰好为 1 chunk（chunk_index 差 2）
            if a.doc_id != b.doc_id or b.chunk_index - a.chunk_index != 2:
                continue

            # 获取间隙 chunk 数据
            gap_idx = a.chunk_index + 1
            gap_row = df[
                (df["doc_id"] == a.doc_id)
                & (df["chunk_index"] == gap_idx)
            ]
            if gap_row.empty:
                continue
            row = gap_row.iloc[0]

            # 解析 LanceDB 字符串列
            try:
                gap_page_nums = ast.literal_eval(row["page_nums"])
                if not isinstance(gap_page_nums, list):
                    gap_page_nums = []
            except (ValueError, SyntaxError):
                gap_page_nums = []
            try:
                gap_keywords = ast.literal_eval(row["has_financial_keywords"])
                if not isinstance(gap_keywords, list):
                    gap_keywords = []
            except (ValueError, SyntaxError):
                gap_keywords = []

            # 构建 ChunkCandidate 对象（章节字段带旧表兼容防御，同 retriever）
            gap_chapter_title = row.get("chapter_title", "")
            gap_chapter_index = row.get("chapter_index", "")
            gap_chunk = ChunkCandidate(
                id=row["id"],
                text=row["text"],
                doc_id=row.get("doc_id", a.doc_id),
                doc_name=row.get("doc_name", a.doc_name),
                page_nums=gap_page_nums,
                chunk_index=int(row["chunk_index"]),
                length=int(row.get("length", 0)),
                type=row.get("type", "mixed"),
                has_financial_keywords=gap_keywords,
                chapter_title=(
                    gap_chapter_title if isinstance(gap_chapter_title, str) else ""
                ),
                chapter_index=(
                    gap_chapter_index if isinstance(gap_chapter_index, str) else ""
                ),
            )

            # 把gap的数据追加到结果列表
            result.append(gap_chunk)
            filled_count += 1
            logger.info(
                "间隙填充：%s 在 chunk %d 和 %d 之间插入 chunk %d（%d/%d）",
                a.doc_id[:16], a.chunk_index, b.chunk_index,
                gap_idx, filled_count, max_fill,
            )

        if filled_count > 0:
            logger.info(
                "间隙填充完成：原始 %d 个候选 → 额外引入 %d 个间隙 chunk",
                original_count, filled_count,
            )

        return result

    def _fill_gaps(
        self, candidates: List[ChunkCandidate],
    ) -> List[ChunkCandidate]:
        """
        后处理：间隙填充 + 邻居扩展（双策略，共享总量上限）。

        策略一（gap_filling）：同文档内入围 chunk 之间若间隔 ≤ gap_fill_width
        个空缺，逐条从 LanceDB 读取并补入。
        策略二（neighbor_extension_width）：每个入围 chunk 的邻居也拉入候选池。
        0=关闭，1=±1，2=±2。gap 填充先建语义连续块，邻居扩展再在块边缘外扩。

        gap_fill_width=0 且 neighbor_extension_width=0 时快速返回。
        两者额外引入的 chunk 数受 gap_fill_ratio 约束。
        """
        # 快速出口
        cfg = self.config
        if not candidates:
            return candidates
        if cfg.gap_fill_width == 0 and cfg.neighbor_extension_width == 0:
            return candidates

        # 计算间隙填充数量上限
        original = {c.id for c in candidates}
        candidates.sort(key=lambda c: (c.doc_id, c.chunk_index))
        original_count = len(candidates)
        max_fill = int(original_count * cfg.gap_fill_ratio + 0.9999)

        # 预先准备pandas DataFrame数据
        try:
            table = self.retriever.table
            if table is None:
                table = self.retriever.db.open_table(self.retriever.table_name)
        except Exception:
            logger.warning("无法打开 LanceDB 表，跳过间隙+邻居填充")
            return candidates
        df = table.to_pandas()
        if df is None or df.empty:
            return candidates

        # doc_id × chunk_index → row 快速查找
        row_map: Dict[tuple, Any] = {}
        for _, row in df.iterrows():
            row_map[(row.get("doc_id", ""), int(row["chunk_index"]))] = row

        def _build(row, doc_id, doc_name):
            try:
                pn = ast.literal_eval(row["page_nums"])
                if not isinstance(pn, list):
                    pn = []
            except (ValueError, SyntaxError):
                pn = []
            try:
                kw = ast.literal_eval(row["has_financial_keywords"])
                if not isinstance(kw, list):
                    kw = []
            except (ValueError, SyntaxError):
                kw = []
            ct = row.get("chapter_title", "")
            ci = row.get("chapter_index", "")
            return ChunkCandidate(
                id=row["id"], text=row["text"],
                doc_id=doc_id, doc_name=doc_name,
                page_nums=pn, chunk_index=int(row["chunk_index"]),
                length=int(row.get("length", 0)), type=row.get("type", "mixed"),
                has_financial_keywords=kw,
                chapter_title=ct if isinstance(ct, str) else "",
                chapter_index=ci if isinstance(ci, str) else "",
            )

        filled = 0
        fill_char_limit = tokens_to_chars(cfg.gap_fill_token_limit)
        filled_chars = 0
        gap_added, nb_added = 0, 0

        # 策略一：缺口填充
        if cfg.gap_fill_width > 0:
            fill_gap_result: List[ChunkCandidate] = []
            for i, a in enumerate(candidates):
                fill_gap_result.append(a)
                if filled >= max_fill or i + 1 >= len(candidates):
                    # 填充名额已用尽，就不再走下面的填充逻辑，直接不停的continue，把原始候选加入结果集。
                    # 如果是最后一个候选，也无需再找它后面是否还有缺口。
                    continue
                b = candidates[i + 1]
                gap_size = b.chunk_index - a.chunk_index - 1
                if a.doc_id != b.doc_id or gap_size < 1 or gap_size > cfg.gap_fill_width:
                    continue
                for off in range(1, gap_size + 1):
                    if filled >= max_fill:
                        break
                    key = (a.doc_id, a.chunk_index + off)
                    row = row_map.get(key)
                    if row is None:
                        continue
                    row_text = row.get("text", "")
                    # 字符硬上限——与 ratio 弹性上限同时作用
                    if filled_chars + len(row_text) > fill_char_limit:
                        break
                    gc = _build(row, a.doc_id, a.doc_name)
                    fill_gap_result.append(gc)
                    original.add(gc.id)
                    filled += 1; gap_added += 1
                    filled_chars += len(row_text)
                    logger.info(
                        "间隙填充：%s 在 chunk %d 和 %d 之间插入 chunk %d（%d/%d）",
                        a.doc_id[:16], a.chunk_index, b.chunk_index,
                        a.chunk_index + off, filled, max_fill,
                    )
            candidates = fill_gap_result

        # 策略二：邻居扩展（width=0 跳过，1=±1，2=±2）
        if cfg.neighbor_extension_width > 0:
            nbw = cfg.neighbor_extension_width
            neighbor_extension_result: List[ChunkCandidate] = list(candidates)
            for c in candidates:
                if filled >= max_fill:
                    break
                # range 从 1 开始，跳过 off=0（自身），双向同时覆盖
                radius = range(1, nbw + 1)
                for off in (*(-r for r in radius), *radius):
                    if filled >= max_fill:
                        break
                    key = (c.doc_id, c.chunk_index + off)
                    row = row_map.get(key)
                    if row is None or row["id"] in original:
                        continue
                    row_text = row.get("text", "")
                    if filled_chars + len(row_text) > fill_char_limit:
                        break
                    nc = _build(row, c.doc_id, c.doc_name)
                    neighbor_extension_result.append(nc)
                    original.add(nc.id)
                    filled += 1; nb_added += 1
                    filled_chars += len(row_text)
            candidates = neighbor_extension_result

        if gap_added or nb_added:
            logger.info(
                "后处理完成：原始 %d 个候选 → 间隙填充 %d + 邻居扩展 %d"
                "（合计 %d，上限 %d）",
                original_count, gap_added, nb_added, filled, max_fill,
            )
        # 邻居扩展后列表可能失序——重排保证下游 LLM 打分看到连续上下文
        candidates.sort(key=lambda c: (c.doc_id, c.chunk_index))
        return candidates

    # ── RAG 生成工具 ──────────────────────────────────────

    @staticmethod
    def _format_history_context(
        history_context: Optional[List[Dict]],
        turn_limit: int = 0,
    ) -> str:
        """
        将历史 Q&A 列表格式化为 prompt 用的文本块。

        Args:
            history_context: 历史 Q&A 列表，每项含 query/answer/sources（索引信息）
            turn_limit: 最多保留最近 N 轮，0 表示不限制

        Returns:
            格式化的历史对话文本。history_context 为空或 None 时返回空字符串。
        """
        if not history_context:
            return ""

        # 仅保留最近 N 轮（硬上限），防止历史上下文挤占 RAG 文档空间
        if turn_limit > 0 and len(history_context) > turn_limit:
            history_context = history_context[-turn_limit:]

        # 单条来源引用：定位符按有效性组装（同 _answer_with_rag 的规则），
        # 不产出"页码 []"这类空定位
        def _format_ref(s: Dict) -> str:
            loc_parts: List[str] = []
            if s.get("page_nums"):
                loc_parts.append(f"页码 {s['page_nums']}")
            if s.get("chapter_title"):
                loc_parts.append(f"章节 {s['chapter_title']}")
            loc_parts.append(f"chunk #{s.get('chunk_index', '?')}")
            return f"{s.get('doc_name', '?')} ({', '.join(loc_parts)})"

        parts: List[str] = ["[历史对话]"]
        for i, entry in enumerate(history_context):
            sources = entry.get("sources", [])
            refs = ", ".join(_format_ref(s) for s in sources) if sources else "无"
            parts.append(
                f"历史问答 {i + 1}:\n"
                f"  问题: {entry.get('query', '')}\n"
                f"  回答: {entry.get('answer', '')}\n"
                f"  参考文档: {refs}"
            )
        parts.append("---")
        return "\n".join(parts)

    def _answer_with_rag(
        self,
        query: str,
        chunks: List[ChunkCandidate],
        low_confidence: bool = False,
        history_context: Optional[List[Dict]] = None,
    ) -> str:
        """拼接 chunk 上下文，调用 LLM 生成带来源引用的回答。"""
        context_parts: List[str] = []
        for i, c in enumerate(chunks):
            # 定位符按有效性组装：页码（PDF）/章节（TXT）都可能缺失，
            # chunk # 恒在——绝不给 LLM 展示"页码 []"这类空定位，
            # 否则 LLM 会在回答中原样引用空页码
            loc_parts: List[str] = []
            if c.page_nums:
                loc_parts.append(f"页码 {c.page_nums}")
            if c.chapter_title:
                loc_parts.append(f"章节 {c.chapter_title}")
            loc_parts.append(f"chunk #{c.chunk_index}")
            source_label = f"来源: {c.doc_name or c.doc_id}, " + ", ".join(loc_parts)
            context_parts.append(
                f"--- 文档片段 {i + 1} ({source_label}) ---\n{c.text}"
            )
        # ⚠️ 上下文 trim 逻辑暂在 pipeline 层处理——正确的位置是 LLM client 层
        # （保护上下文是 client 的职责，不应依赖 pipeline 的实现细节）。
        # 当前仅此一个调用点，日后若有第二处 AnswerLLM 调用，应将此逻辑
        # 迁入 AnswerLLM，接受 chunks 列表 + 分隔符作为参数。
        context = "\n\n".join(context_parts)
        rag_limit_chars = tokens_to_chars(self.config.llm_context_rag_token_limit)
        if len(context) > rag_limit_chars and len(context_parts) > 2:
            # 从中间向两侧移除整段——保留首尾、切掉中间——直到 ≤ 上限
            while len(context_parts) > 2 and len("\n\n".join(context_parts)) > rag_limit_chars:
                context_parts.pop(len(context_parts) // 2)
            context_parts.insert(len(context_parts) // 2, "…")
            context = "\n\n".join(context_parts)

        # 组装 user message：历史（如有）+ 文档片段 + 查询
        # 历史采用"内嵌自定义格式"拼接在 user content 中，
        # 而非 API 消息数组中的多轮 assistant/user role 交替。
        # 原因：RAG 场景下每次检索的文档上下文不同，历史应作为"参考"
        # 而非"对话延续"。内嵌格式降低了历史回答在模型注意力中的权重，
        # 确保当前检索结果主导回答。
        history_block = self._format_history_context(
            history_context, turn_limit=self.config.history_turn_limit,
        )
        if history_block:
            user_message = (
                f"{history_block}\n\n"
                f"文档片段：\n\n{context}\n\n"
                f"请回答以下问题：{query}"
            )
        else:
            user_message = f"文档片段：\n\n{context}\n\n请回答以下问题：{query}"

        logger.info(
            "RAG 生成开始，共 %d 个 chunk 上下文，总长度 %d 字符",
            len(chunks), len(context),
        )
        answer = self._llm_answer.ask(RAG_SYSTEM_PROMPT, user_message)
        if low_confidence:
            answer = "⚠️ 低置信度提醒：以下回答基于匹配度较低的文档片段，可能不够准确。\n\n" + answer
        return answer

    def _answer_direct(
        self,
        query: str,
        history_context: Optional[List[Dict]] = None,
    ) -> str:
        """纯 LLM 回答（无 RAG 上下文）。"""
        logger.info("纯 LLM 回答开始（无 RAG 上下文）")
        history_block = self._format_history_context(
            history_context, turn_limit=self.config.history_turn_limit,
        )
        if history_block:
            user_message = f"{history_block}\n\n问题：{query}"
        else:
            user_message = f"问题：{query}"
        return self._llm_answer.ask(DIRECT_LLM_SYSTEM_PROMPT, user_message)


# ── 全局单例 ──────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """获取 RAGPipeline 全局单例（懒加载，Python 原生，不绑定框架）。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
