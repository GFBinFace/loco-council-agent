"""
LanceDB 混合检索器。

LanceDB 中存储的 chunk 数据结构（与 retrieval_types.ChunkCandidate 对应）：

    id          str    — chunk 唯一 ID，格式 "{file_md5}_chunk_{index}"
    text        str    — chunk 完整文本
    vector      float[]— 1024 维 BGE-M3 嵌入向量
    type        str    — "text" | "table" | "mixed"
    doc_id      str    — 所属文档的 MD5
    doc_name    str    — 文档原始文件名
    page_nums   str    — 来源页码，存储为字符串形式，如 "[1, 2]"，读取时用 ast.literal_eval 还原
    chunk_index int    — 文档内 chunk 序号（0-based）
    length      int    — 文本字符数
    has_financial_keywords  str — 含有的财务关键词，存储为字符串形式，如 "['资产', '负债']"
"""

import ast
from typing import List, Optional

import lancedb
import numpy as np
from lancedb.rerankers import RRFReranker  # LanceDB 内置 RRF 重排序

from config import Config
from _types.retrieval_types import ChunkCandidate

from utils import get_file_logger
logger = get_file_logger(__file__)

class LanceDBHybridRetriever:
    """LanceDB混合检索器（向量 + BM25）- 修正版"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.db = lancedb.connect(config.lance_db_dir)
        self.table_name = "financial_docs"
        self.table = None
        self.reranker = RRFReranker()

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 30,
        allowed_doc_ids: Optional[List[str]] = None,
    ) -> List[ChunkCandidate]:
        """混合检索（向量 + BM25，RRF 融合）。

        Args:
            query: 用户查询文本（用于 BM25 全文检索）
            query_vector: 查询向量（用于语义搜索，由 Embedder.encode 产生）
            top_k: 返回结果数量
            allowed_doc_ids: 限定检索的文档 ID 列表。None 表示不限定，空列表则不检索任何文档。

        Returns:
            ChunkCandidate 列表，按 RRF 融合分数降序排列。
            表不存在时返回空列表。
        """
        # 每次检索都重新 open_table——既保证索引后新 FTS 数据立即可见
        # （LanceDB 0.16 Table 对象内部缓存了旧的 FTS index handle），
        # 又保证搜完即释放句柄（table=None），避免句柄在 Windows 上
        # 长期持有文件锁阻塞后续索引的 LanceDB 写入。
        # open_table 是纯元数据操作（~0.2ms），不加载向量或 FTS 数据。
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            logger.warning("LanceDB 表 %s 不存在，返回空结果", self.table_name)
            return []

        logger.info(
            "混合检索开始，query=%.80s...，top_k=%d", query, top_k,
        )

        # 检索数据
        builder = (
            self.table.search(query_type="hybrid")
            .vector(query_vector.tolist())
            .text(query)
            .rerank(reranker=self.reranker)
            .limit(top_k)
        )
        if allowed_doc_ids is not None:
            if not allowed_doc_ids:
                # 没有启用的文档，无需检索任何文档，直接返回空列表。
                return []
            ids_str = ", ".join(f"'{did}'" for did in allowed_doc_ids)
            builder = builder.where(f"doc_id IN ({ids_str})") # 限定检索的文档 ID
        results_df = builder.to_pandas()

        # 将 LanceDB 结果转换为 ChunkCandidate 列表
        candidates: List[ChunkCandidate] = []
        for _, row in results_df.iterrows():
            # LanceDB 存储时用 str() 序列化了列表，这里用 ast.literal_eval 还原
            try:
                page_nums = ast.literal_eval(row["page_nums"])
                if not isinstance(page_nums, list):
                    page_nums = []
            except (ValueError, SyntaxError):
                page_nums = []

            try:
                has_financial_keywords = ast.literal_eval(row["has_financial_keywords"])
                if not isinstance(has_financial_keywords, list):
                    has_financial_keywords = []
            except (ValueError, SyntaxError):
                has_financial_keywords = []

            # 章节字段：旧版建的表无此列，Series.get 缺省空串；
            # 再做类型防御（NaN 等非字符串一律归空）
            chapter_title = row.get("chapter_title", "")
            chapter_index = row.get("chapter_index", "")
            candidates.append(ChunkCandidate(
                id=row["id"],
                text=row["text"],
                doc_id=row.get("doc_id", ""),
                doc_name=row.get("doc_name", ""),
                page_nums=page_nums,
                chunk_index=int(row["chunk_index"]),
                length=int(row.get("length", 0)),
                type=row.get("type", "mixed"),
                has_financial_keywords=has_financial_keywords,
                chapter_title=chapter_title if isinstance(chapter_title, str) else "",
                chapter_index=chapter_index if isinstance(chapter_index, str) else "",
                hybrid_score=float(row.get("_relevance_score", 0.0)),
            ))

        logger.info(
            "混合检索完成，返回 %d 个候选 chunk", len(candidates),
        )
        # 搜完释放句柄——LanceDB 0.16 没有 close()，
        # 依赖 CPython 引用计数立即回收 + Rust Drop 关闭文件句柄
        self.table = None
        return candidates

