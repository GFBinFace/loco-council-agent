"""检索管线数据契约。

定义贯穿 混合检索→CrossEncoder二次排序→LLM打分与分级收网→间隙填充→生成
全链路的 chunk 数据对象，各阶段逐步填入对应的 score 字段，无需拆装。
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Literal, Optional


@dataclass
class ChunkCandidate:
    """贯穿检索管线的 chunk 数据对象。

    对应 LanceDB 中的一行，在内存中流经各阶段。
    各阶段逐步填入对应的 score 字段：

        混合检索:       hybrid_score
        CrossEncoder二次排序: cross_encoder_score
        LLM打分:        llm_score, llm_reason

    page_nums 和 has_financial_keywords 从 LanceDB 的字符串列解析而来，
    由 retriever.search() 负责还原为 Python list。
    """

    # ── 标识字段（LanceDB 出库时填充）──
    id: str
    text: str
    doc_id: str
    doc_name: str
    page_nums: List[int]           # 来源页码，1-based
    chunk_index: int               # 文档内 chunk 序号（0-based，用于间隙填充）
    length: int                    # 文本长度（字符数）
    type: str                      # "text" | "table" | "mixed"
    has_financial_keywords: List[str]  # 含有的财务关键词

    # ── 章节定位（TXT 章节分块时有值，其余为空串）──
    chapter_title: str = ""        # 所属章节标题，如 "天界卷 第三百二十九章 终之章(上)"
    chapter_index: str = ""        # 章节序号，如 "329/330"

    # ── 混合检索写入（RRF 融合分数）──
    hybrid_score: float = 0.0

    # ── CrossEncoder二次排序写入 ──
    cross_encoder_score: Optional[float] = None

    # ── LLM打分写入（0-10 分）──
    llm_score: Optional[int] = None
    llm_reason: Optional[str] = None


@dataclass
class SearchResult:
    """search() / continue_search() 的返回结果。

    通过 status 字段区分三种状态：

    - "success":      正常完成，answer 有值
    - "needs_user_choice": 需要用户选择，pending_decision 指示场景
    - "error":        执行出错，error_message 有值
    """

    status: Literal["success", "needs_user_choice", "error"]
    query: str

    # ── status == "success" 时 ──
    answer: Optional[str] = None
    sources: List[ChunkCandidate] = field(default_factory=list)
    source_count: int = 0

    # ── status == "needs_user_choice" 时 ──
    # "zero_results"  — 混合检索返回 0 个候选
    # "low_confidence" — LLM 打分后所有 chunk 得分 < 5
    pending_decision: Optional[str] = None
    # low_confidence 场景下的最高 LLM 分数，展示给用户参考
    top_score: Optional[int] = None

    # ── status == "error" 时 ──
    error_message: Optional[str] = None

    # ── token 消耗（所有状态均可能填充）──
    token_usage: Optional[Dict[str, int]] = None  # {"input": N, "output": N}


# ═══════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════

class ContinueChoice(StrEnum):
    """continue_search() 的用户选择。

    当 search() 返回 status="needs_user_choice" 时，
    调用方展示选项后，用此枚举调用 continue_search()。
    """

    # 放弃本次查询（零结果和低置信度场景均可用）
    ABANDON = "abandon"
    # 不使用知识库资料，由 AI 直接回答（零结果和低置信度场景均可用）
    DIRECT_LLM = "direct_llm"
    # 使用低置信度候选 chunk 做 RAG 回答（仅低置信度场景可用）
    RAG = "rag"
