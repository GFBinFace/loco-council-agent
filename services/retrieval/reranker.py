"""
重排序器（Reranker）。

封装 CrossEncoder二次排序 和 LLM打分与分级收网，
对外暴露统一的重排序入口。
"""

from typing import Any, Dict, List

import numpy as np
from sentence_transformers import CrossEncoder

from config import Config
from services.llm.client import RerankLLM
from _types.retrieval_types import ChunkCandidate

from utils import get_file_logger
logger = get_file_logger(__file__)


class Reranker:
    """重排序器：CrossEncoder二次排序 + LLM打分与分级收网。"""

    def __init__(self, config: Config = Config()):
        import os
        # 模型由 download_models.py 预下载到 HF_HOME，禁止运行时联网检查
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.config = config
        # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 已在 pipeline.__init__ 中强制设置
        self.model = CrossEncoder(config.rerank_model)

    # ── CrossEncoder 二次排序 ──────────────────────────────────

    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[ChunkCandidate],
        top_k: int = 10,
    ) -> List[ChunkCandidate]:
        """CrossEncoder 二次排序并收窄候选集。

        对混合检索的候选 chunk 用本地 BGE-Reranker 模型重新打分，
        按分数降序取前 top_k 个。

        Args:
            query: 用户查询文本
            candidates: 混合检索输出的候选 chunk 列表
            top_k: 收窄后的数量上限

        Returns:
            经 CrossEncoder 重打分后的 chunk 列表（已填充 cross_encoder_score），
            按分数降序排列，长度 ≤ top_k。
            候选为空时返回空列表。
        """
        if not candidates:
            return []

        # 构建 (query, text) 对
        pairs = [(query, c.text) for c in candidates]

        # predict 返回 list[float]，分数越高越相关
        logger.info("CrossEncoder 二次排序开始，%d 个候选", len(candidates))
        scores: List[float] = self.model.predict(pairs)  # type: ignore[assignment]

        # 赋值 cross_encoder_score
        for c, score in zip(candidates, scores):
            c.cross_encoder_score = float(score)

        # 按分数降序排序
        candidates.sort(key=lambda c: c.cross_encoder_score or 0.0, reverse=True)

        # 收窄到 top_k
        result = candidates[:top_k]
        logger.info(
            "CrossEncoder 二次排序完成，%d → %d 个候选",
            len(candidates), len(result),
        )

        return result

    # ── LLM 打分 + 分级收网 ────────────────────────────────

    def llm_score_and_sieve(
        self,
        query: str,
        candidates: List[ChunkCandidate],
        rerank_llm: RerankLLM,
    ) -> List[ChunkCandidate]:
        """LLM 打分并执行分级收网。

        Args:
            query: 用户查询文本
            candidates: CrossEncoder二次排序输出的候选 chunk 列表
            rerank_llm: RerankLLM 实例，用于调用 LLM 打分

        Returns:
            分级收网后入围的 chunk 列表（已填充 llm_score 和 llm_reason）。
            所有 chunk 得分 < 5 时返回空列表（触发低置信度流程）。
            候选为空时返回空列表。
        """
        if not candidates:
            return []

        # 调用 LLM 打分（RerankLLM 仅负责通信，返回原始分数列表）
        raw_scores = rerank_llm.score_chunks(query, candidates)

        # 将 LLM 返回的分数与候选 chunk 对齐，缺失的用 -1 补齐
        score_map: Dict[str, Dict[str, Any]] = {}
        for s in raw_scores:
            cid = s.get("chunk_id", "")
            score_map[cid] = {
                "chunk_id": cid,
                "score": s.get("score", -1),
                "reason": s.get("reason", ""),
            }
        for c in candidates:
            item = score_map.get(c.id)
            if item:
                c.llm_score = item["score"]
                c.llm_reason = item["reason"]
            else:
                c.llm_score = -1
                c.llm_reason = "LLM 未返回该 chunk 的评分"

        scored = [c for c in candidates if c.llm_score is not None and c.llm_score >= 0]
        logger.info(
            "LLM 打分完成，%d 个候选，有效分数 %d 个（范围 %d-%d）",
            len(candidates),
            len(scored),
            min(c.llm_score for c in scored) if scored else -1,
            max(c.llm_score for c in scored) if scored else -1,
        )

        # 分级收网
        return self._sieve_by_tiers(candidates)

    def _sieve_by_tiers(self, candidates: List[ChunkCandidate]) -> List[ChunkCandidate]:
        """分级收网。

        按蓝图 §3.4 逻辑：
        N = 候选数，对每个阈值（8, 7, 6, 5）：
          如果 ≥阈值的数量 > N/2，取全部 ≥阈值 → 结束
          兜底（最后一个阈值 5）→ 不论数量多少都取

        如果取 ≥5 后仍为空（即所有 chunk 得分 < 5），返回空列表。
        """
        if not candidates:
            return []

        N = len(candidates)
        half_N = N / 2
        tiers = list(self.config.rerank_score_tiers)  # (8, 7, 6, 5)

        for i, tier in enumerate(tiers):
            qualified = [c for c in candidates if c.llm_score is not None and c.llm_score >= tier]
            count = len(qualified)

            is_last_tier = (i == len(tiers) - 1)

            if count > half_N or is_last_tier:
                logger.info(
                    "分级收网：阈值 ≥%d，%d/%d 个 chunk 入围（%s）",
                    tier, count, N,
                    "兜底" if is_last_tier else f"超过半数({half_N:.1f})",
                )
                return qualified

        # 不会到达这里（最后一个 tier 一定能返回），但保持防御
        return []
