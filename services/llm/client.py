"""
LLM 客户端模块 — 三个独立业务类。

每个类封装各自业务场景的底层模型调用，未来可独立切换模型。
当前阶段三个类底层均使用 DeepSeek（OpenAI 兼容接口）。

- ChunkingLLM : 分块业务（桩代码，供未来从 FinancialTableChunker 提取）
- RerankLLM   : Rerank 打分业务（LLM打分与分级收网）
- AnswerLLM   : 最终回答业务（步骤 4 的 RAG 生成 / 纯 LLM 回答）
"""

import json
import os
import re
import time
from typing import Any, Dict, List

from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from config import Config
from services.llm.prompts.rerank import SCORING_SYSTEM_PROMPT, SCORING_USER_TEMPLATE
from _types.retrieval_types import ChunkCandidate
from services.llm.tools.submit_scores import SUBMIT_SCORES_TOOL

from utils import get_file_logger
logger = get_file_logger(__file__)


# ═══════════════════════════════════════════════════════════════
# 基类：共享底层设施
# ═══════════════════════════════════════════════════════════════

class _BaseLLM:
    """
    LLM 客户端基类。

    封装 OpenAI 客户端初始化、指数退避重试、JSON 响应解析。
    子类通过参数注入各自的 model、api_key、base_url 和重试策略。
    """

    def __init__(
        self,
        config: Config,
        model: str,
        api_key: str,
        base_url: str,
        max_retries: int,
        retry_base_delay: float,
    ):
        self.config = config
        self._model = model
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        _api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not _api_key:
            logger.warning("DEEPSEEK_API_KEY 未设置，%s 将无法工作", type(self).__name__)
        self._client = OpenAI(api_key=_api_key, base_url=base_url)
        # Token 消耗累加器（input=prompt_tokens, output=completion_tokens）
        self._token_usage: Dict[str, int] = {"input": 0, "output": 0}

    # ── 重试逻辑（复制自 financial_chunker.py）────────────────

    def _call_with_retry(self, **kwargs: Any) -> Any:
        """
        带指数退避的 LLM 调用。

        可重试：网络超时、5xx、429（限流）
        不重试：4xx（参数错误、认证失败等）

        所有派生类必须通过本方法发起 LLM 通信，
        否则 token 统计等附加功能可能失效。
        """
        max_retries = self._max_retries
        base_delay = self._retry_base_delay
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                # 累加 token 消耗（跨批次、跨调用自动汇总）
                usage = response.usage
                if usage:
                    self._token_usage["input"] += usage.prompt_tokens or 0
                    self._token_usage["output"] += usage.completion_tokens or 0
                return response
            except (APIConnectionError, RateLimitError, InternalServerError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM 调用失败（%s），%d/%d 次重试，等待 %.1fs",
                        type(e).__name__, attempt + 1, max_retries, delay,
                    )
                    time.sleep(delay)
            except APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM 调用失败（HTTP %d），%d/%d 次重试，等待 %.1fs",
                            e.status_code, attempt + 1, max_retries, delay,
                        )
                        time.sleep(delay)
                else:
                    logger.error(
                        "LLM 调用失败（HTTP %d），4xx 错误不重试，"
                        "响应: %s",
                        e.status_code,
                        getattr(e, "body", getattr(e, "message", "无详情")),
                    )
                    raise  # 4xx 不重试
            except Exception as e:
                last_error = e
                logger.exception(
                    "LLM 调用遇到未知异常（%s），%d/%d 次重试",
                    type(e).__name__, attempt + 1, max_retries,
                )
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)

        logger.error(
            "LLM 调用重试 %d 次后仍失败（最后错误: %s）",
            max_retries, type(last_error).__name__,
        )
        raise last_error  # type: ignore[misc]

    def ask(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """
        发送单轮 system + user 消息，返回 LLM 回答文本。

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            temperature: 采样温度
            max_tokens: 生成 token 上限

        Returns:
            LLM 生成的文本。

        Raises:
            Exception: 重试耗尽后仍失败时向上抛出，由调用方决定降级策略。
        """
        response = self._call_with_retry(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = response.usage
        logger.info(
            "%s.ask 完成，input_tokens=%d，output_tokens=%d",
            type(self).__name__,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        return response.choices[0].message.content or ""

    def get_token_usage_and_reset(self) -> Dict[str, int]:
        """
        读取并重置本实例的累计 token 消耗。

        返回当前累计的 {"input": N, "output": N}，
        然后将内部计数器归零（为下一次操作做准备）。
        """
        result = dict(self._token_usage)
        self._token_usage = {"input": 0, "output": 0}
        return result

    # ── JSON 解析工具 ────────────────────────────────────────

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """
        从 LLM 文本响应中提取 JSON 对象。

        防御性处理：去除 markdown 代码块标记，处理空响应。
        解析失败返回空 dict。
        """
        if not raw:
            return {}
        raw = raw.strip()
        # 去除 markdown 代码块包裹
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM 返回非 JSON 文本，原始内容前 200 字符: %s", raw[:200])
            return {}


# ═══════════════════════════════════════════════════════════════
# ChunkingLLM — 桩代码
# ═══════════════════════════════════════════════════════════════

class ChunkingLLM(_BaseLLM):
    """
    分块业务 LLM 客户端。

    当前为桩代码，供未来从 FinancialTableChunker 提取 LLM 调用逻辑。
    在检索管线中不使用此类。
    """

    def __init__(self, config: Config = Config()):
        super().__init__(
            config,
            model=config.chunking_model,
            api_key=config.chunking_api_key,
            base_url=config.chunking_base_url,
            max_retries=config.chunking_max_retries,
            retry_base_delay=config.chunking_retry_base_delay,
        )


# ═══════════════════════════════════════════════════════════════
# RerankLLM — LLM 打分
# ═══════════════════════════════════════════════════════════════

class RerankLLM(_BaseLLM):
    """
    Rerank 打分业务 LLM 客户端。

    用于 LLM打分与分级收网：对 CrossEncoder二次排序后的候选 chunk 逐一打分（0-10 分制）。
    每批最多 10 个 chunk，超出则分批调用。
    """

    # 单批最大 chunk 数量，防止超出 LLM 上下文窗口
    BATCH_SIZE: int = 10

    def __init__(self, config: Config = Config()):
        super().__init__(
            config,
            model=config.rerank_llm_model,
            api_key=config.rerank_llm_api_key,
            base_url=config.rerank_llm_base_url,
            max_retries=config.chunking_max_retries,
            retry_base_delay=config.chunking_retry_base_delay,
        )

    def score_chunks(
        self, query: str, chunks: List[ChunkCandidate]
    ) -> List[Dict[str, Any]]:
        """
        调用 LLM 对候选 chunk 逐一打分，返回原始评分列表。

        LLM 仅负责通信——分批发送、接收 tool call，
        不做分数与输入 chunk 的对齐（对齐由 Reranker 负责）。

        Returns:
            LLM 返回的原始 scores 列表，顺序不定，可能缺失或多余。
            调用方（Reranker）负责将分数匹配到具体的 ChunkCandidate。
        """
        all_scores: List[Dict[str, Any]] = []
        total_batches = (len(chunks) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        batch_no = 0

        for batch_start in range(0, len(chunks), self.BATCH_SIZE):
            batch_no += 1
            batch = chunks[batch_start:batch_start + self.BATCH_SIZE]
            logger.info("LLM 打分 第 %d/%d 批，%d 个 chunk", batch_no, total_batches, len(batch))
            all_scores.extend(self._score_one_batch(query, batch))

        return all_scores

    def _score_one_batch(
        self, query: str, chunks: List[ChunkCandidate]
    ) -> List[Dict[str, Any]]:
        """
        对单批 chunk 调用 LLM 打分（通过 tool calling 强制结构化输出）。

        仅返回 LLM 原始响应，不做对齐和补齐。
        通信异常时返回空列表。
        """
        chunk_blocks_parts: List[str] = []
        for i, c in enumerate(chunks):
            preview = c.text[:1200]
            if len(c.text) > 1200:
                preview += "…[截断]"
            chunk_blocks_parts.append(
                f"--- 片段 {i + 1} (id: {c.id}) ---\n{preview}"
            )

        user_content = SCORING_USER_TEMPLATE.format(
            query=query,
            chunk_count=len(chunks),
            chunk_blocks="\n\n".join(chunk_blocks_parts),
        )

        # 通信层失败（重试耗尽的网络/超时/HTTP 错误）直接向上抛——
        # 契约：LLM 正常作答走返回值；未能作答抛异常，由调用链顶层
        # （Controller）统一转为"检索失败"，绝不伪装成"没有相关内容"。
        # 下方 tool call / 文本解析的兜底属于"作答了但格式差"的业务容错，保留。
        response = self._call_with_retry(
            model=self._model,
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            tools=[SUBMIT_SCORES_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "submit_scores"},
            },
        )

        usage = response.usage
        logger.info(
            "LLM 打分完成，input_tokens=%d，output_tokens=%d",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            if len(msg.tool_calls) != 1:
                logger.warning("预期 1 个 tool call，实际收到 %d 个", len(msg.tool_calls))
                return []
            tc = msg.tool_calls[0]
            if tc.function.name != "submit_scores":
                logger.warning("预期 tool 名为 submit_scores，实际为 %s", tc.function.name)
                return []
            try:
                data = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning("LLM 评分 tool call 参数非 JSON")
                return []
            return data.get("scores", [])
        else:
            logger.warning("LLM 未调用 submit_scores tool，尝试文本解析")
            raw = msg.content or ""
            data = self._parse_json_response(raw)
            return data.get("scores", [])


# ═══════════════════════════════════════════════════════════════
# AnswerLLM — 最终回答
# ═══════════════════════════════════════════════════════════════

class AnswerLLM(_BaseLLM):
    """
    最终回答 LLM 客户端——纯通信代理。

    只负责发送 system prompt + user message，返回 LLM 回答文本。
    业务逻辑（上下文拼接、prompt 选择、低置信度处理）由 pipeline 负责。

    契约（继承自 _BaseLLM.ask）：正常作答走返回值；未能作答（含网络错误）
    抛异常，由 Controller 统一转为 error 结果——绝不把失败伪装成回答文本，
    否则假回答会以 success 状态落入对话历史。
    """

    def __init__(self, config: Config = Config()):
        super().__init__(
            config,
            model=config.answer_model,
            api_key=config.answer_api_key,
            base_url=config.answer_base_url,
            max_retries=config.chunking_max_retries,
            retry_base_delay=config.chunking_retry_base_delay,
        )

