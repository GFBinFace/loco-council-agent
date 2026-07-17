"""测试 llm_client LLM 客户端。"""

import json
from unittest.mock import MagicMock, patch

import pytest

from _types.retrieval_types import ChunkCandidate


def _make_chunk(chunk_id: str = "c0", text: str = "测试文本") -> ChunkCandidate:
    """构造测试用 ChunkCandidate。"""
    return ChunkCandidate(
        id=chunk_id, text=text, doc_id="abc", doc_name="t.pdf",
        page_nums=[1], chunk_index=0, length=len(text),
        type="mixed", has_financial_keywords=[],
    )


def _mock_openai_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """构造 mock OpenAI chat.completions.create 返回值（文本响应）。"""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    mock_resp.choices[0].message.tool_calls = None
    mock_resp.usage = MagicMock()
    mock_resp.usage.prompt_tokens = prompt_tokens
    mock_resp.usage.completion_tokens = completion_tokens
    mock_resp.usage.total_tokens = prompt_tokens + completion_tokens
    return mock_resp


def _mock_tool_response(scores: list[dict], prompt_tokens: int = 100, completion_tokens: int = 50):
    """构造 mock OpenAI chat.completions.create 返回值（tool call 响应）。"""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = None
    tc = MagicMock()
    tc.function.name = "submit_scores"
    tc.function.arguments = json.dumps({"scores": scores})
    mock_resp.choices[0].message.tool_calls = [tc]
    mock_resp.usage = MagicMock()
    mock_resp.usage.prompt_tokens = prompt_tokens
    mock_resp.usage.completion_tokens = completion_tokens
    mock_resp.usage.total_tokens = prompt_tokens + completion_tokens
    return mock_resp


# ═══════════════════════════════════════════════════════════════
# RerankLLM
# ═══════════════════════════════════════════════════════════════

class TestRerankLLMScoreChunks:
    """RerankLLM.score_chunks() 测试"""

    @pytest.fixture
    def rerank_llm(self):
        """创建 mock OpenAI 的 RerankLLM。"""
        with patch("services.llm.client.OpenAI") as mock_openai:
            from services.llm.client import RerankLLM
            llm = RerankLLM()
            llm._client = mock_openai.return_value
            yield llm

    def test_score_chunks_returns_correct_structure(self, rerank_llm):
        """正常场景：LLM 通过 tool call 返回评分 → 正确解析"""
        chunks = [_make_chunk(f"c{i}", f"文本内容{i}") for i in range(3)]
        mock_resp = _mock_tool_response([
            {"chunk_id": "c0", "score": 8, "reason": "高度相关"},
            {"chunk_id": "c1", "score": 5, "reason": "部分相关"},
            {"chunk_id": "c2", "score": 2, "reason": "弱相关"},
        ])
        rerank_llm._client.chat.completions.create.return_value = mock_resp

        result = rerank_llm.score_chunks("查询", chunks)

        assert len(result) == 3
        assert result[0] == {"chunk_id": "c0", "score": 8, "reason": "高度相关"}
        assert result[1]["score"] == 5
        assert result[2]["score"] == 2

    def test_score_chunks_handles_tool_call_json_error(self, rerank_llm):
        """LLM tool call 参数非 JSON → 返回空列表（上层负责补齐 -1）"""
        chunks = [_make_chunk("c0", "文本")]
        mock_resp = _mock_tool_response([{"chunk_id": "c0", "score": 8}])
        mock_resp.choices[0].message.tool_calls[0].function.arguments = "不是合法JSON"
        rerank_llm._client.chat.completions.create.return_value = mock_resp

        result = rerank_llm.score_chunks("查询", chunks)

        assert result == []

    def test_score_chunks_raw_return_may_be_incomplete(self, rerank_llm):
        """LLM 只返回部分 chunk → 不做对齐和补齐，原样返回"""
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        mock_resp = _mock_tool_response([
            {"chunk_id": "c0", "score": 8, "reason": "好"},
            {"chunk_id": "c2", "score": 3, "reason": "差"},
        ])
        rerank_llm._client.chat.completions.create.return_value = mock_resp

        result = rerank_llm.score_chunks("查询", chunks)

        # 只返回 2 条，不做补齐（对齐由 Reranker 负责）
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c0"
        assert result[1]["chunk_id"] == "c2"

    def test_score_chunks_wrong_tool_name_returns_empty(self, rerank_llm):
        """tool call 名字不对 → 返回空列表"""
        chunks = [_make_chunk("c0")]
        mock_resp = _mock_tool_response([{"chunk_id": "c0", "score": 8}])
        mock_resp.choices[0].message.tool_calls[0].function.name = "wrong_tool"
        rerank_llm._client.chat.completions.create.return_value = mock_resp

        result = rerank_llm.score_chunks("查询", chunks)

        assert result == []

    def test_score_chunks_batches_large_input(self, rerank_llm):
        """超过 10 个 chunk 时分批调用 LLM"""
        chunks = [_make_chunk(f"c{i}") for i in range(12)]
        mock_resp = _mock_tool_response(
            [{"chunk_id": f"c{i}", "score": 7, "reason": "OK"} for i in range(10)]
        )
        rerank_llm._client.chat.completions.create.return_value = mock_resp

        result = rerank_llm.score_chunks("查询", chunks)

        # 12 个 chunk，BATCH_SIZE=10 → 2 批，每批返回 10 条，共 20 条
        assert rerank_llm._client.chat.completions.create.call_count == 2
        assert len(result) == 20

    def test_score_chunks_llm_call_failure_raises(self, rerank_llm):
        """契约：通信层失败（重试耗尽）向上抛异常，不伪装成空列表。

        空列表会被上层解读为"没有相关内容"→ 低置信度分支，
        把基础设施故障伪装成业务结论。
        """
        chunks = [_make_chunk("c0")]
        rerank_llm._client.chat.completions.create.side_effect = Exception("网络错误")

        with pytest.raises(Exception, match="网络错误"):
            rerank_llm.score_chunks("查询", chunks)


# ═══════════════════════════════════════════════════════════════
# AnswerLLM
# ═══════════════════════════════════════════════════════════════

class TestAnswerLLMAsk:
    """AnswerLLM.ask() 测试——纯通信代理"""

    @pytest.fixture
    def answer_llm(self):
        """创建 mock OpenAI 的 AnswerLLM。"""
        with patch("services.llm.client.OpenAI") as mock_openai:
            from services.llm.client import AnswerLLM
            llm = AnswerLLM()
            llm._client = mock_openai.return_value
            yield llm

    def test_ask_sends_prompts_and_returns_answer(self, answer_llm):
        """正常场景：传入 system_prompt 和 user_message，返回 LLM 答案"""
        mock_resp = _mock_openai_response("根据文档，资产负债表显示...")
        answer_llm._client.chat.completions.create.return_value = mock_resp

        answer = answer_llm.ask("系统提示", "用户消息")

        call_kwargs = answer_llm._client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "系统提示"}
        assert messages[1] == {"role": "user", "content": "用户消息"}
        assert answer == "根据文档，资产负债表显示..."

    def test_ask_llm_failure_raises(self, answer_llm):
        """契约：通信层失败（重试耗尽）向上抛异常，不返回伪装回答。

        伪装文案会以 status=success 落入对话历史，
        由 Controller 统一转为 error 结果才是正路。
        """
        answer_llm._client.chat.completions.create.side_effect = Exception("超时")

        with pytest.raises(Exception, match="超时"):
            answer_llm.ask("系统提示", "用户消息")
