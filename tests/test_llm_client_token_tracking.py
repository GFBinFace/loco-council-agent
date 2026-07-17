"""测试 _BaseLLM token 累加器行为。

验证 _call_with_retry 自动累加 token、get_token_usage_and_reset 返回并清零、
跨批次多次调用累加。
"""

from unittest.mock import MagicMock, patch

import pytest

from config import Config


# ── 辅助函数 ──────────────────────────────────────────────


def _make_response_mock(prompt: int, completion: int) -> MagicMock:
    """构造模拟的 OpenAI chat.completions.create 返回值。"""
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    resp = MagicMock()
    resp.usage = usage
    return resp


# ── 测试类 ──────────────────────────────────────────────────


class TestBaseLLMTokenAccumulation:
    """验证 _BaseLLM._call_with_retry 中的 token 累加器。"""

    @pytest.fixture
    def client(self):
        """构造 AnswerLLM 实例（轻量子类，便于测试）。"""
        from services.llm.client import AnswerLLM
        cfg = Config()
        return AnswerLLM(cfg)

    def test_single_call_accumulates_tokens(self, client):
        """单次 API 调用后 token 正确累加。"""
        resp = _make_response_mock(prompt=100, completion=50)
        with patch.object(client._client.chat.completions, "create", return_value=resp):
            client.get_token_usage_and_reset()
            client._call_with_retry(model="test", messages=[])
            usage = client.get_token_usage_and_reset()
        assert usage == {"input": 100, "output": 50}

    def test_multiple_calls_accumulate_across_batches(self, client):
        """多次调用跨批次累加 token。"""
        resp1 = _make_response_mock(prompt=200, completion=80)
        resp2 = _make_response_mock(prompt=150, completion=60)
        with patch.object(
            client._client.chat.completions, "create", side_effect=[resp1, resp2]
        ):
            client.get_token_usage_and_reset()
            client._call_with_retry(model="test", messages=[])
            client._call_with_retry(model="test", messages=[])
            usage = client.get_token_usage_and_reset()
        assert usage == {"input": 350, "output": 140}

    def test_get_token_usage_and_reset_clears_counter(self, client):
        """get_token_usage_and_reset 返回后内部计数器清零。"""
        resp = _make_response_mock(prompt=50, completion=25)
        with patch.object(client._client.chat.completions, "create", return_value=resp):
            client.get_token_usage_and_reset()
            client._call_with_retry(model="test", messages=[])
            first = client.get_token_usage_and_reset()
            assert first == {"input": 50, "output": 25}
            second = client.get_token_usage_and_reset()
            assert second == {"input": 0, "output": 0}

    def test_response_without_usage_does_not_crash(self, client):
        """usage 为 None 时不影响累加器（防御性处理）。"""
        resp = MagicMock()
        resp.usage = None
        with patch.object(client._client.chat.completions, "create", return_value=resp):
            client.get_token_usage_and_reset()
            client._call_with_retry(model="test", messages=[])
            usage = client.get_token_usage_and_reset()
        assert usage == {"input": 0, "output": 0}


class TestChunkingLLMTokenAccumulation:
    """验证 ChunkingLLM（通过 FinancialTableChunker）的 token 累加。"""

    @pytest.fixture
    def chunking_llm(self):
        from services.llm.client import ChunkingLLM
        cfg = Config()
        return ChunkingLLM(cfg)

    def test_chunking_llm_inherits_accumulator(self, chunking_llm):
        """ChunkingLLM 通过基类继承 token 累加器。"""
        resp = _make_response_mock(prompt=500, completion=200)
        with patch.object(
            chunking_llm._client.chat.completions, "create", return_value=resp
        ):
            chunking_llm.get_token_usage_and_reset()
            chunking_llm._call_with_retry(model="test", messages=[])
            usage = chunking_llm.get_token_usage_and_reset()
        assert usage == {"input": 500, "output": 200}


class TestRerankLLMTokenAccumulation:
    """验证 RerankLLM 跨批次 token 累加。"""

    @pytest.fixture
    def rerank_llm(self):
        from services.llm.client import RerankLLM
        cfg = Config()
        return RerankLLM(cfg)

    def test_rerank_llm_inherits_accumulator(self, rerank_llm):
        """RerankLLM 通过基类继承 token 累加器。"""
        resp = _make_response_mock(prompt=300, completion=100)
        with patch.object(
            rerank_llm._client.chat.completions, "create", return_value=resp
        ):
            rerank_llm.get_token_usage_and_reset()
            rerank_llm._call_with_retry(model="test", messages=[])
            usage = rerank_llm.get_token_usage_and_reset()
        assert usage == {"input": 300, "output": 100}
