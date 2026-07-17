"""测试 TextChunker — TXT 章节感知切块（规则先行，LLM 兜底）。

覆盖：内置规则命中（含卷名前缀/缩进）、规则密度门槛拒绝、
LLM 兜底成功/失败/无效 regex、重试携带失败 regex、空文本边界，
以及 ChunkingLLM 接口存在性回归守卫（曾因缺失 ask() 导致章节路径全程静默降级）。
"""

from unittest.mock import MagicMock

from config import Config
from services.indexing.chunkers.text_chunker import TextChunker
from services.llm.client import ChunkingLLM

# 主流格式样本：带卷名前缀 + 正文全角空格缩进（复刻真实事故文件形态）
_SAMPLE_MAINSTREAM = (
    "变脸之初卷 第一章 武士和处男\n"
    + "　　罗迪一向是一个很本分的人。" * 30 + "\n\n"
    + "变脸之初卷 第二章 失败的拦路抢劫\n"
    + "　　剧情发展，冲突升级。" * 30 + "\n\n"
    + "变脸之初卷 第三章 帝都豪门\n"
    + "　　真相大白，故事结束。" * 30 + "\n"
)

# 长尾格式样本：内置规则无法命中，必须走 LLM
_SAMPLE_EXOTIC = (
    "【卷一】风起\n\n" + "武士登场，故事开始。" * 30 + "\n\n"
    + "【卷二】云涌\n\n" + "剧情发展，冲突升级。" * 30 + "\n\n"
    + "【卷三】尘落\n\n" + "真相大白，故事结束。" * 30 + "\n"
)


def _make_chunker(ask_mock: MagicMock) -> TextChunker:
    """构造 LLM 被 mock 掉的 TextChunker。"""
    chunker = TextChunker(Config())
    llm = MagicMock()
    llm.ask = ask_mock
    llm.get_token_usage_and_reset.return_value = {"input": 10, "output": 5}
    chunker._llm = llm
    return chunker


def test_chunking_llm_has_ask_interface():
    # 回归守卫：TextChunker 依赖 ChunkingLLM.ask，
    # 该方法曾不存在导致章节识别路径全程静默降级
    assert callable(getattr(ChunkingLLM, "ask", None))


# ── 内置规则路径 ──────────────────────────────────────────


def test_text_chunker_rule_hits_prefixed_titles_without_llm():
    # 带卷名前缀的主流格式由内置规则命中，LLM 一次都不调用
    ask = MagicMock()
    chunker = _make_chunker(ask)
    chunks, _ = chunker.chunk_text(_SAMPLE_MAINSTREAM, "doc1", "test.txt")
    ask.assert_not_called()
    assert len(chunks) == 3
    assert [c["chapter_index"] for c in chunks] == ["1/3", "2/3", "3/3"]
    assert all("第" in c["chapter_title"] for c in chunks)


def test_text_chunker_rule_density_too_low_escalates_to_llm():
    # 12400+ 字符只有 1 个标题 → 密度低于 1/5000 门槛 → 规则拒绝，升级 LLM
    sparse = "第一章 孤章\n" + ("　　正文。" * 8 + "\n") * 300
    assert len(sparse) > 10000
    ask = MagicMock(side_effect=Exception("测试中断"))
    chunker = _make_chunker(ask)
    chunks, _ = chunker.chunk_text(sparse, "doc1", "test.txt")
    assert ask.called
    assert all("chapter_title" not in c for c in chunks)  # LLM 失败 → 段落降级


# ── LLM 兜底路径 ──────────────────────────────────────────


def test_text_chunker_llm_regex_answer_line_parsed():
    # LLM 按新协议回答（先抄标题行，最后 regex: 行）→ 正确提取并切块
    answer = "找到的标题行：\n【卷一】风起\n\nregex: ^【卷.+?】.*$"
    chunker = _make_chunker(MagicMock(return_value=answer))
    chunks, token_usage = chunker.chunk_text(_SAMPLE_EXOTIC, "doc1", "test.txt")
    assert len(chunks) == 3
    assert [c["chapter_title"] for c in chunks] == ["【卷一】风起", "【卷二】云涌", "【卷三】尘落"]
    assert token_usage == {"input": 10, "output": 5}


def test_text_chunker_retry_prompt_carries_failed_regex():
    # 第一次 regex 可编译但 0 命中 → 第二次提示词应携带失败的 regex
    ask = MagicMock(side_effect=[
        "regex: ^ZZZ不存在的标题$",
        "regex: ^【卷.+?】.*$",
    ])
    chunker = _make_chunker(ask)
    chunks, _ = chunker.chunk_text(_SAMPLE_EXOTIC, "doc1", "test.txt")
    assert ask.call_count == 2
    second_prompt = ask.call_args_list[1][0][1]
    assert "^ZZZ不存在的标题$" in second_prompt
    assert len(chunks) == 3  # 第二次的 regex 生效


def test_text_chunker_llm_failure_falls_back_to_paragraph():
    chunker = _make_chunker(MagicMock(side_effect=Exception("网络错误")))
    chunks, _ = chunker.chunk_text(_SAMPLE_EXOTIC, "doc1", "test.txt")
    assert len(chunks) > 0
    assert all("chapter_title" not in c for c in chunks)


def test_text_chunker_invalid_regex_falls_back_to_paragraph():
    # LLM 返回不可编译的 regex → 两次尝试均失败 → 降级段落切分
    chunker = _make_chunker(MagicMock(return_value="regex: ((("))
    chunks, _ = chunker.chunk_text(_SAMPLE_EXOTIC, "doc1", "test.txt")
    assert len(chunks) > 0
    assert all("chapter_title" not in c for c in chunks)


def test_text_chunker_llm_answers_none_falls_back():
    # LLM 判断无章节标题（regex: NONE）→ 降级段落切分，不报错
    chunker = _make_chunker(MagicMock(return_value="regex: NONE"))
    chunks, _ = chunker.chunk_text(_SAMPLE_EXOTIC, "doc1", "test.txt")
    assert len(chunks) > 0
    assert all("chapter_title" not in c for c in chunks)


def test_text_chunker_empty_text_returns_empty_list():
    chunker = _make_chunker(MagicMock(return_value="regex: ^第.+?章.*$"))
    chunks, _ = chunker.chunk_text("", "doc1", "test.txt")
    assert chunks == []


def test_chunk_by_paragraph_never_splits_paragraphs():
    # 三段落，每段 ~160 字符，chunk_size 设 500 → 聚合后 2 chunk；
    # 每个段落必须完整出现在某个 chunk 中，不被切开
    marker_a, marker_b, marker_c = "[段落A]", "[段落B]", "[段落C]"
    para_fill = "　　正文填充。" * 30  # ~160 chars/para
    text = "\n\n".join([
        f"{marker_a}\n{para_fill}",
        f"{marker_b}\n{para_fill}",
        f"{marker_c}\n{para_fill}",
    ])
    chunks = TextChunker._chunk_by_paragraph(
        text, "d", "n", chunk_size=500,
    )
    assert len(chunks) == 2
    full_text = "\n".join(c["text"] for c in chunks)
    # 每个段落的标记词必须完整出现（说明没有被切碎）
    for marker in (marker_a, marker_b, marker_c):
        assert marker in full_text, f"{marker} 被切碎了"
