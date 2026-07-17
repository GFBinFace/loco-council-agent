"""
文本 Chunker — TXT 文件的章节感知切块。

LLM 识别章节标题 regex → 程序按 regex 全文拆分 → 子分块。
regex 提取失败时降级为按段落边界切分。
"""

import re

from config import Config
from services.llm.prompts.text_chunker import FIRST_ATTEMPT_PROMPT, SECOND_ATTEMPT_PROMPT

from utils import get_file_logger
logger = get_file_logger(__file__)


class TextChunker:
    """章节感知切块器——通过 LLM 识别章节标题模式，程序负责拆分。"""

    def __init__(self, config: Config = Config()):
        self.config = config
        from services.llm.client import ChunkingLLM
        self._llm = ChunkingLLM(config)

    # ── 公开入口 ──────────────────────────────────────────

    def chunk_text(
        self, text: str, doc_id: str, doc_name: str = "",
    ):
        """
        对 TXT 全文进行章节感知切块。

        Args:
            text:     TXT 文件全文
            doc_id:   文档唯一标识（文件 MD5）
            doc_name: 原始文件名

        Returns:
            (chunks, token_usage) —— chunk 列表和 LLM token 消耗。
        """
        self._llm.get_token_usage_and_reset()
        total_chars = len(text)
        logger.info(
            "开始 TXT 章节切块：doc=%s, 共 %d 字符", doc_id[:16], total_chars,
        )

        # ── 规则先行：内置 regex 家族零成本识别，未达门槛才升级 LLM ──
        regex = self._match_chapter_rules(text)
        if regex is None:
            regex = self._extract_chapter_regex(text)
        # LLM 调用（若发生）到此全部结束，统一结算 token 并落日志
        token_usage = self._llm.get_token_usage_and_reset()
        logger.info(
            "章节 regex 提取 LLM token 消耗: input=%d, output=%d",
            token_usage["input"], token_usage["output"],
        )

        # ── 用 regex 拆分章节 ──
        if regex:
            matches = list(re.finditer(regex, text, re.MULTILINE))
            if matches:
                chapters = []
                for i, m in enumerate(matches):
                    start = m.start()
                    end = (
                        matches[i + 1].start()
                        if i + 1 < len(matches)
                        else len(text)
                    )
                    chapters.append({
                        "title": m.group(0).strip(),
                        "text": text[start:end].strip(),
                    })
                logger.info("章节识别完成，共 %d 章", len(chapters))
                chunks = self._sub_chunk_chapters(chapters, doc_id, doc_name)
                logger.info("章节切块完成，共 %d 个 chunk", len(chunks))
                return chunks, token_usage

        # ── 降级：按段落边界切分 ──
        logger.warning(
            "TXT 章节 regex 提取失败，降级为段落切分（%s，%d 字符）",
            doc_name, total_chars,
        )
        chunks = self._chunk_by_paragraph(
            text, doc_id, doc_name,
            chunk_size=self.config.chunk_soft_max,
        )
        logger.info("段落切块完成，共 %d 个 chunk", len(chunks))
        return chunks, token_usage

    # ── 章节 regex 提取 ──────────────────────────────────

    # 内置章节规则（规则先行，LLM 兜底）：覆盖中文书主流标题格式。
    # 行首容纳缩进与短前缀（如卷名 '变脸之初卷 第一章 …'），
    # 行尾限长以排除正文中顺带提及"第X章"的长句。
    _CHAPTER_RULE_PATTERNS = (
        r"^.{0,20}第[一二三四五六七八九十百千零〇两\d０-９]+[章回节卷部篇].{0,30}$",
        r"^\s{0,8}\d{1,4}[、.．\s]\s*\S.{0,28}$",
        r"^\s{0,8}[Cc]hapter\s+\d+.{0,40}$",
    )

    def _match_chapter_rules(self, text: str):
        """
        用内置规则家族识别章节标题 regex，全部未达密度门槛返回 None。

        验证门槛比 LLM 层更严（txt_chapter_rule_min_density，默认每 5000
        字符至少 1 个标题）——规则是盲猜家族，宁可漏判交给 LLM，
        不可误判把正文切碎。
        """
        check_text = (
            text[:self.config.txt_chapter_boundary]
            if len(text) >= self.config.txt_chapter_boundary
            else text
        )
        if not check_text:
            return None
        required = max(
            1, len(check_text) // self.config.txt_chapter_rule_min_density,
        )
        for pattern in self._CHAPTER_RULE_PATTERNS:
            matches = re.findall(pattern, check_text, re.MULTILINE)
            if len(matches) >= required:
                logger.info(
                    "内置规则命中章节标题: %s（%d 个匹配，门槛 %d）",
                    pattern, len(matches), required,
                )
                return pattern
        logger.info(
            "内置规则未命中章节标题（门槛：每 %d 字符 ≥1），升级 LLM 提取",
            self.config.txt_chapter_rule_min_density,
        )
        return None

    # 取样参数：[首次样本量, 重试样本量, 展示窗口]，三者联动。暂无调整为配置项的需求。
    _SAMPLE_SIZES = (6000, 12000)
    _WINDOW = 3000

    def _extract_chapter_regex(self, text: str):
        """
        LLM 提取章节标题 regex。两次尝试，均失败返回 None。

        取样策略：取文本前 N 字符发给 LLM，但只展示首尾各 _WINDOW 字符
        （开头包含第一个章节标题，末尾提供第二个实例以确认模式）。
        第二次尝试附上第一次失败的 regex，引导 LLM 修正锚定。
        """
        failed_regex = "（上次调用未返回结果）"
        for attempt in range(2):
            sample = text[:self._SAMPLE_SIZES[attempt]]
            text_sample = (
                f"{sample[:self._WINDOW]}\n...\n{sample[-self._WINDOW:]}"
            )
            if attempt == 0:
                prompt = FIRST_ATTEMPT_PROMPT.format(text_sample=text_sample)
            else:
                prompt = SECOND_ATTEMPT_PROMPT.format(
                    text_sample=text_sample, failed_regex=failed_regex,
                )
            try:
                # 双日志：LLM 网络调用，开始与结果各记一条
                logger.info(
                    "开始 LLM 提取章节 regex（第 %d/2 次），样本 %d 字符",
                    attempt + 1, len(sample),
                )
                answer = self._parse_regex_answer(
                    self._llm.ask(
                        "你是一个精通正则表达式的文本处理专家。",
                        prompt,
                    )
                )
                failed_regex = answer or failed_regex
                if not answer or answer.upper() == "NONE":
                    logger.warning(
                        "LLM 判断样本中无章节标题（第 %d 次）", attempt + 1,
                    )
                    continue
                re.compile(answer)
                # 宽松验证：每 txt_chapter_llm_min_density 字符至少 1 个标题。
                # 中长篇仅截取头部 txt_chapter_boundary 字符，避免全文扫描。
                check_text = (
                    text[:self.config.txt_chapter_boundary]
                    if len(text) >= self.config.txt_chapter_boundary
                    else text
                )
                matches = re.findall(answer, check_text, re.MULTILINE)
                required = max(
                    1, len(check_text) // self.config.txt_chapter_llm_min_density,
                )
                if len(matches) >= required:
                    logger.info(
                        "章节 regex 提取成功: %s（%d 个匹配）",
                        answer, len(matches),
                    )
                    return answer
                logger.warning(
                    "章节 regex 验证失败（第 %d 次）：%d 个匹配，需要 ≥ %d；"
                    "LLM 返回: %s",
                    attempt + 1, len(matches), required, answer,
                )
            except Exception as exc:
                logger.warning(
                    "章节 regex 提取异常（第 %d 次）: %s", attempt + 1, exc,
                )
        return None

    @staticmethod
    def _parse_regex_answer(answer: str) -> str:
        """从 LLM 回答中提取 regex——取最后一个以 'regex:' 开头的行。

        提示词要求 LLM 先抄写标题行再归纳，最后一行以 'regex:' 输出结果；
        无 'regex:' 行时按旧协议处理整段回答（兼容直答的模型）。
        """
        for line in reversed(answer.strip().splitlines()):
            line = line.strip().strip("`").strip()
            if line.startswith("regex:"):
                return line[6:].strip()
        return answer.strip().strip("`").strip()

    # ── 章节子分块 ────────────────────────────────────────

    def _sub_chunk_chapters(
        self, chapters: list, doc_id: str, doc_name: str,
    ) -> list:
        """将每章按 chunk_soft_max 做子分块。"""
        chunks = []
        total = len(chapters)
        for i, ch in enumerate(chapters):
            ch_text = ch["text"]
            if len(ch_text) <= self.config.chunk_soft_max:
                chunks.append({
                    "id": f"{doc_id}_chunk_{len(chunks)}",
                    "text": ch_text,
                    "type": "text",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_nums": [],
                    "chunk_index": len(chunks),
                    "length": len(ch_text),
                    "chapter_title": ch["title"],
                    "chapter_index": f"{i + 1}/{total}",
                    "has_financial_keywords": [],
                })
            else:
                sub_chunks = self._chunk_by_paragraph(
                    ch_text, doc_id, doc_name,
                    start_index=len(chunks),
                    chunk_size=self.config.chunk_soft_max,
                )
                for sc in sub_chunks:
                    sc["chapter_title"] = ch["title"]
                    sc["chapter_index"] = f"{i + 1}/{total}"
                chunks.extend(sub_chunks)
        return chunks

    # ── 降级切分：按段落贪心聚合 ─────────────────────────────

    @staticmethod
    def _chunk_by_paragraph(
        text: str,
        doc_id: str,
        doc_name: str,
        start_index: int = 0,
        chunk_size: int = 4000,
    ) -> list:
        """
        按段落边界贪心聚合切分——绝不从中切开任何一个段落。

        行为：用双换行（\\n\\n）将文本拆为段落，逐段贪心追加到当前
        chunk，累计长度超过 chunk_size 时在段落边界处关闭当前 chunk、
        另开新 chunk。超大的单一段落（自身 > chunk_size）不受限制地
        作为独立 chunk 带走（宁肯语义完整，不把叙事从中砍断）。

        替换原因（2026-07-16）：之前用 rfind/find 在 chunk_size 附近
        搜索段落边界，后向窗口不足时向前扩展——两个方向都可能把段落
        从中切开，被硬切的叙事单元无法在任何 chunk 中找到完整描述。
        参见 memory/chunkers-design.md 事故档案。
        """
        paragraphs = text.split("\n\n")
        chunks = []
        idx = start_index
        buf: list[str] = []
        buf_len = 0

        def _emit():
            nonlocal buf, buf_len
            segment = "\n\n".join(buf).strip()
            if segment:
                chunks.append({
                    "id": f"{doc_id}_chunk_{idx + len(chunks)}",
                    "text": segment,
                    "type": "text",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_nums": [],
                    "chunk_index": idx + len(chunks),
                    "length": len(segment),
                    "has_financial_keywords": [],
                })
            buf = []
            buf_len = 0

        for para in paragraphs:
            plen = len(para)
            if buf_len + plen > chunk_size and buf:
                _emit()
            buf.append(para)
            buf_len += plen
        if buf:
            _emit()
        return chunks
