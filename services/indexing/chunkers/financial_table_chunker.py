"""
金融文档 LLM 驱动切块器。

核心设计：
- 逐批将带行号前缀的 markdown 页面发送给 DeepSeek LLM
- LLM 通过决策表判断表格/文本区域，返回 line_range + context_summary + header_source
- 代码按 line_range 提取原文内容，组装最终的 chunk 文本
- fetch_page tool 支持 LLM 跨页查找表头
"""

import json
import logging
import re
from typing import Dict, List, Optional

from config import Config, LogConfig
from utils import write_debug_data, write_debug_data_lines
from services.llm.prompts.financial_chunker import SYSTEM_PROMPT
from services.llm.tools.fetch_page import FETCH_PAGE_TOOL
from services.llm.tools.submit_chunks import SUBMIT_CHUNKS_TOOL

from utils import get_file_logger
logger = get_file_logger(__file__)

# ============================================================
# 金融关键词 — 用于标记 chunk 是否含关键财务术语
# ============================================================

FINANCIAL_KEYWORDS = (
    '营业收入', '净利润', '毛利率', '资产负债率',
    'ROE', 'ROA', '现金流', '应收账款',
    '应付账款', '存货', '固定资产', '无形资产',
)

# ============================================================
# FinancialTableChunker
# ============================================================


class FinancialTableChunker:
    """金融文档专用切块器 — LLM 驱动，保留语义完整性"""

    def __init__(self, config: Config = Config()):
        self.config = config
        # 工作状态（chunk_pages 调用期间有效）
        self._pages: List[str] = []
        self._doc_id: str = ""      # 文件 MD5
        self._doc_name: str = ""    # 原始文件名
        self._total_pages: int = 0
        self._fetch_count: int = 0
        self._chapter_map: Optional[List[Dict]] = None

        # 统一使用 ChunkingLLM 封装 LLM 通信（重试、连接）
        from services.llm.client import ChunkingLLM
        self._llm = ChunkingLLM(config)

        # Debug data log
        self._debug_logger: Optional[logging.Logger] = None
        if LogConfig.DEBUG_DATA_MODE != "off":
            from utils import get_debug_data_logger
            self._debug_logger = get_debug_data_logger(__file__)

    # ── 公开入口 ──────────────────────────────────────────

    def chunk_pages(
        self, pages: List[str], doc_id: str, doc_name: str = ""
    ):
        """
        对 PDF 页面列表进行 LLM 驱动切块。

        Args:
            pages:    每页 markdown 文本，1-based（pages[0] 为占位空串）
            doc_id:   文档唯一标识（文件 MD5）
            doc_name: 原始文件名（可读标识）

        Returns:
            (chunks, token_usage) —— chunk 列表和 LLM token 消耗。
        """
        self._llm.get_token_usage_and_reset()
        self._pages = pages
        self._doc_id = doc_id
        self._doc_name = doc_name
        self._total_pages = len(pages) - 1  # 减去 [0] 占位
        self._fetch_count = 0

        # 1. 收集所有真实页码（跳过 [0] 占位）。空白页保留作为天然隔断
        all_page_nums = list(range(1, len(pages)))
        non_empty_pages = [pn for pn in all_page_nums if pages[pn].strip()]
        if not non_empty_pages:
            logger.info("所有页面均为空，跳过切块")
            return [], {"input": 0, "output": 0}

        logger.info(
            "开始 LLM 切块：doc=%s, 总页=%d, 非空页=%d",
            doc_id, self._total_pages, len(non_empty_pages),
        )

        # 2. 章节地图（可选）——仅基于非空页构建
        if self.config.chapter_map_mode != "off":
            self._chapter_map = self._build_chapter_map(non_empty_pages)

        # 3. 分批处理（空白页保留，LLM 看到 [空白页] 标记作为天然隔断）
        all_chunks: List[Dict] = []
        batch_size = self.config.pages_per_batch
        # 保留批次信息供行覆盖校验使用：[([batch_pages], [batch_chunks]), ...]
        batch_info: List[tuple] = []

        for batch_start in range(0, len(all_page_nums), batch_size):
            batch_pages = all_page_nums[batch_start : batch_start + batch_size]
            batch_chunks = self._process_batch(batch_pages)
            all_chunks.extend(batch_chunks)
            batch_info.append((list(batch_pages), batch_chunks))
            logger.info(
                "  批次完成：第 %d-%d 页 → %d 个 chunk",
                batch_pages[0], batch_pages[-1], len(batch_chunks),
            )

        # 4. 行覆盖校验 — 两级回退策略
        all_chunks = self._validate_and_fix_coverage(batch_info, all_chunks)

        # 5. 编号
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_index"] = i
            chunk["id"] = f"{doc_id}_chunk_{i}"

        token_usage = self._llm.get_token_usage_and_reset()
        logger.info("切块完成：共 %d 个 chunk", len(all_chunks))
        self._debug_dump_chunks(all_chunks)
        return all_chunks, token_usage

    def _debug_dump_chunks(self, all_chunks: List[Dict]) -> None:
        """调试用：将全部 chunk 的完整内容和元数据写入 data log"""
        if not self._debug_logger:
            return

        lines = [f"\n{'═' * 60}"]
        lines.append(f"  切块结果汇总：{len(all_chunks)} 个 chunk")
        lines.append(f"{'═' * 60}\n")

        for c in all_chunks:
            idx = c.get("chunk_index", "?")
            ctype = c.get("type", "?")
            pages = c.get("page_nums", [])
            length = c.get("length", 0)
            segs = c.get("_segments", [])
            cs = c.get("_context_summary")
            hs = c.get("_header_source")

            # 头部
            seg_desc = ", ".join(
                f"p{s['page']} L{s['line_range'][0]}-L{s['line_range'][1]}"
                for s in segs
            )
            lines.append(f"┌─ Chunk #{idx} ───────────────────────────────")
            lines.append(f"│ 类型: {ctype}  |  页: {pages}  |  {length} 字符  |  {len(segs)} seg")
            if cs:
                lines.append(f"│ 概述: {cs}")
            if hs:
                lines.append(f"│ 表头: page={hs.get('page')}, lines={hs.get('lines')}")
            lines.append(f"│ segments: {seg_desc}")
            lines.append(f"├─ 内容 ────────────────────────────────────")

            # 正文
            text = c.get("text", "")
            for tline in text.split("\n"):
                lines.append(f"│ {tline}")

            lines.append(f"└──────────────────────────────────────────\n")

        write_debug_data_lines(self._debug_logger, lines)

    # ── 批次处理 ──────────────────────────────────────────

    def _process_batch(self, page_nums: List[int]) -> List[Dict]:
        """处理一批页面：格式化 → LLM（含 tool calling）→ 组装"""
        self._fetch_count = 0  # 每批重置 fetch 计数

        # 格式化输入
        user_content = self._format_batch_for_llm(page_nums)

        # 调用 LLM — 正常路径通过 submit_chunks tool 返回 List[Dict]，
        # 兜底路径通过文本解析返回 List[Dict]
        try:
            chunk_defs = self._call_llm_with_tools(user_content, page_nums)
        except Exception:
            logger.exception(
                "批次 LLM 调用失败（pages=%s）", page_nums,
            )
            raise

        # 按 line_range 提取原文，组装最终 chunk
        assembled = self._assemble_chunks(chunk_defs)

        # 硬切超限 chunk：长度超过 chunk_hard_max 的强制按规则切分
        assembled = self._split_oversized_chunks(assembled)

        # 过滤越界 chunk：丢弃 segments 中引用非本批页面的 chunk
        batch_set = set(page_nums)
        filtered = []
        for c in assembled:
            seg_pages = {s["page"] for s in c.get("_segments", [])}
            if seg_pages - batch_set:
                logger.warning(
                    "  丢弃越界 chunk：pages=%s 不在本批 %s",
                    sorted(seg_pages - batch_set), page_nums,
                )
                continue
            filtered.append(c)

        if len(filtered) < len(assembled):
            logger.info(
                "  越界过滤：%d → %d 个 chunk",
                len(assembled), len(filtered),
            )

        return filtered

    # ── 行覆盖校验 ────────────────────────────────────────

    def _validate_and_fix_coverage(
        self, batch_info: List[tuple], all_chunks: List[Dict]
    ) -> List[Dict]:
        """行覆盖校验 + 两级回退处理。

        逐批校验 → 漏行 → 两级回退：
          第一级：整批驳回重做（漏行 > threshold 且未耗尽重试次数）
          第二级：专项修补（漏行 ≤ threshold 或第一级耗尽）
        """
        for bi, (batch_pages, batch_chunks) in enumerate(batch_info):
            missing = self._find_missing_lines(batch_pages, batch_chunks)
            if not missing:
                continue

            total_missing = sum(len(v) for v in missing.values())
            logger.warning(
                "  批次 %d（第 %d-%d 页）有 %d 个非空行未被覆盖",
                bi, batch_pages[0], batch_pages[-1], total_missing,
            )
            write_debug_data(
                self._debug_logger,
                f"批次 {bi} 行覆盖校验失败：{total_missing} 个漏行 → "
                + ", ".join(f"第{p}页 L{ls}" for p, ls in missing.items()),
            )

            # ── 两级回退入口 ──
            if total_missing <= self.config.coverage_direct_fix_threshold:
                logger.info(
                    "  → 漏行 %d ≤ 阈值 %d，直接走第二级专项修补",
                    total_missing, self.config.coverage_direct_fix_threshold,
                )
                write_debug_data(
                    self._debug_logger,
                    "决策：漏行 ≤ 阈值 → 直接走第二级专项修补",
                )
                fixed_chunks = self._fix_missing_lines_tier2(
                    missing, batch_chunks
                )
                # 替换本批 chunk
                all_chunks = self._replace_batch_chunks(
                    all_chunks, batch_chunks, fixed_chunks
                )

            else:
                write_debug_data(
                    self._debug_logger,
                    f"决策：漏行 {total_missing} > 阈值 → 走第一级整批驳回重做",
                )
                fixed_chunks = self._retry_batch_tier1(
                    batch_pages, missing, batch_chunks
                )
                all_chunks = self._replace_batch_chunks(
                    all_chunks, batch_chunks, fixed_chunks
                )

        return all_chunks

    @staticmethod
    def _replace_batch_chunks(
        all_chunks: List[Dict], old_batch: List[Dict], new_batch: List[Dict]
    ) -> List[Dict]:
        """在 all_chunks 中用 new_batch 逐位置替换 old_batch 的 chunk。

        通过对象身份（id）定位，因为 chunk dict 尚未编号，id 字段为空。
        找到 old_batch 中每个 chunk 在 all_chunks 中的位置，按序替换为 new_batch。
        """
        if not old_batch:
            return all_chunks

        # 找到 old_batch 第一个和最后一个 chunk 在 all_chunks 中的位置
        first_id = id(old_batch[0])
        last_id = id(old_batch[-1])
        start_idx = None
        end_idx = None

        for i, c in enumerate(all_chunks):
            if id(c) == first_id:
                start_idx = i
            if id(c) == last_id:
                end_idx = i

        if start_idx is None or end_idx is None:
            logger.warning("无法定位旧批次 chunk 位置，追加到末尾")
            return all_chunks + new_batch

        logger.info(
            "  替换批次 chunk：位置 [%d:%d]（%d 个旧 chunk → %d 个新 chunk）",
            start_idx, end_idx + 1, len(old_batch), len(new_batch),
        )
        return all_chunks[:start_idx] + new_batch + all_chunks[end_idx + 1:]

    def _find_missing_lines(
        self, batch_pages: List[int], batch_chunks: List[Dict]
    ) -> Dict[int, List[int]]:
        """找出 batch_pages 中未被 batch_chunks 覆盖的非空行。

        Returns:
            {page_num: [line_num, ...]} 仅包含有漏行的页
        """
        missing: Dict[int, List[int]] = {}

        for pn in batch_pages:
            page_text = self._pages[pn]
            page_lines = page_text.split("\n")

            # 本页所有非空行号（1-based）
            non_empty: set[int] = {
                i + 1 for i, line in enumerate(page_lines) if line.strip()
            }

            # 本页被覆盖的行号（来自 chunks 的 segments）
            covered: set[int] = set()
            for chunk in batch_chunks:
                for seg in chunk.get("_segments", []):
                    if seg.get("page") == pn:
                        slr = seg.get("line_range", [])
                        if len(slr) == 2:
                            covered.update(range(slr[0], slr[1] + 1))

            page_missing = sorted(non_empty - covered)
            if page_missing:
                missing[pn] = page_missing

        return missing

    # ── 第一级：整批驳回重做 ──────────────────────────────

    def _retry_batch_tier1(
        self,
        batch_pages: List[int],
        initial_missing: Dict[int, List[int]],
        batch_chunks: List[Dict],
    ) -> List[Dict]:
        """整批驳回重做：将漏行信息反馈给 LLM，重新分块。

        内部循环最多 coverage_max_batch_retries 次。
        耗尽后转第二级专项修补。
        """
        max_retries = self.config.coverage_max_batch_retries
        current_chunks = batch_chunks
        current_missing = initial_missing

        for attempt in range(1, max_retries + 1):
            # 构建反馈信息
            feedback = self._format_missing_feedback(current_missing)
            logger.info(
                "  [第一级] 第 %d/%d 次驳回重做，漏行: %s",
                attempt, max_retries,
                {p: ls for p, ls in current_missing.items()},
            )

            # 格式化输入，附加漏行反馈
            user_content = self._format_batch_for_llm(batch_pages)
            user_content = feedback + "\n\n" + user_content

            # 重新调用 LLM
            self._fetch_count = 0
            chunk_defs = self._call_llm_with_tools(user_content, batch_pages)
            current_chunks = self._assemble_chunks(chunk_defs)

            # 再校验
            current_missing = self._find_missing_lines(batch_pages, current_chunks)
            if not current_missing:
                logger.info("  [第一级] 第 %d 次重做后覆盖完整", attempt)
                return current_chunks

        # 第一级耗尽 → 转第二级
        total_remaining = sum(len(v) for v in current_missing.values())
        logger.warning(
            "  [第一级] %d 次重试耗尽，仍有 %d 个漏行 → 转第二级专项修补",
            max_retries, total_remaining,
        )
        return self._fix_missing_lines_tier2(
            current_missing, current_chunks
        )

    @staticmethod
    def _format_missing_feedback(missing: Dict[int, List[int]]) -> str:
        """将漏行信息格式化为 LLM 可读的反馈文本"""
        lines = [
            "## ⚠️ 上一轮分块遗漏了以下非空行，请在本轮中务必覆盖：",
        ]
        for pn in sorted(missing.keys()):
            line_nums = missing[pn]
            ranges = []
            # 将连续行号压缩为区间展示
            start = line_nums[0]
            end = line_nums[0]
            for ln in line_nums[1:]:
                if ln == end + 1:
                    end = ln
                else:
                    ranges.append((start, end))
                    start = end = ln
            ranges.append((start, end))
            range_str = ", ".join(
                f"L{s}" if s == e else f"L{s}-L{e}" for s, e in ranges
            )
            lines.append(f"- 第{pn}页: {range_str}")
        return "\n".join(lines)

    # ── 第二级：专项修补 ──────────────────────────────────

    def _fix_missing_lines_tier2(
        self,
        missing: Dict[int, List[int]],
        batch_chunks: List[Dict],
    ) -> List[Dict]:
        """专项修补：对每个漏行，收集其前后上下文，批量询问 LLM 归属判定。

        返回修改后的 chunk 列表（可能包含新增 chunk）。
        """
        # 为每个漏行构建上下文
        orphan_items: List[Dict] = []
        for pn in sorted(missing.keys()):
            for ln in missing[pn]:
                item = self._build_orphan_context(pn, ln, batch_chunks)
                orphan_items.append(item)

        if not orphan_items:
            return batch_chunks

        logger.info(
            "  [第二级] 对 %d 个漏行进行专项修补",
            len(orphan_items),
        )

        # 调用 LLM 判定归属
        fix_prompt = self._format_tier2_prompt(orphan_items)
        fix_result = self._call_tier2_llm(fix_prompt)

        if not fix_result:
            # LLM 失败或返回空 → 最后防线：所有漏行独立成块
            logger.warning(
                "  [第二级] LLM 未返回有效修复方案，%d 个漏行全部降级为 standalone",
                len(orphan_items),
            )
            return self._apply_standalone_fallback(batch_chunks, orphan_items)

        # 应用修复
        return self._apply_tier2_fixes(fix_result, batch_chunks, orphan_items)

    def _build_orphan_context(
        self, page_num: int, line_num: int, batch_chunks: List[Dict]
    ) -> Dict:
        """为单个漏行构建上下文：漏行内容 + 前后 chunk 文本"""
        page_text = self._pages[page_num]
        page_lines = page_text.split("\n")
        orphan_text = page_lines[line_num - 1] if line_num <= len(page_lines) else ""

        # 找到本页该行前后的 chunk
        chunk_before = None
        chunk_after = None

        for chunk in batch_chunks:
            for seg in chunk.get("_segments", []):
                if seg.get("page") != page_num:
                    continue
                slr = seg.get("line_range", [])
                if len(slr) != 2:
                    continue
                seg_start, seg_end = slr[0], slr[1]

                if seg_end < line_num:
                    if chunk_before is None or seg_end > chunk_before[1]:
                        chunk_before = (chunk, seg_end)
                elif seg_start > line_num:
                    if chunk_after is None or seg_start < chunk_after[1]:
                        chunk_after = (chunk, seg_start)

        return {
            "page": page_num,
            "line": line_num,
            "text": orphan_text,
            "context_before": chunk_before[0]["text"][-300:] if chunk_before else "(无)",
            "context_after": chunk_after[0]["text"][:300] if chunk_after else "(无)",
        }

    def _format_tier2_prompt(self, orphan_items: List[Dict]) -> str:
        """构建第二级 LLM 请求的 prompt"""
        parts = [
            "你是一位金融文档编辑。以下行在分块时被遗漏，请判断每行应如何归属。",
            "",
            "## 已有 chunk 上下文",
            "（每个漏行给出了前后相邻 chunk 的文本片段，供判断语义归属）",
            "",
        ]

        for i, item in enumerate(orphan_items):
            parts.append(f"### 漏行 {i + 1}")
            parts.append(f"- 位置：第{item['page']}页 L{item['line']}")
            parts.append(f"- 内容：`{item['text']}`")
            parts.append(f"- 上文（前一 chunk 的末尾）：{item['context_before']}")
            parts.append(f"- 下文（后一 chunk 的开头）：{item['context_after']}")
            parts.append("")

        parts.extend([
            "## 输出要求",
            "返回一个 JSON 对象：",
            "{",
            '  "fixes": [',
            '    {"line_index": 0, "action": "merge_before", "reason": "..."},',
            '    {"line_index": 1, "action": "standalone", "reason": "..."},',
            "  ]",
            "}",
            "",
            "action 说明：",
            '- "merge_before" — 归属到前一个 chunk',
            '- "merge_after"  — 归属到后一个 chunk',
            '- "standalone"  — 独立成块（纯文本类型）',
            "line_index 对应上面漏行编号（从 0 开始）。",
            "如某行前后均无 chunk，则必然为 standalone。",
        ])

        return "\n".join(parts)

    def _call_tier2_llm(self, prompt: str) -> List[Dict]:
        """调用 LLM 进行漏行归属判定，返回 fixes 列表"""
        messages = [
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._llm._call_with_retry(
                model=self.config.chunking_model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error("第二级 LLM 调用失败: %s", e)
            return []

        raw = response.choices[0].message.content or ""
        usage = response.usage
        write_debug_data(
            self._debug_logger,
            f"第二级 LLM 响应："
            f"{'in=' + str(usage.prompt_tokens) + ' out=' + str(usage.completion_tokens) if usage else '无 token 数据'}，"
            f"raw 长度 {len(raw)} 字符",
        )
        # 解析 JSON
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
            return data.get("fixes", [])
        except json.JSONDecodeError:
            logger.warning("第二级 LLM 返回非 JSON，回退到 standalone: %s", raw[:200])
            return []

    def _build_standalone_chunk(self, page: int, line: int) -> Dict:
        """为单个漏行创建独立的纯文本 chunk"""
        page_text = self._pages[page]
        page_lines = page_text.split("\n")
        orphan_text = page_lines[line - 1] if line <= len(page_lines) else ""
        return {
            "text": orphan_text,
            "type": "text",
            "doc_id": self._doc_id,
            "doc_name": self._doc_name,
            "page_nums": [page],
            "length": len(orphan_text),
            "has_financial_keywords": [
                kw for kw in FINANCIAL_KEYWORDS if kw in orphan_text
            ],
            "_context_summary": None,
            "_header_source": None,
            "_segments": [{"page": page, "line_range": [line, line]}],
        }

    def _apply_standalone_fallback(
        self, batch_chunks: List[Dict], orphan_items: List[Dict]
    ) -> List[Dict]:
        """最后防线：将所有漏行都创建为独立的 standalone chunk"""
        import copy
        result = copy.deepcopy(batch_chunks)
        for item in orphan_items:
            result.append(self._build_standalone_chunk(item["page"], item["line"]))
        return result

    def _apply_tier2_fixes(
        self,
        fixes: List[Dict],
        batch_chunks: List[Dict],
        orphan_items: List[Dict],
    ) -> List[Dict]:
        """将 LLM 的归属判定应用到 chunk 列表中。

        - merge_before/merge_after：修改对应 chunk 的 segments
        - standalone：创建新 chunk 并追加到列表
        """
        import copy
        result = copy.deepcopy(batch_chunks)

        for fix in fixes:
            idx = fix.get("line_index", -1)
            if idx < 0 or idx >= len(orphan_items):
                continue

            item = orphan_items[idx]
            action = fix.get("action", "standalone")
            reason = fix.get("reason", "")

            page = item["page"]
            line = item["line"]

            if action == "standalone":
                result.append(self._build_standalone_chunk(page, line))
                logger.debug("  [第二级] L%d → standalone（%s）", line, reason)

            elif action in ("merge_before", "merge_after"):
                target_chunk = self._find_neighbor_chunk(
                    result, page, line, direction=action
                )
                if target_chunk:
                    self._extend_segment(target_chunk, page, line)
                    new_text = self._reassemble_chunk_text(target_chunk)
                    target_chunk["text"] = new_text
                    target_chunk["length"] = len(new_text)
                    if page not in target_chunk["page_nums"]:
                        target_chunk["page_nums"].append(page)
                    logger.debug("  [第二级] L%d → %s（%s）", line, action, reason)
                else:
                    logger.debug(
                        "  [第二级] L%d → 找不到邻居 chunk，降级为 standalone", line
                    )
                    result.append(self._build_standalone_chunk(page, line))

        return result

    def _find_neighbor_chunk(
        self, chunks: List[Dict], page: int, line: int, direction: str
    ) -> Optional[Dict]:
        """找到指定行前面或后面最近的 chunk。

        direction = "merge_before" → 找该行前面最近的 chunk
        direction = "merge_after"  → 找该行后面最近的 chunk
        """
        best_chunk = None
        best_dist = float("inf")

        for chunk in chunks:
            for seg in chunk.get("_segments", []):
                if seg.get("page") != page:
                    continue
                slr = seg.get("line_range", [])
                if len(slr) != 2:
                    continue
                seg_start, seg_end = slr[0], slr[1]

                if direction == "merge_before" and seg_end < line:
                    dist = line - seg_end
                    if dist < best_dist:
                        best_dist = dist
                        best_chunk = chunk
                elif direction == "merge_after" and seg_start > line:
                    dist = seg_start - line
                    if dist < best_dist:
                        best_dist = dist
                        best_chunk = chunk

        return best_chunk

    @staticmethod
    def _extend_segment(chunk: Dict, page: int, line: int) -> None:
        """扩展 chunk 中指定页的 segment 以包含指定行"""
        segments = chunk.get("_segments", [])
        for seg in segments:
            if seg.get("page") != page:
                continue
            slr = seg.get("line_range", [])
            if len(slr) != 2:
                continue
            # 扩展最近的那个 segment
            if slr[1] == line - 1:
                # 行紧接在 segment 之后，扩展 end
                slr[1] = line
                return
            if slr[0] == line + 1:
                # 行紧接在 segment 之前，扩展 start
                slr[0] = line
                return

        # 没找到可扩展的 segment，新增一个并保持 page+line 排序
        segments.append({"page": page, "line_range": [line, line]})
        segments.sort(key=lambda s: (s.get("page", 0), s.get("line_range", [0, 0])[0]))

    def _reassemble_chunk_text(self, chunk: Dict) -> str:
        """根据 chunk 的 segments 重新组装嵌入文本"""
        segments = chunk.get("_segments", [])
        data_parts = []
        for seg in segments:
            sp = seg.get("page", 0)
            slr = seg.get("line_range", [])
            if len(slr) != 2:
                continue
            page_text = self._pages[sp] if 0 < sp < len(self._pages) else ""
            page_lines = page_text.split("\n")
            start, end = max(1, slr[0]), min(len(page_lines), slr[1])
            if start > end:
                continue
            data_parts.append("\n".join(page_lines[start - 1 : end]))

        data_text = "\n".join(data_parts)
        embed_parts = []
        cs = chunk.get("_context_summary")
        if cs:
            embed_parts.append(f"[概述] {cs}")
        hs = chunk.get("_header_source")
        if hs:
            ht = self._extract_header_text(hs)
            if ht:
                embed_parts.append(f"[表头]\n{ht}")
        embed_parts.append(data_text)
        return "\n".join(embed_parts)

    # ── 页面格式化 ────────────────────────────────────────

    @staticmethod
    def _detect_page_type(page_text: str) -> str:
        """检测页面是否包含表格。

        阈值 ≥2：单行 | 可能是分隔线或噪声，连续两行以上才判定为表格区域。
        空页返回 "空白页"。
        """
        if not page_text.strip():
            return "空白页"
        lines = page_text.split("\n")
        pipe_lines = [l for l in lines if l.strip().startswith("|")]
        return "含表格" if len(pipe_lines) >= 2 else "纯文本"

    def _format_page_for_llm(self, page_num: int) -> str:
        """将单页格式化为 LLM 输入，带行号前缀和类型标注。

        空白页仅输出一行标注，不占用 token 也保留隔断信息。
        """
        page_text = self._pages[page_num]
        page_type = self._detect_page_type(page_text)

        header = f"[第{page_num}页，共{self._total_pages}页]  [{page_type}]"
        if page_type == "空白页":
            return f"{header}\n（本页为空）"

        lines = page_text.split("\n")
        body = "\n".join(f"L{i + 1}: {line}" for i, line in enumerate(lines))
        return f"{header}\n{body}"

    def _format_batch_for_llm(self, page_nums: List[int]) -> str:
        """格式化一批页面"""
        parts = [self._format_page_for_llm(pn) for pn in page_nums]

        # 如果有章节地图，拼在最前面
        if self._chapter_map:
            map_text = self._chapter_map_to_text(page_nums)
            if map_text:
                parts.insert(0, map_text)

        return "\n\n".join(parts)

    # ── 章节地图 ──────────────────────────────────────────

    def _build_chapter_map(self, non_empty_pages: List[int]) -> Optional[List[Dict]]:
        """生成章节地图（当前仅 rule_only 模式，后续扩展 LLM 模式）"""
        mode = self.config.chapter_map_mode
        if mode == "off":
            return None

        entries: List[Dict] = []
        current_start: Optional[int] = None
        current_title: Optional[str] = None

        for pn in non_empty_pages:
            text = self._pages[pn]
            # 规则：抓每页前 200 字符中的章节标题
            head = text[:200]
            m = re.search(r"#{1,3}\s+(.+)", head)
            if m:
                title = m.group(1).strip()
                if current_start is not None:
                    entries.append({
                        "start": current_start,
                        "end": pn - 1,
                        "title": current_title or "未命名",
                    })
                current_start = pn
                current_title = title

        # 最后一段
        if current_start is not None:
            entries.append({
                "start": current_start,
                "end": non_empty_pages[-1],
                "title": current_title or "未命名",
            })

        if entries:
            logger.info("章节地图（rule_only）：%d 个章节", len(entries))
        return entries or None

    def _chapter_map_to_text(self, page_nums: List[int]) -> str:
        """将章节地图转为 LLM 可读文本（仅包含与当前批次相关的章节）"""
        if not self._chapter_map:
            return ""

        batch_start = page_nums[0]
        batch_end = page_nums[-1]
        relevant = [
            e for e in self._chapter_map
            if e["start"] <= batch_end and e["end"] >= batch_start
        ]

        if not relevant:
            return ""

        lines = [
            "## 章节地图（供参考，了解当前内容所属章节）",
            "| 起始页 | 结束页 | 章节标题 |",
            "|--------|--------|----------|",
        ]
        for e in relevant:
            lines.append(f"| {e['start']} | {e['end']} | {e['title']} |")

        return "\n".join(lines)

    def _call_llm_with_tools(
        self, user_content: str, page_nums: List[int]
    ) -> List[Dict]:
        """调用 DeepSeek LLM，处理 tool calling 循环，返回 chunk 定义列表。

        流程：
        1. LLM 可调用 fetch_page 跨页查表头
        2. LLM 必须调用 submit_chunks 提交最终结果
        3. 如 LLM 未调 submit_chunks 直接返回文本，解析为兜底
        4. fetch_page 用尽后强制要求 submit_chunks
        """
        tools = [FETCH_PAGE_TOOL, SUBMIT_CHUNKS_TOOL]
        try:
            system_content = SYSTEM_PROMPT.format(
                chunk_soft_max=self.config.chunk_soft_max,
            )
        except Exception:
            logger.exception("LLM system prompt 格式化失败")
            raise

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        write_debug_data(
            self._debug_logger,
            f"开始新批次：pages={page_nums}，"
            f"user_content 长度 {len(user_content)} 字符",
        )

        round_num = 0
        while self._fetch_count < self.config.fetch_page_limit:
            round_num += 1
            response = self._llm._call_with_retry(
                model=self.config.chunking_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=8192,
            )

            msg = response.choices[0].message

            # 记录 token 消耗
            usage = response.usage
            if usage:
                logger.info(
                    "  LLM token: in=%d out=%d total=%d",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                )
                write_debug_data(
                    self._debug_logger,
                    f"LLM 第 {round_num} 轮交互："
                    f"in={usage.prompt_tokens} out={usage.completion_tokens} "
                    f"total={usage.total_tokens}",
                )

            # 无 tool call → LLM 直接返回了文本，兜底解析
            if not msg.tool_calls:
                content = msg.content or ""
                logger.info(
                    "  LLM 未调用 tool，回退到文本解析"
                    "（finish_reason=%s，content 长度 %d）",
                    getattr(response.choices[0], "finish_reason", "?"),
                    len(content),
                )
                write_debug_data(
                    self._debug_logger,
                    f"LLM 未调用 tool，走文本解析兜底。"
                    f"响应内容（前500字符）：{content[:500]}",
                )
                try:
                    return self._parse_llm_response(content, page_nums)
                except Exception:
                    logger.exception(
                        "文本解析兜底失败，响应内容: %s", content[:500],
                    )
                    raise

            # 处理 tool calls
            messages.append(msg)  # assistant message with tool_calls
            submitted_chunks: Optional[List[Dict]] = None

            for tc in msg.tool_calls:
                if tc.function.name == "fetch_page":
                    try:
                        args = json.loads(tc.function.arguments)
                        page_num = int(args.get("page", 0))
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning("fetch_page 参数解析失败: %s", e)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"错误：无法解析 fetch_page 参数：{e}",
                        })
                        self._fetch_count += 1
                        continue
                    result = self._do_fetch_page(page_num)
                    write_debug_data(
                        self._debug_logger,
                        f"LLM 调用 fetch_page(第{page_num}页)，"
                        f"第 {self._fetch_count + 1} 次，返回 {len(result)} 字符",
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    self._fetch_count += 1

                elif tc.function.name == "submit_chunks":
                    try:
                        args = json.loads(tc.function.arguments)
                        submitted_chunks = args.get("chunks", [])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("submit_chunks 参数解析失败: %s", e)
                        submitted_chunks = []
                    # 摘要每个 chunk 的关键字段
                    chunk_summary = []
                    for c in submitted_chunks:
                        segs = c.get("segments", [])
                        pages = sorted({s.get("page", 0) for s in segs})
                        chunk_summary.append(
                            f"{c.get('chunk_type', '?')}/"
                            f"pages={pages}/"
                            f"{len(segs)}seg"
                        )
                    write_debug_data(
                        self._debug_logger,
                        f"LLM 调用 submit_chunks：提交 {len(submitted_chunks)} 个 chunk — "
                        + " | ".join(chunk_summary)
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "已收到切块结果。",
                    })
                    break  # submit_chunks 是终点，跳过本轮其余 tool call

                else:
                    logger.warning("未知 tool call: %s", tc.function.name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"错误：未知工具 {tc.function.name}",
                    })

            # submit_chunks 被调用 → 直接返回结果，不再循环
            if submitted_chunks is not None:
                return submitted_chunks

        logger.warning(
            "fetch_page 达到上限 %d 次，强制要求 LLM 调用 submit_chunks",
            self.config.fetch_page_limit,
        )
        # 最后尝试：强制调用 submit_chunks
        messages.append({
            "role": "user",
            "content": "fetch_page 调用次数已达上限，请立即调用 submit_chunks 提交当前已有的分析结果。",
        })
        response = self._llm._call_with_retry(
            model=self.config.chunking_model,
            messages=messages,
            tools=[SUBMIT_CHUNKS_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_chunks"}},
            temperature=0.1,
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "submit_chunks":
                    try:
                        args = json.loads(tc.function.arguments)
                        return args.get("chunks", [])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error("强制 submit_chunks 参数解析失败: %s", e)

        content_preview = (
            (msg.content or "")[:300] if hasattr(msg, "content") else "N/A"
        )
        logger.error(
            "LLM 最终仍未能调用 submit_chunks（content: %s）",
            content_preview,
        )
        raise RuntimeError("LLM 在 fetch_page 耗尽后未能提交切块结果")

    # ── fetch_page 服务端实现 ──────────────────────────────

    def _do_fetch_page(self, page_num: int) -> str:
        """
        执行 fetch_page tool：返回指定页的格式化内容。

        Args:
            page_num: 页码（1-based）

        Returns:
            带行号前缀的页面内容，或错误提示
        """
        if page_num < 1 or page_num >= len(self._pages):
            return f"[错误] 页码 {page_num} 超出范围（共 {self._total_pages} 页）"

        content = self._pages[page_num]
        if not content.strip():
            return f"[第{page_num}页]（本页为空）"

        # 以相同格式返回，LLM 可直接引用行号
        header = f"[第{page_num}页，仅供参考，请勿纳入分块]"
        lines = content.split("\n")
        body = "\n".join(f"L{i + 1}: {line}" for i, line in enumerate(lines))
        logger.debug("  fetch_page(%d) → %d 行", page_num, len(lines))
        return f"{header}\n{body}"

    # ── 响应解析 ──────────────────────────────────────────

    def _parse_llm_response(
        self, raw: str, page_nums: List[int]
    ) -> List[Dict]:
        """
        解析 LLM 的 JSON 响应。

        支持多页批处理：每页的结果合并为统一的 chunk 定义列表。
        每个 chunk 定义包含 chunk_type、context_summary、
        header_source、segments。
        """
        raw = raw.strip()
        # LLM 常将 JSON 输出包裹在 ```json ... ``` 代码块中，先剥离再解析
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        # 三级回退解析策略（仅兜底路径使用，正常路径走 submit_chunks tool）：
        #   1. 单对象 {"chunks": [...]}
        #   2. 多页数组 [{"chunks": [...]}, ...]
        #   3. 括号计数提取（LLM 可能在 JSON 前后附加了说明文字）
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "chunks" in data:
                return self._normalize_response(data)
        except json.JSONDecodeError:
            pass

        # LLM 可能返回多页的数组 [{"chunks": [...]}, ...]
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                all_chunks = []
                for item in data:
                    if isinstance(item, dict) and "chunks" in item:
                        all_chunks.extend(self._normalize_response(item))
                if all_chunks:
                    return all_chunks
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 片段（用括号计数处理嵌套对象和数组）
        extracted = self._extract_json_object(raw, key="chunks")
        if extracted:
            try:
                data = json.loads(extracted)
                if isinstance(data, dict) and "chunks" in data:
                    return self._normalize_response(data)
                if isinstance(data, list):
                    all_chunks = []
                    for item in data:
                        if isinstance(item, dict) and "chunks" in item:
                            all_chunks.extend(self._normalize_response(item))
                    if all_chunks:
                        return all_chunks
            except json.JSONDecodeError:
                pass

        logger.error("无法解析 LLM 响应为 JSON：%s", raw[:500])
        raise ValueError(f"LLM 返回了无法解析的响应：{raw[:300]}...")

    @staticmethod
    def _extract_json_object(text: str, key: str) -> Optional[str]:
        """从文本中提取包含指定 key 的最外层 JSON 对象或数组。

        使用括号计数，正确处理嵌套的 {} 和 []。
        """
        # 找到 key 的位置
        idx = text.find(f'"{key}"')
        if idx == -1:
            return None

        # 从 key 往前找最近的 { 或 [
        start = idx
        brace_depth = 0
        bracket_depth = 0
        found_start = False
        while start >= 0:
            ch = text[start]
            if ch == '}' or ch == ']':
                if ch == '}':
                    brace_depth += 1
                else:
                    bracket_depth += 1
            elif ch == '{':
                if brace_depth == 0 and bracket_depth == 0:
                    found_start = True
                    break
                brace_depth -= 1
            elif ch == '[':
                if brace_depth == 0 and bracket_depth == 0:
                    found_start = True
                    break
                bracket_depth -= 1
            start -= 1

        if not found_start:
            return None

        # 从 start 往后找对应的闭合括号
        end = start
        brace_depth = 0
        bracket_depth = 0
        is_object = text[start] == '{'
        while end < len(text):
            ch = text[end]
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if is_object and brace_depth == 0 and bracket_depth == 0:
                    return text[start : end + 1]
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth -= 1
                if not is_object and brace_depth == 0 and bracket_depth == 0:
                    return text[start : end + 1]
            end += 1

        return None

    def _normalize_response(self, data: Dict) -> List[Dict]:
        """提取 chunk 定义列表"""
        return data.get("chunks", [])

    # ── Chunk 组装 ────────────────────────────────────────

    def _assemble_chunks(self, chunk_defs: List[Dict]) -> List[Dict]:
        """根据 LLM 返回的定义，提取原文内容，组装最终 chunk"""
        result = []
        for cdef in chunk_defs:
            try:
                assembled = self._assemble_one_chunk(cdef)
                if assembled:
                    result.append(assembled)
            except Exception as e:
                logger.warning("组装 chunk 失败: %s, 定义=%s", e, cdef)
        return result

    def _split_oversized_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """对超过 chunk_hard_max 的 chunk 强制按规则切分。

        - 表格 chunk：在数据行边界切，每段带相同表头 + context_summary
        - 文本 chunk：在段落边界（空行）切
        """
        hard_max = self.config.chunk_hard_max
        result = []
        for c in chunks:
            if c["length"] <= hard_max:
                result.append(c)
                continue

            logger.info(
                "  硬切超限 chunk：type=%s, %d → 拆分", c["type"], c["length"],
            )
            sub = self._do_split_chunk(c, hard_max)
            result.extend(sub)

        return result

    def _do_split_chunk(self, chunk: Dict, max_size: int) -> List[Dict]:
        """切分单个超限 chunk，返回若干子 chunk"""
        ctype = chunk["type"]
        if ctype == "table":
            return self._split_table_chunk(chunk, max_size)
        else:
            return self._split_text_chunk(chunk, max_size)

    def _split_table_chunk(self, chunk: Dict, max_size: int) -> List[Dict]:
        """在表格数据行边界切分，每段带表头 + 概述"""
        text = chunk["text"]
        lines = text.split("\n")

        # 分离前缀（[概述]、[表头]）和表格数据行
        prefix_lines = []
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("[概述]") or line.startswith("[表头]"):
                prefix_lines.append(line)
                data_start = i + 1
            elif line.strip().startswith("|"):
                break  # 第一个表格行开始
            else:
                data_start = i + 1

        prefix = "\n".join(prefix_lines) + "\n" if prefix_lines else ""
        data_lines = lines[data_start:]
        prefix_len = len(prefix)

        sub_chunks = []
        current = []
        current_len = prefix_len

        for line in data_lines:
            line_len = len(line) + 1  # +1 for \n
            if current_len + line_len > max_size and current:
                sub_text = prefix + "\n".join(current)
                sub_chunks.append(self._make_sub_chunk(chunk, sub_text))
                current = [line]
                current_len = prefix_len + line_len
            else:
                current.append(line)
                current_len += line_len

        if current:
            sub_text = prefix + "\n".join(current)
            sub_chunks.append(self._make_sub_chunk(chunk, sub_text))

        return sub_chunks or [chunk]

    def _split_text_chunk(self, chunk: Dict, max_size: int) -> List[Dict]:
        """在段落边界切分文本，每段带概述"""
        text = chunk["text"]
        paras = text.split("\n\n")

        # 第一个段落可能包含 [概述] 前缀
        prefix = ""
        if paras and paras[0].startswith("[概述]"):
            prefix = paras[0] + "\n\n"
            paras = paras[1:]

        sub_chunks = []
        current = []
        current_len = len(prefix)

        for para in paras:
            para_len = len(para) + 2  # +2 for \n\n
            if current_len + para_len > max_size and current:
                sub_text = prefix + "\n\n".join(current)
                sub_chunks.append(self._make_sub_chunk(chunk, sub_text))
                current = [para]
                current_len = len(prefix) + para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            sub_text = prefix + "\n\n".join(current)
            sub_chunks.append(self._make_sub_chunk(chunk, sub_text))

        return sub_chunks or [chunk]

    @staticmethod
    def _make_sub_chunk(original: Dict, text: str) -> Dict:
        """用切分后的文本创建子 chunk，继承原始元数据"""
        return {
            "text": text,
            "type": original["type"],
            "doc_id": original["doc_id"],
            "doc_name": original.get("doc_name", ""),
            "page_nums": list(original.get("page_nums", [])),
            "length": len(text),
            "has_financial_keywords": list(original.get("has_financial_keywords", [])),
            "_context_summary": original.get("_context_summary"),
            "_header_source": original.get("_header_source"),
            "_segments": list(original.get("_segments", [])),
        }

    def _assemble_one_chunk(self, cdef: Dict) -> Optional[Dict]:
        """组装单个 chunk，遍历 segments 逐页提取内容并拼接"""
        chunk_type = cdef.get("chunk_type", "mixed")
        context_summary = cdef.get("context_summary")
        header_source = cdef.get("header_source")

        # ── 解析 segments ──
        segments = cdef.get("segments")
        if not segments:
            logger.warning("chunk 缺少有效的 segments: %s", cdef)
            return None

        # ── 逐 segment 提取文本 ──
        data_parts: List[str] = []
        page_nums: List[int] = []

        for seg in segments:
            sp = seg.get("page", 0)
            slr = seg.get("line_range", [])
            if not slr or len(slr) != 2:
                logger.warning("segment 缺少有效 line_range: %s", seg)
                continue

            start_line, end_line = slr[0], slr[1]

            # 提取对应页正文
            page_text = self._pages[sp] if 0 < sp < len(self._pages) else ""
            page_lines = page_text.split("\n")
            max_lines = len(page_lines)

            # 校验行号范围
            if start_line < 1 or end_line > max_lines or start_line > end_line:
                logger.warning(
                    "segment line_range 越界：page=%d, range=[%d, %d], 实际行数=%d",
                    sp, start_line, end_line, max_lines,
                )
                start_line = max(1, start_line)
                end_line = min(max_lines, end_line)
                if start_line > end_line:
                    continue

            # line_range 是 1-based 闭区间
            selected = page_lines[start_line - 1 : end_line]
            data_parts.append("\n".join(selected))
            if sp not in page_nums:
                page_nums.append(sp)

        if not data_parts:
            return None

        data_text = "\n".join(data_parts)

        # 跳过空内容 chunk
        if not data_text.strip():
            logger.debug("跳过空 chunk：segments=%s", segments)
            return None

        # 提取表头文本
        header_text = self._extract_header_text(header_source)

        # 拼接 chunk 嵌入文本
        embed_parts = []
        if context_summary:
            embed_parts.append(f"[概述] {context_summary}")
        if header_text:
            embed_parts.append(f"[表头]\n{header_text}")
        embed_parts.append(data_text)
        embed_text = "\n".join(embed_parts)

        assembled = {
            "text": embed_text,
            "type": chunk_type,
            "doc_id": self._doc_id,
            "page_nums": page_nums,
            "length": len(embed_text),
            "has_financial_keywords": [kw for kw in FINANCIAL_KEYWORDS if kw in data_text],
            "_context_summary": context_summary,
            "_header_source": header_source,
            "_segments": segments,
        }
        seg_detail = ", ".join(
            f"p{s['page']} L{s['line_range'][0]}-L{s['line_range'][1]}"
            for s in segments
        )
        write_debug_data(
            self._debug_logger,
            f"chunk 组装完成：type={chunk_type}, {len(segments)} segment(s), "
            f"{len(embed_text)} 字符, pages={page_nums}, segments: {seg_detail}"
        )
        return assembled

    def _extract_header_text(self, header_source: Optional[Dict]) -> str:
        """根据 header_source 提取表头文本"""
        if not header_source:
            return ""

        hp = header_source.get("page", 0)
        hlines = header_source.get("lines", [])
        if not hlines or hp < 1 or hp >= len(self._pages):
            return ""

        page_text = self._pages[hp]
        page_lines = page_text.split("\n")
        selected = []
        for ln in hlines:
            if 1 <= ln <= len(page_lines):
                selected.append(page_lines[ln - 1])

        return "\n".join(selected)
