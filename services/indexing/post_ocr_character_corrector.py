"""
OCR后处理：词级字符修正。

公开接口：
- correct_word(word)      → 单词语义修正（罗马数字、固定映射、词典校对）
- correct_text(text)      → 全文修正（逐词修正 + 实体名模糊匹配）
- correct_ocr_result(text) → correct_text 的向后兼容别名

支持：罗马数字纠正、OCR 固定映射、金融词典校对、实体名字典匹配。
"""

from __future__ import annotations

import re
import string
from typing import Dict, List, Tuple

from utils import get_file_logger
logger = get_file_logger(__file__)


class PostOcrCharacterCorrector:
    """OCR后处理：词级智能修正。"""

    # ── 罗马数字 OCR 常见错误（词级精确匹配，≥2 字符，避免单字符误杀） ──
    _ROMAN_FIXES: dict[str, str] = {
        "IIl": "III", "Ill": "III", "llI": "III", "lll": "III",
        "lII": "III", "Il": "II", "ll": "II", "Iv": "IV", "lV": "IV",
        "Vl": "VI", "Vlll": "VIII",
    }

    # ── 正确的罗马数字模式（用于拆分拼接词，如 PartnersIV → Partners IV） ──
    _ROMAN_PATTERNS: tuple[str, ...] = (
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    )
    _ROMAN_CHARS: frozenset[str] = frozenset("IVX")

    # ── OCR 固定映射（I→l, rn→m 等无法用编辑距离自动修复的错字） ──
    _WORD_FIXES: dict[str, str] = {
        "lnvestment": "Investment",
        "lnvestments": "Investments",
        "lnc": "Inc",
        "lndustrial": "Industrial",
        "lndia": "India",
        "lntegrated": "Integrated",
        "lnsurance": "Insurance",
        "lndependent": "Independent",
        "lntroduction": "Introduction",
    }

    # ── 金融英文词典（等长且编辑距离=1 时自动纠正） ──
    _FINANCIAL_DICT: set[str] = {
        "investment", "investments", "capital", "limited", "holdings",
        "partners", "fund", "funds", "growth", "resources", "private",
        "equity", "securities", "management", "international",
        "corporation", "company", "group", "financial", "assets",
        "liabilities", "income", "revenue", "expense", "balance",
        "statement", "consolidated", "annual", "report",
        "depreciation", "amortization", "goodwill", "intangible",
        "aviation", "infrastructure", "logistics", "finance",
        "asia", "china", "global", "development", "opportunity",
        "special", "debt", "secured", "lending", "share", "stock",
        "bond", "note", "loan", "deposit", "interest", "principal",
        "maturity", "portfolio", "asset", "enterprise",
        "industrial", "commercial", "property", "technology",
        "telecommunications", "energy", "consumer", "healthcare",
        "insurance", "banking", "manufacturing",
        "transportation", "shipping", "aerospace",
        "shareholders", "dividend", "operating", "current",
        "noncurrent", "deferred", "profit", "total",
        "structured", "secured", "unsecured", "subordinated",
        "convertible", "redeemable", "perpetual", "cumulative",
        # 合法专有名词 —— 防止被编辑距离 1 误杀
        "aria", "sunrise", "citron", "holisol", "kingvest",
        "bright", "pine", "sino", "ocean",
    }

    # ── OCR 常见字符混淆对（仅允许这些替换，避免误杀） ──
    _ALLOWED_SUBSTITUTIONS: set[frozenset[str]] = frozenset({
        frozenset({"v", "y"}),       # Investment ↔ Inyestment
        frozenset({"rn", "m"}),      # Modern ↔ Modem
        frozenset({"I", "l"}),       # III ↔ lII
        frozenset({"c", "e"}),       # Finance ↔ Financx
        frozenset({"i", "l"}),       # Industrial ↔ lndustrial (covered by _WORD_FIXES)
        frozenset({"h", "b"}),       # Shareholder ↔ Bareholder
        frozenset({"r", "t"}),       # Portfolio ↔ Porttolio
    })

    # ── 金融实体名字典（规范形式，随实际数据持续扩充） ──
    # 每个条目存储原始大小写和标点格式，用于替换时的展示
    _ENTITY_DICT: set[str] = {
        "Sunrise Capital II, L.P.",
        "Sunrise Capital III, L.P.",
        "Sunrise Capital IV, L.P.",
    }

    # 归一化正则：匹配所有非字母数字非空白字符（标点）
    _ENTITY_NORMALIZE_RE = re.compile(r"[^\w\s]")

    # 首尾标点（剥离后复原用）
    _PUNCT = frozenset(string.punctuation)

    # ── 公开接口 ──

    @classmethod
    def correct_word(cls, word: str) -> str:
        """修正单个词的 OCR 错误。

        处理顺序：标点剥离 → 拼接词拆分 → 固定映射 → 罗马数字 → 词典校对 → 复原标点。

        Args:
            word: OCR 识别的一个词（可能带首尾标点，如 "Il,"）

        Returns:
            纠正后的词（无需纠正则原样返回）
        """
        if not word:
            return word

        # ── 剥离首尾标点 ──
        start = 0
        while start < len(word) and word[start] in cls._PUNCT:
            start += 1
        end = len(word)
        while end > start and word[end - 1] in cls._PUNCT:
            end -= 1

        prefix = word[:start]
        core = word[start:end]
        suffix = word[end:]

        if not core or not core[0].isalpha():
            return word

        # ── 0. 拼接词拆分（如 PartnersIll → Partners III） ──
        core = cls._split_concatenated(core)

        # ── 若含空格 → 逐词处理再拼接 ──
        if " " in core:
            sub_words = core.split(" ")
            corrected_parts = [cls._correct_single(w) for w in sub_words]
            return prefix + " ".join(corrected_parts) + suffix

        return prefix + cls._correct_single(core) + suffix

    @classmethod
    def _correct_single(cls, word: str) -> str:
        """处理不含空格的单个词（标点剥离 → 固定映射 → 罗马数字 → 词典校对）。"""
        if not word:
            return word

        # ── 剥离首尾标点 ──
        start = 0
        while start < len(word) and word[start] in cls._PUNCT:
            start += 1
        end = len(word)
        while end > start and word[end - 1] in cls._PUNCT:
            end -= 1

        prefix = word[:start]
        core = word[start:end]
        suffix = word[end:]

        if not core or not core[0].isalpha():
            return word

        # 1. 固定映射（lnvestment → Investment 等）
        fixed = cls._WORD_FIXES.get(core)
        if fixed is not None:
            return prefix + fixed + suffix

        # 2. 罗马数字 OCR 纠正（IIl → III 等）
        fixed = cls._ROMAN_FIXES.get(core)
        if fixed is not None:
            return prefix + fixed + suffix

        # 3. 词典校对：等长 + 编辑距离 1 且差异字符在允许列表内
        lower = core.lower()
        if lower in cls._FINANCIAL_DICT:
            return word

        for term in cls._FINANCIAL_DICT:
            if len(term) != len(lower):
                continue
            if cls._is_allowed_edit(term, lower):
                corrected = term[0].upper() + term[1:] if core[0].isupper() else term
                return prefix + corrected + suffix

        return word

    @classmethod
    def _split_concatenated(cls, core: str) -> str:
        """拆分被 PaddleOCR 错误拼接的词。

        如 PartnersIll → Partners III、PartnersIV → Partners IV。
        不拆分正确的罗马数字（如 IV 不会拆成 I V）。
        """
        all_patterns = list(cls._ROMAN_FIXES.items())
        all_patterns += [(p, p) for p in cls._ROMAN_PATTERNS]

        for err, fix in all_patterns:
            # 1. 后缀：前面是普通字母（小写，排除 FedEx→FedE X 等专有名词误拆）
            if core.endswith(err) and len(core) > len(err):
                base = core[:-len(err)].rstrip()
                if base and cls._can_split_before(base[-1], is_suffix=True):
                    return base + " " + fix
            # 2. 词中粘连：如 "HoldingsIVLimited" → 拆；"Holdings IV Limited" → 不拆
            idx = core.find(err)
            while idx > 0:
                before = core[:idx]
                after = core[idx + len(err):]
                before_char = before[-1] if before else ""
                after_char = after[0] if after else ""
                if cls._can_split_before(before_char, is_suffix=False) and (
                        not after_char or not after_char.isalpha()):
                    return before + " " + fix + after
                idx = core.find(err, idx + 1)
        return core

    @classmethod
    def _can_split_before(cls, char: str, *, is_suffix: bool) -> bool:
        """判断罗马数字拆分点是否合法。

        单字符罗马数字模式（I/V/X）风险高，要求前面是小写字母
        （如 CapitalI → 拆；FedEx → 不拆，E 是大写）。
        """
        if not char.isalpha():
            return False
        if char.upper() in cls._ROMAN_CHARS:
            return False
        if is_suffix and char.isupper():
            return False
        return True

    @classmethod
    def _is_allowed_edit(cls, term: str, word: str) -> bool:
        """检查两个等长词的差异是否在允许的 OCR 混淆字符对列表中。"""
        diffs = [(a, b) for a, b in zip(term, word) if a != b]
        if len(diffs) != 1:
            return False
        pair = frozenset(diffs[0])
        return any(pair == allowed for allowed in cls._ALLOWED_SUBSTITUTIONS)

    # ── 实体名模糊匹配 ──

    @staticmethod
    def _normalize_entity(text: str) -> str:
        """归一化实体名：去标点、合并空白、转小写。

        将 "Sunrise Capital III, L.P." 归一化为 "sunrise capital iii lp"，
        使得标点差异（如 OCR 误插的冒号）在匹配时被忽略。
        """
        text = PostOcrCharacterCorrector._ENTITY_NORMALIZE_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str, max_dist: int = 2) -> int:
        """计算两个字符串的编辑距离（Levenshtein），超过 max_dist 时早停。

        早停优化：一旦当前行的最小值 > max_dist，终止计算并返回 max_dist + 1。
        对于实体名匹配场景（max_dist=2），绝大多数窗口在前几行就被杀掉，
        计算量从 O(L²) 降到 O(L)。

        Args:
            s1: 较长字符串
            s2: 较短字符串
            max_dist: 距离上限，超过此值直接返回 max_dist + 1
        """
        if len(s1) < len(s2):
            return PostOcrCharacterCorrector._levenshtein_distance(s2, s1, max_dist)
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            row_min = i + 1  # 追踪当前行最小值
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                val = min(insertions, deletions, substitutions)
                curr_row.append(val)
                if val < row_min:
                    row_min = val
            # 早停：当前行最小值已超上限，后续只会更大
            if row_min > max_dist:
                return max_dist + 1
            prev_row = curr_row

        return prev_row[-1]

    @classmethod
    def _correct_entities(cls, text: str) -> Tuple[str, Dict]:
        """全文本实体名模糊匹配与修正。

        扫描全文，查找与 _ENTITY_DICT 中实体名匹配的文本片段：

        - 归一化后精确匹配：替换为字典中的规范形式
        - 归一化后编辑距离 = 1：自动修正（前提：原始 span 未吞入额外字母/数字）
        - 归一化后编辑距离 = 2：记录 WARNING 日志，不自动修正
        - 多个候选编辑距离均为 1：记录 WARNING 日志，不自动修正（歧义）

        替换从右向左进行以保证位置索引不被破坏。

        Args:
            text: OCR 识别后的全文

        Returns:
            (修正后的文本, {"corrections": int, "warnings": list})
        """
        if not text or not cls._ENTITY_DICT:
            return text, {"corrections": 0, "warnings": []}

        # ── 预处理：归一化全文本并建立字符位置映射 ──
        # norm_to_orig[i] = 原始文本中归一化后第 i 个字符对应的位置
        norm_chars: List[str] = []
        norm_to_orig: List[int] = []

        for orig_idx, ch in enumerate(text):
            if ch.isspace():
                if norm_chars and norm_chars[-1] != " ":
                    norm_chars.append(" ")
                    norm_to_orig.append(orig_idx)
            elif ch.isalnum():
                norm_chars.append(ch.lower())
                norm_to_orig.append(orig_idx)
            # 标点直接丢弃

        norm_text = "".join(norm_chars)

        # ── 收集所有匹配 span ──
        # 每个匹配: (orig_start, orig_end, canonical_entity, distance)
        matches: List[Tuple[int, int, str, int]] = []

        for entity in sorted(cls._ENTITY_DICT, key=len, reverse=True):
            norm_entity = cls._normalize_entity(entity)
            entity_letter_count = sum(1 for c in norm_entity if c.isalpha())
            dist2_warn_count = 0  # 每个实体最多报 3 条距离=2 的警告

            # 滑动窗口搜索，窗口大小 = len(norm_entity) - 2 ~ +2
            for window_size in range(
                max(1, len(norm_entity) - 2),
                min(len(norm_text), len(norm_entity) + 2) + 1,
            ):
                for start in range(len(norm_text) - window_size + 1):
                    end = start + window_size
                    window = norm_text[start:end]

                    # 快速预滤：窗口与实体首/尾字符不匹配，大概率距离 ≥ 3
                    if window[0] != norm_entity[0] and window[-1] != norm_entity[-1]:
                        # 不是严格过滤，仅做弱预判——仍算距离但加早停
                        pass

                    dist = cls._levenshtein_distance(
                        window, norm_entity, max_dist=2
                    )
                    if dist > 2:
                        continue

                    # 映射回原始文本位置
                    orig_start = norm_to_orig[start]
                    orig_end = (
                        norm_to_orig[end - 1] + 1 if end > 0 else orig_start
                    )
                    # 扩展至尾随的句号/逗号——这些标点在归一化时被丢弃，
                    # 但属于实体名规范格式的一部分（如 L.P. 的句点）。
                    # 不扩展则会导致替换后出现重复标点（L.P..）。
                    while orig_end < len(text) and text[orig_end] in (".", ","):
                        orig_end += 1
                    orig_span = text[orig_start:orig_end]

                    if dist == 0 or dist == 1:
                        # ── 安全阀：防吞所有格等额外字母 ──
                        # 计算原始 span 相对于字典实体多出的字母数
                        span_letter_count = sum(
                            1 for c in orig_span if c.isalpha()
                        )
                        extra_letters = span_letter_count - entity_letter_count
                        if extra_letters > 0:
                            # 多余字母是"可剥离后缀"还是"嵌入词中的 OCR 噪声"？
                            # 如果去掉尾部字母后归一化结果 == 实体名，说明是后缀
                            # （如 's 所有格），拒绝修正。
                            is_suffix = False
                            for trim_n in range(1, extra_letters + 1):
                                if window[:-trim_n] == norm_entity:
                                    is_suffix = True
                                    break
                            if is_suffix:
                                logger.info(
                                    "实体名匹配被安全阀拦截"
                                    "（多余字母为可剥离后缀）: "
                                    "'%s' → '%s'",
                                    orig_span.strip(), entity,
                                )
                                continue

                        matches.append((orig_start, orig_end, entity, dist))
                    elif dist == 2:
                        dist2_warn_count += 1
                        if dist2_warn_count <= 3:
                            logger.warning(
                                "实体名疑似 OCR 错误（编辑距离=2，未自动修正）: "
                                "'%s' → '%s'",
                                orig_span.strip(), entity,
                            )
                        elif dist2_warn_count == 4:
                            logger.warning(
                                "实体 '%s' 距离=2 警告已达上限，"
                                "后续同类警告被抑制。"
                                "如该实体是文档中确实存在的不同实体，"
                                "请将其加入 _ENTITY_DICT。",
                                entity,
                            )

        if not matches:
            return text, {"corrections": 0, "warnings": []}

        # ── 冲突检测：多个候选、重叠 span ──
        # 按编辑距离升序、span 长度降序排列（优先选距离小、覆盖长的匹配）
        # m = (orig_start, orig_end, entity, dist)
        matches.sort(key=lambda m: (m[3], -(m[1] - m[0])))

        warnings: List[str] = []
        kept: List[Tuple[int, int, str, int]] = []
        occupied: List[Tuple[int, int]] = []

        for orig_start, orig_end, entity, dist in matches:
            # 检查是否与已占用区间重叠
            overlaps = any(
                orig_start < occ_end and orig_end > occ_start
                for occ_start, occ_end in occupied
            )
            if overlaps:
                continue

            # 同一 span 出现多个候选：仅当存在相同编辑距离的竞争者
            # 时才视为歧义。距离更小者直接胜出（如距离 0 精确命中
            # 不受距离 1 候选干扰）。
            same_span_others = [
                m for m in matches
                if m[0] == orig_start and m[1] == orig_end and m[2] != entity
            ]
            if same_span_others:
                same_dist_others = [m for m in same_span_others if m[3] == dist]
                if same_dist_others:
                    candidates = [entity] + [m[2] for m in same_dist_others]
                    logger.warning(
                        "实体名修正歧义（多个距离=%d 候选，未自动修正）: "
                        "span='%s', candidates=%s",
                        dist,
                        text[orig_start:orig_end].strip(),
                        candidates,
                    )
                    warnings.append(
                        f"实体名修正歧义: '{text[orig_start:orig_end].strip()}' "
                        f"→ {candidates}"
                    )
                    occupied.append((orig_start, orig_end))
                    continue

            kept.append((orig_start, orig_end, entity, dist))
            occupied.append((orig_start, orig_end))

        if not kept:
            return text, {"corrections": 0, "warnings": warnings}

        # ── 应用替换（从右向左，保证位置不变） ──
        kept.sort(key=lambda m: m[0], reverse=True)

        corrected_text = text
        correction_count = 0
        for orig_start, orig_end, entity, dist in kept:
            old_span = corrected_text[orig_start:orig_end]
            # 跳过无操作替换（span 与实体名完全一致）
            if old_span == entity:
                continue
            corrected_text = (
                corrected_text[:orig_start]
                + entity
                + corrected_text[orig_end:]
            )
            correction_count += 1
            logger.info(
                "实体名修正 (编辑距离=%d): '%s' → '%s'",
                dist, old_span.strip(), entity,
            )

        return corrected_text, {
            "corrections": correction_count,
            "warnings": warnings,
        }

    # ── 向后兼容：全文本校正 ──

    @classmethod
    def correct_text(cls, text: str) -> Tuple[str, Dict]:
        """全文本校正（向后兼容）。

        处理顺序：
        1. 按空白切分为词，逐词调用 correct_word 修正
        2. 实体名模糊匹配修正（编辑距离 ≤1 自动修正，=2 报警）

        多词上下文模式（如 "Partners III L.P."）的词级修正由 correct_word 各自完成。

        Args:
            text: OCR 识别的原始文本

        Returns:
            (修正后的文本, {'corrections': int, 'details': [], 'entity_corrections': int, 'entity_warnings': list})
        """
        if not text:
            return text, {"corrections": 0, "details": [], "entity_corrections": 0, "entity_warnings": []}

        # ── 步骤1：逐词修正 ──
        tokens = re.split(r"(\s+)", text)
        corrected_tokens = []
        word_correction_count = 0

        for token in tokens:
            if token.isspace() or not token:
                corrected_tokens.append(token)
            else:
                corrected = cls.correct_word(token)
                if corrected != token:
                    word_correction_count += 1
                    logger.info(
                        "词级修正: '%s' → '%s'", token.strip(), corrected.strip()
                    )
                corrected_tokens.append(corrected)

        word_corrected_text = "".join(corrected_tokens)

        # ── 步骤2：实体名模糊匹配修正 ──
        entity_corrected_text, entity_result = cls._correct_entities(
            word_corrected_text
        )

        return entity_corrected_text, {
            "corrections": word_correction_count + entity_result["corrections"],
            "details": [],
            "entity_corrections": entity_result["corrections"],
            "entity_warnings": entity_result["warnings"],
        }

    # 向后兼容别名
    correct_ocr_result = correct_text