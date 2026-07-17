"""基于 PaddleOCR 文字坐标重建无边框表格。

用 OCR 检测到的每个字的坐标进行行聚类 + 列投影，绕开 img2table
ARLSA 经典算法对粗体/大字/异常字形的结构误判。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils import get_file_logger
logger = get_file_logger(__file__)


def _cluster_rows(
    words: list[dict[str, Any]],
    overlap_ratio: float = 0.35,
) -> list[list[dict[str, Any]]]:
    """按 y 坐标重叠度将词语聚类为行。

    Args:
        words: 按 y1 排序后的词语列表
        overlap_ratio: 判定为同一行的最小纵向重叠比例

    Returns:
        每行为一个词语列表
    """
    if not words:
        return []

    # 计算中位字高，用于后续阈值
    heights = [w["y2"] - w["y1"] for w in words]
    median_h = float(np.median(heights)) if heights else 1.0

    rows: list[list[dict[str, Any]]] = []
    current_row = [words[0]]
    cur_y1, cur_y2 = words[0]["y1"], words[0]["y2"]

    for w in words[1:]:
        overlap = min(cur_y2, w["y2"]) - max(cur_y1, w["y1"])
        min_height = min(w["y2"] - w["y1"], cur_y2 - cur_y1)

        # 纵向无明显重叠 → 新行
        if overlap < overlap_ratio * max(min_height, median_h * 0.5):
            rows.append(current_row)
            current_row = [w]
            cur_y1, cur_y2 = w["y1"], w["y2"]
        else:
            current_row.append(w)
            cur_y1 = min(cur_y1, w["y1"])
            cur_y2 = max(cur_y2, w["y2"])

    if current_row:
        rows.append(current_row)

    return rows


def _detect_column_boundaries(
    all_rows: list[list[dict[str, Any]]],
    min_col_gap: int = 8,
    gap_row_ratio: float = 0.25,
) -> list[int]:
    """通过跨行空白投影检测列分隔线。

    将所有行的文字区域叠加到 x 轴上，覆盖率低于 gap_row_ratio 的
    连续区域视为列间空白。

    Args:
        all_rows: 行聚类结果
        min_col_gap: 列间最小像素间距
        gap_row_ratio: 空白区域允许被多少比例的行覆盖仍视为间隙

    Returns:
        列边界 x 坐标列表（含首尾），长度为 列数+1
    """
    # ── 滤除全宽标题行（如页眉、公司名），它们跨越多列会污染列投影 ──
    max_x = max((w["x2"] for row in all_rows for w in row), default=0)
    min_x = min((w["x1"] for row in all_rows for w in row), default=0)
    table_width = max_x - min_x

    table_rows: list[list[dict[str, Any]]] = []
    for row in all_rows:
        row_x1 = min(w["x1"] for w in row)
        row_x2 = max(w["x2"] for w in row)
        # 单字跨表宽 80% 以上 → 标题行，跳过
        if len(row) == 1 and (row_x2 - row_x1) >= 0.8 * table_width:
            logger.debug("_detect_column_boundaries: 跳过全宽标题行 y=%d", row[0]["y1"])
            continue
        table_rows.append(row)

    if not table_rows:
        return [min_x, max_x]

    all_x1 = [w["x1"] for row in table_rows for w in row]
    all_x2 = [w["x2"] for row in table_rows for w in row]
    if not all_x1:
        return []

    x_min = min(all_x1)
    x_max = max(all_x2)
    width = x_max - x_min
    if width <= 0:
        return [x_min, x_max]

    # 逐行构建覆盖掩码，累加得到跨行覆盖次数
    num_rows = len(table_rows)
    coverage = np.zeros(width, dtype=np.int32)
    for row in table_rows:
        row_mask = np.zeros(width, dtype=np.bool_)
        for w in row:
            start = max(0, w["x1"] - x_min)
            end = min(width, w["x2"] - x_min)
            row_mask[start:end] = True
        coverage += row_mask.astype(np.int32)

    # 覆盖行数低于阈值的连续区域 = 列间隙
    gap_threshold = max(1, int(num_rows * gap_row_ratio))
    gaps: list[tuple[int, int]] = []  # (gap_start, gap_end) 绝对坐标
    in_gap = False
    gap_start = 0

    for x in range(width):
        if coverage[x] <= gap_threshold:
            if not in_gap:
                gap_start = x
                in_gap = True
        else:
            if in_gap:
                if x - gap_start >= min_col_gap:
                    gaps.append((gap_start + x_min, x - 1 + x_min))
                in_gap = False

    if in_gap and width - gap_start >= min_col_gap:
        gaps.append((gap_start + x_min, x_max))

    logger.debug(
        "_detect_column_boundaries: num_rows=%d, gap_threshold=%d, gaps=%s",
        num_rows, gap_threshold,
        [(gs, ge, ge - gs) for gs, ge in gaps],
    )

    # 从间隙中提取列分隔线
    boundaries = [x_min]
    for g_start, g_end in gaps:
        boundaries.append((g_start + g_end) // 2)
    boundaries.append(x_max)

    return boundaries


def _assign_cells(
    row_words: list[dict[str, Any]],
    col_boundaries: list[int],
) -> list[str]:
    """将一行内的词语按 x 中心分配到对应列。

    Args:
        row_words: 一行中的所有词语
        col_boundaries: _detect_column_boundaries 的输出

    Returns:
        每列一个字符串（列内多词用空格拼接）
    """
    nb_cols = len(col_boundaries) - 1
    # 存储 (x_center, value) 以便列内按 x 坐标排序，
    # 避免 PaddleOCR 返回的词序不稳定导致 "末尾单词插入到前面"。
    cells: list[list[tuple[float, str]]] = [[] for _ in range(nb_cols)]

    for w in row_words:
        x_center = (w["x1"] + w["x2"]) / 2
        for i in range(nb_cols):
            if col_boundaries[i] <= x_center <= col_boundaries[i + 1]:
                cells[i].append((x_center, w["value"]))
                break

    return [
        " ".join(v for _, v in sorted(cell, key=lambda x: x[0]))
        for cell in cells
    ]


def reconstruct_table(
    words: list[dict[str, Any]],
    min_confidence: int = 0,
    min_col_gap: int = 8,
) -> list[list[str]]:
    """从 PaddleOCR 原始词语坐标重建无边框表格。

    三步：行聚类 → 列投影 → 单元格分配。

    Args:
        words: PaddleOCR 输出的词语列表，每项含
               value, x1, y1, x2, y2, confidence
        min_confidence: 最低置信度（0-100），低于此值的词语丢弃
        min_col_gap: 列间最小空白宽度（像素）

    Returns:
        二维表格，table[row][col] = 单元格文本
    """
    # 过滤低置信度词语
    words = [w for w in words if w.get("confidence", 100) >= min_confidence]
    if not words:
        return []

    # 按 y1 排序
    words.sort(key=lambda w: (w["y1"], w["x1"]))

    # 1. 行聚类
    rows = _cluster_rows(words)
    if not rows:
        return []

    logger.debug(
        "reconstruct_table: %d words → %d rows",
        len(words), len(rows),
    )

    # 2. 列检测
    col_boundaries = _detect_column_boundaries(rows, min_col_gap=min_col_gap)
    nb_cols = len(col_boundaries) - 1

    if nb_cols <= 1:
        # 没检测到列分隔 → 整行作为单列输出
        logger.debug("reconstruct_table: 未检测到列分隔，单列输出")
        return [[" ".join(w["value"] for w in row)] for row in rows]

    logger.debug("reconstruct_table: 检测到 %d 列", nb_cols)

    # 3. 单元格分配
    table = []
    for row_words in rows:
        row_cells = _assign_cells(row_words, col_boundaries)
        table.append(row_cells)

    return table


def table_to_markdown(table: list[list[str]]) -> str:
    """将二维表格转为 Markdown 格式字符串。

    Args:
        table: reconstruct_table 的输出

    Returns:
        Markdown 表格字符串
    """
    if not table:
        return ""

    lines: list[str] = []
    for row in table:
        # 转义单元格内的管道符
        escaped = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")

    return "\n".join(lines)
