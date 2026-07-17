#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LanceDB 数据检视工具 —— 查看已有 chunk 的摘要或全文"""
import argparse
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
import lancedb
import pandas as pd

DB_PATH = "./data/lancedb"
TABLE_NAME = "financial_docs"


def list_chunks():
    """列出所有 chunk 的摘要"""
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()

    print(f"\n{'='*60}")
    print(f"  共 {len(df)} 个 chunk")
    print(f"{'='*60}\n")

    for i, (_, row) in enumerate(df.iterrows()):
        text = row["text"]
        parts = text.split("\n")
        # 跳过 [概述] / [表头] 行，从正文开始预览
        body_start = 0
        for j, p in enumerate(parts):
            if not p.startswith("[概述]") and not p.startswith("[表头]"):
                body_start = j
                break
        preview_lines = parts[body_start : body_start + 3]
        preview = "\n    ".join(preview_lines)
        if len(preview) > 150:
            preview = preview[:150] + "..."

        print(
            f"[{i}]  type={row['type']:5s}  "
            f"pages={str(row['page_nums']):10s}  "
            f"{row['length']:5d} chars  "
            f"id={row['id']}"
        )
        print(f"    {preview}")
        print()


def show_chunk(index: int):
    """显示指定 chunk 的完整文本"""
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()

    if index < 0 or index >= len(df):
        print(f"❌ 索引 {index} 越界，共 {len(df)} 个 chunk")
        sys.exit(1)

    row = df.iloc[index]
    print(f"\n{'='*60}")
    print(f"  [{index}]  type={row['type']}  pages={row['page_nums']}  {row['length']} chars")
    print(f"  id={row['id']}  doc_id={row['doc_id']}")
    print(f"{'='*60}\n")
    print(row["text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanceDB 数据检视")
    parser.add_argument(
        "index",
        nargs="?",
        type=int,
        default=None,
        help="chunk 索引（省略则列出全部摘要）",
    )
    args = parser.parse_args()

    if args.index is not None:
        show_chunk(args.index)
    else:
        list_chunks()
