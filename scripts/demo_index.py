#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv(override=True)

import datetime
import logging
import os
import queue
import time

os.makedirs("logs", exist_ok=True)
_file_handler = logging.FileHandler("logs/demo_index.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), _file_handler],
)
for _mod in ("pipeline", "retriever", "reranker", "llm_client",
              "doc_manager", "chunkers.financial_table_chunker", "embedder",
              "progress_reporter", "logging_utils"):
    logging.getLogger(_mod).setLevel(logging.INFO)

import fitz
import numpy as np

from config import Config
# 以下模块依赖 PaddleOCR，改为函数内懒加载，避免 test_llm_chunking 触发导入：
#   from services.indexing.ocr_engine import SecurePDFProcessor
#   from services.pipeline import RAGPipeline
#   from services.indexing.preprocessor import ImagePreprocessor

def main(testFile):
    """
    本函数设计本意是，批量导入PDF，然后批量做查询。
    但是目前阶段，业务聚焦于单个PDF的索引处理体验。
    所以此函数还没有真正用起来，先留个架子在这里。
    """
    from services.pipeline import RAGPipeline
    # 初始化配置
    config = Config()
    rag = RAGPipeline(config)
    
    # 示例：索引PDF文件
    pdf_files = [
        testFile,  # 可替换为多个pdf文件路径
    ]
    
    def index_progress(status_line: str | None, log_line: str | None):
        """索引进度回调。"""
        if status_line:
            print(f"Status Line: {status_line}")
        if log_line:
            print(f"Log Line: {log_line}")

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            result = rag.index_document(pdf_path, on_progress=index_progress)
            print(f"\n索引结果: {result}")
    
    # 执行查询
    print("\n" + "="*50)
    print("开始查询测试")
    print("="*50)
    
    queries = [
        "公司2023年的营业收入是多少？",
        "毛利率是多少？",
        "资产负债率有什么变化？"
    ]
    
    for query in queries:
        print(f"\n❓ 问题: {query}")
        result = rag.query(query)
        print(f"📖 答案预览: {result['answer'][:300]}...")
        print(f"📚 来源数量: {result['retrieved_count']}")
        
        # 打印来源详情
        for i, source in enumerate(result['sources'][:3]):
            print(f"   来源{i+1}: {source['doc_id']} (类型: {source['type']}, 相关度: {source['score']:.3f})")

def test_PaddleOCR_ocr(testFile):
    """测试PaddleOCR识别策略"""
    from services.indexing.ocr_engine import SecurePDFProcessor
    # 创建PaddleOCR策略配置
    config = Config()
    config.extraction_strategy = "PaddleOCR"

    processor = SecurePDFProcessor(config)
    
    # OCR处理计时
    ocr_start = time.time()
    result = processor.process(testFile)
    ocr_time = time.time() - ocr_start
    
    # ========== 文件保存功能 ==========
    # 创建时间戳目录（添加_PaddleOCR后缀）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("temp", f"{timestamp}_PaddleOCR")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n保存 PaddleOCR 识别结果到: {output_dir}")
    
    # 保存识别结果到单个文件
    output_file = os.path.join(output_dir, "PaddleOCR_result.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # ========== 写入统计数据（在最上面）==========
        f.write("=" * 80 + "\n")
        f.write("PaddleOCR 识别策略 - 统计数据\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"测试文件: {testFile}\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"识别策略: PaddleOCR OCR\n")
        f.write(f"总页数: {result['total_pages']}\n")
        f.write(f"实际处理页数: {result.get('processed_pages', 0)}\n")
        f.write(f"总文本长度: {len(result['text'])} 字符\n")
        f.write(f"OCR处理时间: {ocr_time:.2f}秒\n")
        if result.get('processed_pages', 0) > 0:
            avg_time_per_page = ocr_time / result.get('processed_pages', 1)
            f.write(f"平均每页处理时间: {avg_time_per_page:.2f}秒\n")

        confidences = [page.get('avg_confidence', 0) for page in result['pages']]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            f.write(f"平均置信度: {avg_confidence:.3f}\n")
            f.write(f"最低置信度: {min_confidence:.3f}\n")
            f.write(f"最高置信度: {max_confidence:.3f}\n")
        
        f.write(f"是否有错误: {result.get('has_errors', False)}\n\n")

        # ========== 写入详细识别内容 ==========
        f.write("=" * 80 + "\n")
        f.write("详细识别内容\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入每页识别内容
        for page_result in result['pages']:
            page_num = page_result['page']
            page_text = page_result['text']
            page_confidence = page_result['avg_confidence']
            page_tables = page_result.get('tables', [])
            
            f.write(f"\n{'=' * 60}\n")
            f.write(f"页码: {page_num}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"平均置信度: {page_confidence:.3f}\n")
            f.write(f"文本长度: {len(page_text)} 字符\n")
            f.write(f"是否有表格: {len(page_tables) > 0}\n")
            
            # 保存OCR识别的文本内容
            f.write(f"\nOCR识别内容:\n")
            f.write(f"{'-' * 50}\n")
            f.write(f"{page_text}\n")
            f.write("\n")
    
    print(f"  [SUCCESS] PaddleOCR 识别结果已保存到 PaddleOCR_result.txt")
    print(f"  [INFO] 结果目录: {output_dir}")
    
    return {
        'strategy': 'PaddleOCR',
        'result': result,
        'ocr_time': ocr_time,
        'config': config,
        'output_dir': output_dir
    }

def test_original_text_layer(testFile):
    """测试原始文本层提取策略"""
    from services.indexing.ocr_engine import SecurePDFProcessor
    # 创建文本层提取策略配置
    config = Config()
    config.extraction_strategy = "original_text_layer"

    processor = SecurePDFProcessor(config)
    
    # OCR处理计时
    ocr_start = time.time()
    result = processor.process(testFile)
    ocr_time = time.time() - ocr_start
    
    # ========== 文件保存功能 ==========
    # 创建时间戳目录（添加_original_text_layer后缀）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("temp", f"{timestamp}_original_text_layer")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n保存文本层提取结果到: {output_dir}")
    
    # 保存识别结果到单个文件
    output_file = os.path.join(output_dir, "original_text_layer_result.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # ========== 写入统计数据（在最上面）==========
        f.write("=" * 80 + "\n")
        f.write("原始文本层提取策略 - 统计数据\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"测试文件: {testFile}\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"提取策略: 原始文本层 (PDF Text Layer)\n")
        f.write(f"总页数: {result['total_pages']}\n")
        f.write(f"实际处理页数: {result.get('processed_pages', 0)}\n")
        f.write(f"总文本长度: {len(result['text'])} 字符\n")
        f.write(f"处理时间: {ocr_time:.2f}秒\n")
        if result.get('processed_pages', 0) > 0:
            avg_time_per_page = ocr_time / result.get('processed_pages', 1)
            f.write(f"平均每页处理时间: {avg_time_per_page:.2f}秒\n")
        
        # 置信度统计（文本层100%准确）
        confidences = [page.get('avg_confidence', 1.0) for page in result['pages']]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            f.write(f"平均置信度: {avg_confidence:.3f} (文本层100%准确)\n")
            f.write(f"最低置信度: {min_confidence:.3f}\n")
            f.write(f"最高置信度: {max_confidence:.3f}\n")
        
        f.write(f"是否有错误: {result.get('has_errors', False)}\n\n")
        
        # ========== 写入详细识别内容 ==========
        f.write("=" * 80 + "\n")
        f.write("详细文本层内容\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入每页识别内容
        for page_result in result['pages']:
            page_num = page_result['page']
            page_text = page_result['text']
            page_confidence = page_result['avg_confidence']
            page_tables = page_result.get('tables', [])
            
            f.write(f"\n{'=' * 60}\n")
            f.write(f"页码: {page_num}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"平均置信度: {page_confidence:.3f}\n")
            f.write(f"文本长度: {len(page_text)} 字符\n")
            f.write(f"是否有表格: {len(page_tables) > 0} (文本层不包含表格)\n")
            
            # 保存文本层内容
            f.write(f"\nPDF文本层内容:\n")
            f.write(f"{'-' * 50}\n")
            f.write(f"{page_text}\n")
            f.write("\n")
    
    print(f"  [SUCCESS] 文本层提取结果已保存到 original_text_layer_result.txt")
    print(f"  [INFO] 结果目录: {output_dir}")
    
    return {
        'strategy': 'PDF Text Layer',
        'result': result,
        'ocr_time': ocr_time,
        'config': config,
        'output_dir': output_dir
    }

def process_PaddleOCR_coordinate(testFile, page_index=None):
    """逐页处理 PDF：PaddleOCR + 坐标重建表格 + OCR 后处理，全内存。

    与 generate_table_data 的区别：
    - 不写临时 PNG 文件，不启子进程，PaddleOCR 模型只加载一次
    - 表格重建后的 Markdown 直接在内存中做 correct_text 修正
    - 返回所有页的结果字典，不落盘

    Args:
        testFile: PDF 文件路径
        page_index: 仅处理第 N 页（0-based），None 则处理全部

    Returns:
        {'total_pages': int,
         'processed_pages': int,
         'pages': [{'page': int, 'markdown': str,
                    'word_count': int, 'corrections': int, 'elapsed': float}, ...]}
    """
    import gc
    import cv2
    from paddleocr import PaddleOCR
    from services.indexing.borderless_table import reconstruct_table, table_to_markdown
    from services.indexing.post_ocr_character_corrector import PostOcrCharacterCorrector
    from services.indexing.preprocessor import ImagePreprocessor

    config = Config()
    doc = fitz.open(testFile)
    total_pages = len(doc)

    if page_index is not None:
        if page_index < 0 or page_index >= total_pages:
            raise ValueError(f"page_index={page_index} 超出范围 (共 {total_pages} 页)")
        pages_to_process = [page_index]
    else:
        pages_to_process = list(range(total_pages))

    # PaddleOCR 模型只加载一次
    print("加载 PaddleOCR 模型 ...", flush=True)
    ocr = PaddleOCR(
        lang=config.ocr_lang,
        ocr_version="PP-OCRv5",
        use_doc_unwarping=False,
        text_det_limit_side_len=1280,
        text_det_thresh=0.2,
        text_det_box_thresh=0.4,
        text_det_unclip_ratio=1.9,
    )

    corrector = PostOcrCharacterCorrector()
    page_results = []
    _ROMAN_CHARS = set("IVX")

    def _is_roman(s):
        return all(c in _ROMAN_CHARS for c in s)

    try:
        for page_num in pages_to_process:
            print(f"  处理第 {page_num + 1}/{total_pages} 页 ...", flush=True)
            t_start = time.time()

            # 本页所有需清理的资源，正常/异常路径统一释放
            img = None
            results = None
            all_words = None
            coord_table = None
            coord_md = None
            corrected_md = None
            stats = None

            try:
                # ── 1. PDF 页面 → numpy 图片 ──
                page = doc[page_num]
                pix = page.get_pixmap(dpi=config.ocr_dpi)
                samples = bytes(pix.samples)
                h, w, n = pix.height, pix.width, pix.n
                pix = None  # 立即释放 fitz Pixmap（C 层内存不经 Python GC）
                page = None  # fitz Page 引用释放
                img = np.frombuffer(samples, dtype=np.uint8).reshape(h, w, n)
                del samples
                if n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # ── 2. 图像预处理（裁空白等） ──
                if config.enable_preprocessing:
                    before_rows = img.shape[0]
                    img = ImagePreprocessor.preprocess_for_paddleocr(img)
                    if img.shape[0] != before_rows:
                        print(
                            f"    [CROP] 裁切空白: {before_rows} → {img.shape[0]} 行",
                            flush=True,
                        )

                # BGR → RGB（PaddleOCR 要求）
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # ── 3. PaddleOCR 检测 + 识别 ──
                results = ocr.predict(input=[img])
                del img
                img = None

                all_words = []
                for res in results:
                    rec_boxes = res.get("rec_boxes", [])
                    rec_texts = res.get("rec_texts", [])
                    rec_scores = res.get("rec_scores", [])
                    if hasattr(rec_boxes, "tolist"):
                        rec_boxes = rec_boxes.tolist()
                    for text, score, bbox in zip(rec_texts, rec_scores, rec_boxes):
                        bbox = bbox if isinstance(bbox, list) else bbox.tolist()
                        all_words.append({
                            "value": text,
                            "confidence": int(100 * score),
                            "x1": int(bbox[0]), "y1": int(bbox[1]),
                            "x2": int(bbox[2]), "y2": int(bbox[3]),
                        })
                del results
                results = None

                all_words.sort(key=lambda w: w["y1"])

                # ── 4. 合并相邻罗马数字碎片 ──
                merged_words = []
                skip_next = False
                for i, w in enumerate(all_words):
                    if skip_next:
                        skip_next = False
                        continue
                    nxt = all_words[i + 1] if i + 1 < len(all_words) else None
                    if (nxt and _is_roman(w["value"]) and _is_roman(nxt["value"])
                            and abs(w["x1"] - nxt["x1"]) < 80
                            and abs(w["y1"] - nxt["y1"]) < 30):
                        merged_words.append({
                            "value": w["value"] + nxt["value"],
                            "confidence": min(w["confidence"], nxt["confidence"]),
                            "x1": w["x1"], "y1": w["y1"],
                            "x2": nxt["x2"], "y2": nxt["y2"],
                        })
                        skip_next = True
                    else:
                        merged_words.append(w)
                all_words = merged_words

                # ── 5. 过滤空词 → 坐标重建表格 → Markdown ──
                all_words = [w for w in all_words if w["value"].strip()]
                coord_table = reconstruct_table(all_words, min_confidence=30)
                coord_md = table_to_markdown(coord_table)

                # ── 6. OCR 后处理字符修正 ──
                corrected_md, stats = corrector.correct_text(coord_md)

                elapsed = time.time() - t_start
                print(
                    f"    完成 ({elapsed:.0f}s), "
                    f"修正 {stats['corrections']} 处"
                    f"（实体 {stats.get('entity_corrections', 0)}）",
                    flush=True,
                )

                page_results.append({
                    "page": page_num + 1,
                    "markdown": corrected_md,
                    "word_count": len(all_words),
                    "corrections": stats["corrections"],
                    "entity_corrections": stats.get("entity_corrections", 0),
                    "elapsed": elapsed,
                })

            except Exception:
                elapsed = time.time() - t_start
                print(
                    f"    [ERROR] 第 {page_num + 1} 页失败 ({elapsed:.0f}s)",
                    flush=True,
                )
                import traceback
                traceback.print_exc()
                # 失败页写入空占位
                page_results.append({
                    "page": page_num + 1,
                    "markdown": "",
                    "word_count": 0,
                    "corrections": 0,
                    "entity_corrections": 0,
                    "elapsed": elapsed,
                    "error": True,
                })

            finally:
                # ── 无论如何，释放本页全部资源 ──
                del img, results, all_words, coord_table, coord_md
                del corrected_md, stats
                gc.collect()

    finally:
        doc.close()
        # 释放提取环节的全部资源（模型、校正器等）
        del ocr, corrector
        gc.collect()

    total_elapsed = sum(p["elapsed"] for p in page_results)
    print(
        f"\n  总计: {len(page_results)}/{total_pages} 页, "
        f"耗时 {total_elapsed:.0f}s"
    )

    return {
        "total_pages": total_pages,
        "processed_pages": len(page_results),
        "pages": page_results,
    }

def test_PaddleOCR_coordinate_ocr(testFile):
    from services.indexing.ocr_engine import SecurePDFProcessor
    from config import DebugConfig
    # ── 全内存方案: 走 SecurePDFProcessor.process() ──
    config = Config()
    config.extraction_strategy = "PaddleOCR + coordinate"
    processor = SecurePDFProcessor(config)

    t_start = time.time()
    result = processor.process(testFile)
    elapsed = time.time() - t_start

    # 统计
    total_corrections = sum(p.get("corrections", 0) for p in result["pages"])
    total_entity = sum(p.get("entity_corrections", 0) for p in result["pages"])

    # 输出到 DebugConfig.ocr_cache_dir / <时间戳> /
    # 时间戳格式便于人眼识别：2026-07-14_09-30-15
    cache_dir = DebugConfig.ocr_cache_dir
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(cache_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    for p in result["pages"]:
        idx = p["page"]  # 1-based，与真实页码一致
        out_file = os.path.join(out_dir, f"{idx}.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(p["text"])
        print(f"  第 {p['page']} 页 → {out_file}")

    print(
        f"\n全部 {result['processed_pages']}/{result['total_pages']} 页"
        f" 耗时 {elapsed:.0f}s"
        f" 修正 {total_corrections} 处"
        f"（实体 {total_entity}）"
        f"\n缓存目录: {out_dir}"
        f"\n使用方式: 设置 DebugConfig.use_ocr_cache = True 后重新启动"
    )


def test_ocr(testFile):
    # 测试 PaddleOCR 提取策略
    # print("\n 测试 PaddleOCR 提取策略...")
    # PaddleOCR_result = test_PaddleOCR_ocr(testFile)
    # print(f"PaddleOCR处理时间: {PaddleOCR_result['ocr_time']:.2f}秒")
    # print(f"识别字符数: {len(PaddleOCR_result['result']['text'])}")

    # 测试 原始文本层 提取策略
    # print("\n 测试原始文本层提取策略...")
    # original_text_layer_result = test_original_text_layer(testFile)

    # 测试 PaddleOCR + coordinate 提取策略
    print("\n 测试 PaddleOCR + coordinate 提取策略...")
    PaddleOCR_coordinate_result = test_PaddleOCR_coordinate_ocr(testFile)

def generate_table_data(pdf_path: str, page_index: int = None) -> str:
    """逐页提取 PDF 的表格数据。

    每页在独立子进程中运行，结果写入单独文件：
        data/<pdf名>_<md5>/PaddleOcr/coordinate/<index>.txt
    每个 txt 文件为当前页表格数据的 JSON 数组，无表格时为空文件。

    Args:
        pdf_path: PDF 文件的绝对或相对路径。
        page_index: 仅处理第 N 页（0-based，对应 index 文件名）。
                    为 None 时处理全部页面。

    Returns:
        输出目录路径。

    Raises:
        FileNotFoundError: PDF 文件不存在。
    """
    import hashlib
    import subprocess
    import sys
    import tempfile

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # ── 1. 计算 MD5，建立输出目录 ──
    with open(pdf_path, "rb") as f:
        pdf_md5 = hashlib.md5(f.read()).hexdigest()
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.join(data_dir, f"{pdf_name}_{pdf_md5}", "PaddleOcr/coordinate")
    os.makedirs(output_dir, exist_ok=True)
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # ── 2. 读取配置（仅取 DPI 和 lang，主进程不加载 PaddleOCR） ──
    config = Config()
    ocr_dpi = config.ocr_dpi
    ocr_lang = config.ocr_lang
    enable_preprocess = config.enable_preprocessing

    # ── 3. 构建待处理页码列表 ──
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if page_index is not None:
        if page_index < 0 or page_index >= total_pages:
            raise ValueError(
                f"page_index={page_index} 超出范围 (PDF 共 {total_pages} 页)"
            )
        pages_to_process = [page_index]
    else:
        pages_to_process = list(range(total_pages))

    # ── 4. 逐页：PaddleOCR → 坐标重建表格 → 写 Markdown ──
    for page_num in pages_to_process:
        out_file = os.path.join(output_dir, f"{page_num + 1}.txt")
        if os.path.exists(out_file):
            print(f"  [table] 第 {page_num + 1}/{total_pages} 页 已存在, 跳过")
            continue

        print(f"  [table] 第 {page_num + 1}/{total_pages} 页 ...", flush=True)
        t_start = time.time()

        tmp_path = None
        try:
            # 主进程：PDF 页面 → PNG 临时文件
            page = doc[page_num]
            pix = page.get_pixmap(dpi=ocr_dpi)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            import cv2
            from services.indexing.preprocessor import ImagePreprocessor
            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            if enable_preprocess:
                before_rows = img_array.shape[0]
                img_array = ImagePreprocessor.preprocess_for_paddleocr(img_array)
                if img_array.shape[0] != before_rows:
                    print(f"    [CROP] 裁切空白: {before_rows} → {img_array.shape[0]} 行", flush=True)

            # BGR → RGB
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            from PIL import Image as PILImage
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="table_")
            os.close(tmp_fd)
            PILImage.fromarray(img_array).save(tmp_path, format="PNG")

            # 调试：保存预处理后的图像，验证送进 PaddleOCR 的图是否正确
            debug_img_file = os.path.join(output_dir, f"{page_num + 1}_preprocessed.png")
            import shutil
            shutil.copy(tmp_path, debug_img_file)

            # 子进程：纯 PaddleOCR → 坐标重建 → 写 Markdown
            debug_file = os.path.join(output_dir, f"{page_num + 1}_paddleocr_debug.txt")
            script = (
                "import sys\n"
                f"sys.path.insert(0, {project_dir!r})\n"
                "from services.indexing.borderless_table import reconstruct_table, table_to_markdown\n"
                "from services.indexing.post_ocr_character_corrector import PostOcrCharacterCorrector\n"
                "import cv2\n"
                "from paddleocr import PaddleOCR\n"
                f"img = cv2.imread({tmp_path!r})\n"
                "img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n"
                f"ocr = PaddleOCR(lang={ocr_lang!r}, ocr_version='PP-OCRv5', use_doc_unwarping=False, text_det_limit_side_len=1280, text_det_thresh=0.2, text_det_box_thresh=0.4, text_det_unclip_ratio=1.9)\n"
                "# ── PaddleOCR 检测 ──\n"
                "results = ocr.predict(input=[img])\n"
                "all_words = []\n"
                "debug_lines = ['行号\\t置信度\\t文字\\t(y1,y2,x1,x2)']\n"
                "for res in results:\n"
                "    rec_boxes = res.get('rec_boxes', [])\n"
                "    rec_texts = res.get('rec_texts', [])\n"
                "    rec_scores = res.get('rec_scores', [])\n"
                "    if hasattr(rec_boxes, 'tolist'):\n"
                "        rec_boxes = rec_boxes.tolist()\n"
                "    for text, score, bbox in zip(rec_texts, rec_scores, rec_boxes):\n"
                "        bbox = bbox if isinstance(bbox, list) else bbox.tolist()\n"
                "        all_words.append({\n"
                "            'value': text,\n"
                "            'confidence': int(100 * score),\n"
                "            'x1': int(bbox[0]), 'y1': int(bbox[1]),\n"
                "            'x2': int(bbox[2]), 'y2': int(bbox[3]),\n"
                "        })\n"
                "all_words.sort(key=lambda w: w['y1'])\n"
                "for i, w in enumerate(all_words):\n"
                "    debug_lines.append(\n"
                "        f\"{i}\\t{w['confidence']}\\t{w['value']}\\t\"\n"
                "        f\"({w['y1']},{w['y2']},{w['x1']},{w['x2']})\"\n"
                "    )\n"
                f"with open({debug_file!r}, 'w', encoding='utf-8') as f:\n"
                "    f.write('\\n'.join(debug_lines))\n"
                "# ── 合并相邻罗马数字碎片（如 I + II → III, I + I → II） ──\n"
                "_ROMAN_CHARS = set('IVX')\n"
                "def _is_roman(s): return all(c in _ROMAN_CHARS for c in s)\n"
                "merged_words = []\n"
                "skip_next = False\n"
                "for i, w in enumerate(all_words):\n"
                "    if skip_next:\n"
                "        skip_next = False\n"
                "        continue\n"
                "    nxt = all_words[i + 1] if i + 1 < len(all_words) else None\n"
                "    if (nxt and _is_roman(w['value']) and _is_roman(nxt['value'])\n"
                "        and abs(w['x1'] - nxt['x1']) < 80\n"
                "        and abs(w['y1'] - nxt['y1']) < 30):\n"
                "        merged = w['value'] + nxt['value']\n"
                "        merged_words.append({\n"
                "            'value': merged,\n"
                "            'confidence': min(w['confidence'], nxt['confidence']),\n"
                "            'x1': w['x1'], 'y1': w['y1'],\n"
                "            'x2': nxt['x2'], 'y2': nxt['y2'],\n"
                "        })\n"
                "        skip_next = True\n"
                "    else:\n"
                "        merged_words.append(w)\n"
                "all_words = merged_words\n"
                "# ── 过滤空词（PaddleOCR 偶发检测到空白框） ──\n"
                "all_words = [w for w in all_words if w['value'].strip()]\n"
                "# ── 坐标重建表格 → Markdown ──\n"
                "coord_table = reconstruct_table(all_words, min_confidence=30)\n"
                "coord_md = table_to_markdown(coord_table)\n"
                "# ── OCR 后处理字符修正（在最终 Markdown 文本上直接做） ──\n"
                "corrector = PostOcrCharacterCorrector()\n"
                "import sys as _sys\n"
                "import logging as _logging\n"
                "_logging.basicConfig(\n"
                "    level=_logging.INFO, format='%(message)s', stream=_sys.stderr\n"
                ")\n"
                "corrected_md, stats = corrector.correct_text(coord_md)\n"
                f"with open({out_file!r}, 'w', encoding='utf-8') as f:\n"
                "    f.write(corrected_md)\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=None,
            )
            elapsed = time.time() - t_start
            if proc.returncode != 0:
                err = proc.stderr.strip()[:500]
                print(f"    [ERROR] 子进程退出 {proc.returncode}: {err}")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write("")
            else:
                if proc.stderr:
                    for line in proc.stderr.strip().split('\n'):
                        print(f"    {line}", flush=True)
                print(f"    完成 ({elapsed:.0f}s)", flush=True)

        except Exception as e:
            import traceback
            elapsed = time.time() - t_start
            print(f"    [ERROR] 第 {page_num + 1} 页失败 ({elapsed:.0f}s): {e}")
            traceback.print_exc()
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    doc.close()

    processed = len(pages_to_process)
    print(f"\n  完成: {processed} 页 → {output_dir}")
    return output_dir

def test_llm_chunking(test_file):
    from services.indexing.ocr_engine import SecurePDFProcessor
    # 取得OCR数据
    config = Config()
    config.extraction_strategy = "PaddleOCR + coordinate"
    processor = SecurePDFProcessor(config)
    t_start = time.time()
    ocr_result = processor.try_process(test_file) # 走临时代码，从已有结果里拿现成的
    ocr_elapsed = time.time() - t_start
    if not ocr_result:
        print("  ❌ OCR 提取失败，无法继续分块测试")
        return
    print(f"  [OCR阶段] 耗时 {ocr_elapsed:.1f}s，共 {len(ocr_result) - 1} 页")

    # 拿着OCR数据，找 llm 做分块
    from services.indexing.chunkers.financial_table_chunker import FinancialTableChunker
    from utils import compute_file_md5, extract_doc_name
    chunker = FinancialTableChunker(config)
    t_chunk = time.time()
    chunks, _chunk_tokens = chunker.chunk_pages(
        ocr_result,
        compute_file_md5(test_file),
        extract_doc_name(test_file),
    )
    chunk_elapsed = time.time() - t_chunk
    print(f"  [分块阶段] 耗时 {chunk_elapsed:.1f}s，共 {len(chunks)} 个 chunk")

    # 嵌入 + 入库
    from services.retrieval.embedder import Embedder
    from services.retrieval.retriever import LanceDBHybridRetriever
    embedder = Embedder(config)
    retriever = LanceDBHybridRetriever(config)
    t_store = time.time()
    embeddings = embedder.encode([c['text'] for c in chunks])
    result = retriever.add_chunks(chunks, embeddings)
    store_elapsed = time.time() - t_store
    if result.get('skipped'):
        print(f"  [入库阶段] 耗时 {store_elapsed:.1f}s，文档已存在，跳过")
    else:
        print(f"  [入库阶段] 耗时 {store_elapsed:.1f}s，已存入 {result['added']} 个 chunk")

def get_test_file():
    file_path = ".\data\dev_artifacts\agent开发作业样本.pdf" # 替换为实际的测试文件路径
    if not os.path.exists(file_path):
        print(f"❌ 测试文件 {file_path} 不存在")
        print("请将测试文件放在当前目录下，然后重试")
        exit(1)
    return file_path

if __name__ == "__main__":
    test_file = get_test_file()

    # 暂留
    # main()

    # [测试 OCR] 从pdf提取md，数据落地方案：提取所有页
    # generate_table_data(test_file)
    # [测试 OCR] 从pdf提取md，数据落地方案：只提取idx为 page_index 的页
    # generate_table_data(test_file， page_index=0)
    # [测试 OCR] 从pdf提取md，数据进内存方案。调用的就是pipeline里实际使用的方法。
    test_ocr(test_file)

    # [测试 llm 分块，然后入库]
    # test_llm_chunking(test_file)
