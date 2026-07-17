import gc
import logging
import os
import time
from typing import Callable, Dict, List, Optional

import fitz
import numpy as np
import cv2

import re
from paddleocr import PaddleOCR
from services.indexing.preprocessor import ImagePreprocessor
from config import Config, DebugConfig, LogConfig
from services.indexing.post_ocr_character_corrector import PostOcrCharacterCorrector
from services.indexing.borderless_table import reconstruct_table, table_to_markdown
from utils import get_file_logger, write_debug_data, write_debug_data_lines
logger = get_file_logger(__file__)


class PDFOpenError(Exception):
    """PDF打开失败的异常"""
    pass

class SecurePDFProcessor:
    """安全的PDF处理器：只用OCR结果，后续考虑对比文字层防篡改"""
    
    def __init__(self, config: Config = Config()):
        self.config = config

        # ========== 0. OCR后处理字符 ==========
        self.character_corrector = PostOcrCharacterCorrector()

        # ========== 1. 文本识别引擎 (PP-OCRv5) ==========
        # 懒加载：PaddleOCR 模型初始化耗时较长，推迟到首次调用 do_OCR 时再加载。
        # 避免搜索管线（不需要 OCR）在启动时等待模型加载。
        self._ocr = None

        # Debug data log：记录逐页 OCR 原始结果
        self._debug_logger: Optional[logging.Logger] = None
        if LogConfig.DEBUG_DATA_MODE != "off":
            from utils import get_debug_data_logger
            self._debug_logger = get_debug_data_logger(__file__)

    @property
    def ocr(self):
        """PaddleOCR 引擎——首次访问时初始化，后续复用。"""
        if self._ocr is None:
            self._ocr = PaddleOCR(
                lang=self.config.ocr_lang,
                ocr_version="PP-OCRv5",
                use_doc_unwarping=False,
                text_det_limit_side_len=1280,
                text_det_thresh=0.2,
                text_det_box_thresh=0.4,
                text_det_unclip_ratio=1.9,
            )
        return self._ocr

    def _preprocess_image(self, page_image: np.ndarray) -> np.ndarray:
        """图像预处理（如启用）"""
        if self.config.enable_preprocessing:
            return ImagePreprocessor.preprocess_for_paddleocr(page_image)
        return page_image

    def _process_with_PaddleOCR(self, page_image: np.ndarray, page_num: int) -> Dict:
        """
        PaddleOCR文字识别策略
        
        Args:
            page_image: 页面图像
            page_num: 页码
            
        Returns:
            识别结果：{'text': str, 'tables': [], 'avg_confidence': float}
        """
        result = {
            'text': '',
            'tables': [],
            'avg_confidence': 0.0
        }
        
        try:
            # 图像预处理
            processed = self._preprocess_image(page_image)

            # 调试信息：检查图像
            logger.debug("原始图像: type=%s, shape=%s", type(page_image), page_image.shape if hasattr(page_image, "shape") else "N/A")
            logger.debug("预处理图像: type=%s, shape=%s", type(processed), processed.shape if processed is not None and hasattr(processed, "shape") else "N/A")
            
            # OCR识别 - 使用正确的API调用
            ocr_result = self.ocr.ocr(processed)
                        
            # 处理OCR结果
            texts = []
            confidences = []
            
            if ocr_result and len(ocr_result) > 0:
                # 新版PaddleOCR返回对象字典格式
                result_dict = ocr_result[0]
                
                # 更精确的判断逻辑
                if isinstance(result_dict, dict):
                    # 如果是字典
                    rec_texts = result_dict.get('rec_texts', [])
                    rec_scores = result_dict.get('rec_scores', [])
                    logger.debug("从字典获取: rec_texts=%d, rec_scores=%d", len(rec_texts), len(rec_scores))
                else:
                    # 如果是对象（OCRResult类）
                    try:
                        # 尝试作为对象处理
                        rec_texts = getattr(result_dict, 'rec_texts', [])
                        rec_scores = getattr(result_dict, 'rec_scores', [])
                        logger.debug("从对象获取: rec_texts=%d, rec_scores=%d", len(rec_texts), len(rec_scores))
                    except Exception as e:
                        logger.debug("对象属性访问失败: %s", e)
                        # 尝试作为字典处理
                        rec_texts = []
                        rec_scores = []
                
                # 确保文本和置信度数量匹配
                min_len = min(len(rec_texts), len(rec_scores))
                for i in range(min_len):
                    if rec_texts[i]:
                        texts.append(rec_texts[i])
                        confidences.append(rec_scores[i])
                
                logger.debug("最终提取: texts=%d, confidences=%d", len(texts), len(confidences))
                
                # 拼接文本
                page_ocr_text = '\n'.join(texts)

                # 应用OCR后处理字符修正
                page_ocr_text, char_correction_stats = self.character_corrector.correct_ocr_result(
                    page_ocr_text
                )

                if char_correction_stats['corrections'] > 0:
                    logger.debug("OCR后处理字符修正: %d 个", char_correction_stats["corrections"])
                
                result['text'] = page_ocr_text
                result['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
                
                logger.debug(
                    "PaddleOCR 识别完成: %d 字符, 置信度 %.2f%%",
                    len(page_ocr_text), result["avg_confidence"] * 100,
                )

        except Exception as e:
                logger.error("PaddleOCR 识别失败: %s", e)
        
        return result
    
    # ── 罗马数字字符合并工具 ──
    _ROMAN_CHARS: frozenset[str] = frozenset("IVX")

    @staticmethod
    def _is_roman(s: str) -> bool:
        return all(c in SecurePDFProcessor._ROMAN_CHARS for c in s)

    def _process_with_PaddleOCR_coordinate(
        self, page_image: np.ndarray, page_num: int
    ) -> Dict:
        """PaddleOCR + 坐标重建表格 + OCR 后处理（主打方案）。

        与 _process_with_PaddleOCR 的区别：
        - 使用 predict() 获取带坐标的词级结果
        - 合并相邻罗马数字碎片后用坐标重建无边框表格
        - 在 Markdown 输出上做 correct_text 修正

        Returns:
            {'markdown': str, 'avg_confidence': float}
        """
        result: Dict = {"markdown": "", "avg_confidence": 0.0}
        processed = None
        ocr_results = None
        all_words = None
        coord_table = None
        markdown = None
        corrected_md = None
        stats = None

        try:
            t_start = time.time()

            # ── 1. 图像预处理 ──
            processed = self._preprocess_image(page_image)

            # BGR → RGB（predict 需要 RGB）
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            # ── 2. PaddleOCR 检测 + 识别（带坐标） ──
            ocr_results = self.ocr.predict(input=[processed])
            del processed
            processed = None

            all_words = []
            confidences = []
            for res in ocr_results:
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
                    confidences.append(score)

            all_words.sort(key=lambda w: w["y1"])

            # ── 3. 合并相邻罗马数字碎片 ──
            merged_words = []
            skip_next = False
            for i, w in enumerate(all_words):
                if skip_next:
                    skip_next = False
                    continue
                nxt = all_words[i + 1] if i + 1 < len(all_words) else None
                if (nxt
                        and self._is_roman(w["value"])
                        and self._is_roman(nxt["value"])
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

            # ── 4. 过滤空词 → 坐标重建表格 → Markdown ──
            all_words = [w for w in all_words if w["value"].strip()]
            coord_table = reconstruct_table(all_words, min_confidence=30)
            markdown = table_to_markdown(coord_table)

            # ── 5. OCR 后处理字符修正 ──
            corrected_md, stats = self.character_corrector.correct_text(markdown)

            result["markdown"] = corrected_md
            result["avg_confidence"] = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            result["corrections"] = stats["corrections"]
            result["entity_corrections"] = stats.get("entity_corrections", 0)

            elapsed = time.time() - t_start
            logger.info(
                "PaddleOCR + coord 页 %d: %.1fs, 修正 %d 处（实体 %d）",
                page_num + 1, elapsed, stats["corrections"],
                stats.get("entity_corrections", 0),
            )

        except Exception as e:
            logger.error("PaddleOCR + coordinate 第 %d 页失败: %s", page_num, e)
            import traceback
            traceback.print_exc()

        finally:
            del processed, ocr_results, all_words, coord_table
            del markdown, corrected_md, stats

        return result

    def _process_with_original_text_layer(self, page: fitz.Page, page_num: int) -> Dict:
        """
        原始文本层提取策略（直接提取PDF文本层）
        
        Args:
            page: fitz.Page对象
            page_num: 页码
            
        Returns:
            识别结果：{'text': str, 'tables': List[Dict], 'avg_confidence': float}
        """
        result = {
            'text': '',
            'tables': [],
            'avg_confidence': 1.0  # 文本层置信度设为最高
        }
        
        try:
            # 提取文本层
            text_layer = page.get_text("text")
            
            # 清理文本（去除多余空行）
            lines = [line.strip() for line in text_layer.split('\n')]
            lines = [line for line in lines if line]  # 移除空行
            page_text = '\n'.join(lines)
            result['text'] = page_text
        except Exception as e:
            logger.error("文本层提取第 %d 页失败: %s", page_num, e)
            import traceback
            traceback.print_exc()
        
        return result

    def process(
        self,
        pdf_path: str,
        on_progress: Optional[Callable] = None,
    ) -> Dict:
        """
        处理PDF，返回 OCR 内容。

        根据 config.extraction_strategy 选择策略：
        - "PaddleOCR + coordinate": 主打方案，坐标重建表格 + OCR 后处理
        - "PaddleOCR":              纯文本 OCR + 纠正
        - "original_text_layer":    提取 PDF 文本层

        Raises:
            PDFOpenError: PDF文件无法打开
        """
        if not os.path.exists(pdf_path):
            raise PDFOpenError(f"文件不存在: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except fitz.fitz.FileDataError as e:
            raise PDFOpenError(f"PDF文件损坏或格式错误: {pdf_path}, 错误: {e}")
        except RuntimeError as e:
            if "password" in str(e).lower():
                raise PDFOpenError(f"PDF有密码保护，无法打开: {pdf_path}")
            raise PDFOpenError(f"无法打开PDF: {pdf_path}, 错误: {e}")
        except Exception as e:
            raise PDFOpenError(f"未知错误打开PDF: {pdf_path}, 错误: {e}")

        total_pages = len(doc)
        file_hash = ".."

        all_ocr_text: list = []
        all_ocr_results: list = []
        strategy = self.config.extraction_strategy
        strategy_labels = {
            "PaddleOCR + coordinate": "PaddleOCR + 坐标重建表格",
            "PaddleOCR": "PaddleOCR 纯文本",
            "original_text_layer": "原始文本层提取",
        }
        if strategy not in strategy_labels:
            raise ValueError(f"Unknown extraction strategy: {strategy}")
        logger.info("OCR 策略: %s", strategy_labels[strategy])

        try:
            for page_num in range(total_pages):
                page_display = page_num + 1
                logger.debug("处理第 %d/%d 页 OCR …", page_display, total_pages)
                t_page = time.time()
                # 逐页进度汇报：阶段开始
                if on_progress:
                    on_progress(
                        f"OCR 扫描中… {page_display}/{total_pages}",
                        f"开始处理第 {page_display}/{total_pages} 页 OCR",
                    )

                try:
                    page = doc[page_num]
                    page_extra: dict = {}

                    if strategy == "original_text_layer":
                        result = self._process_with_original_text_layer(page, page_num)
                        page_text = result["text"]
                        page_tables = result["tables"]
                        page_avg_confidence = result["avg_confidence"]

                    elif strategy == "PaddleOCR":
                        img = self._page_to_image(page)
                        result = self._process_with_PaddleOCR(img, page_num)
                        page_text = result["text"]
                        page_tables = result["tables"]
                        page_avg_confidence = result["avg_confidence"]
                        del img

                    elif strategy == "PaddleOCR + coordinate":
                        img = self._page_to_image(page)
                        result = self._process_with_PaddleOCR_coordinate(img, page_num)
                        page_text = result["markdown"]
                        page_tables = []
                        page_avg_confidence = result["avg_confidence"]
                        page_extra = {
                            "corrections": result.get("corrections", 0),
                            "entity_corrections": result.get("entity_corrections", 0),
                        }
                        del img

                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")

                    all_ocr_text.append(page_text)
                    all_ocr_results.append({
                        "page": page_display,
                        "text": page_text,
                        "tables": page_tables,
                        "avg_confidence": page_avg_confidence,
                        **page_extra,
                    })
                    # 记录 OCR 原始结果到 debug data log
                    write_debug_data(
                        self._debug_logger,
                        f"第 {page_display}/{total_pages} 页 OCR 完成 "
                        f"({len(page_text)} 字符, 策略: {strategy})",
                    )
                    write_debug_data_lines(
                        self._debug_logger,
                        [page_text or "(空)"],
                    )
                    # 逐页进度汇报：阶段结束
                    if on_progress:
                        page_elapsed = round(time.time() - t_page, 1)
                        on_progress(
                            None,
                            f"第 {page_display}/{total_pages} 页 OCR 完成，获得 {len(page_text)} 字符，耗时 {page_elapsed}s",
                        )

                except Exception as e:
                    logger.error("第 %d 页 OCR 处理失败: %s", page_display, e)
                    import traceback
                    traceback.print_exc()
                    all_ocr_results.append({
                        "page": page_display,
                        "text": "",
                        "tables": [],
                        "error": str(e),
                        "avg_confidence": 0,
                    })
                    # 记录失败情况到 debug data log
                    write_debug_data(
                        self._debug_logger,
                        f"第 {page_display}/{total_pages} 页 OCR 失败: {e}",
                    )
                    # 逐页进度汇报：失败但仍结束
                    if on_progress:
                        page_elapsed = round(time.time() - t_page, 1)
                        on_progress(
                            None,
                            f"第 {page_display}/{total_pages} 页 OCR 失败，耗时 {page_elapsed}s: {e}",
                        )

                finally:
                    page = None
                    gc.collect()

        finally:
            doc.close()

        full_text = "\n\n".join(all_ocr_text)
        return {
            "text": full_text,
            "pages": all_ocr_results,
            "source": strategy,
            "file_hash": file_hash,
            "total_pages": total_pages,
            "processed_pages": len(all_ocr_results),
            "has_errors": any("error" in p for p in all_ocr_results),
        }

    def _page_to_image(self, page) -> np.ndarray:
        """将 PDF 页面转为 BGR 格式图片（OpenCV 兼容）。

        强制拷贝像素数据后立即释放 fitz Pixmap，避免 C 层内存积累。
        """
        pix = page.get_pixmap(dpi=self.config.ocr_dpi)
        samples = bytes(pix.samples)
        h, w, n = pix.height, pix.width, pix.n
        pix = None  # 释放 fitz Pixmap（C 层内存不经 Python GC）

        img = np.frombuffer(samples, dtype=np.uint8).reshape(h, w, n)
        del samples

        if n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img
    
    def try_process(
        self,
        pdf_path: str,
        on_progress: Optional[Callable] = None,
    ) -> List[str]:
        """
        安全的处理方式，返回每页的 markdown 文本列表。

        某页处理失败时，对应位置为空字符串 ""，保持页码索引不偏移。
        PDF 无法打开时返回空列表。

        适合批量处理，不抛出异常。
        """
        # ── OCR 缓存：跳过 PaddleOCR，从预生成 .md 文件直接加载 ──
        if DebugConfig.use_ocr_cache:
            import glob
            import re
            cache_dir = DebugConfig.ocr_cache_dir
            if not os.path.isdir(cache_dir):
                logger.warning(
                    "OCR 缓存路径不存在，回退到正常 OCR（路径: %s）", cache_dir,
                )
            else:
                subdirs = [
                    d for d in os.listdir(cache_dir)
                    if os.path.isdir(os.path.join(cache_dir, d))
                ]
                if not subdirs:
                    logger.warning(
                        "OCR 缓存路径下无时间戳子目录，回退到正常 OCR（路径: %s）",
                        cache_dir,
                    )
                else:
                    subdirs.sort(reverse=True)
                    latest_dir = os.path.join(cache_dir, subdirs[0])
                    md_files = glob.glob(os.path.join(latest_dir, "*.md"))
                    if not md_files:
                        logger.warning(
                            "OCR 缓存路径下无有效 .md 文件，回退到正常 OCR（路径: %s）",
                            latest_dir,
                        )
                    else:
                        md_files.sort(key=lambda f: int(
                            re.search(r"(\d+)", os.path.basename(f)).group(1)
                        ))
                        logger.info(
                            "命中 OCR 缓存，跳过 PaddleOCR（%s，共 %d 页）",
                            subdirs[0], len(md_files),
                        )
                        pages = [""]
                        for f in md_files:
                            with open(f, "r", encoding="utf-8") as fh:
                                content = fh.read()
                            pages.append(content)
                        if on_progress:
                            on_progress(
                                None,
                                f"从 OCR 缓存加载了 {len(md_files)} 页"
                                f" ({subdirs[0]})",
                            )
                        return pages

            # 开启了缓存但未能找到缓存路径/数据 —— 统一向前端汇报，细节已由 logger 记录
            if on_progress:
                on_progress(None, "未能找到 OCR 缓存数据，回退到正常 OCR 流程")
        # ── OCR 缓存结束 ──

        try:
            result = self.process(pdf_path, on_progress=on_progress)
        except PDFOpenError as e:
            logger.error("PDF 打开失败，无法 OCR: %s", e)
            return []

        pages = result.get("pages", [])
        # 检查 OCR 失败的页面并警告
        failed = [p for p in pages if "error" in p]
        if failed:
            failed_nums = [str(p["page"]) for p in failed]
            logger.warning(
                "以下页面 OCR 提取失败，内容为空：第 %s 页",
                ", ".join(failed_nums),
            )
        # 1-based 索引：pages[0] 为占位空串，pages[1] 对应第1页
        pages_list = [""] + [page.get("text", "") for page in pages]
        return pages_list