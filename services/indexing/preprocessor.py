import cv2
import numpy as np

from utils import get_file_logger
logger = get_file_logger(__file__)


class ImagePreprocessor:
    """扫描件图像预处理，提升OCR准确率"""
    
    @staticmethod
    def preprocess_for_ocr(
        image: np.ndarray,
        denoise: bool = True,
        deskew: bool = True,
        sharpen: bool = True
    ) -> np.ndarray:
        """
        完整的预处理流水线
        
        Args:
            image: 输入图像，支持 BGR 格式（OpenCV 默认）
            denoise: 是否降噪
            deskew: 是否倾斜校正
            sharpen: 是否锐化
        
        Returns:
            预处理后的二值图像
        """
        # 1. 灰度化（确保输入是 BGR 格式）
        if len(image.shape) == 3:
            # 假设输入是 BGR（OpenCV 默认）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. 降噪（可选）
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 3. 倾斜校正（可选）
        if deskew:
            gray = ImagePreprocessor._deskew(gray)
        
        # 4. CLAHE 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 5. 自适应二值化
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 2
        )
        
        # 6. 形态学操作（连接断裂笔画）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 7. 锐化（可选）
        if sharpen:
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
            binary = cv2.filter2D(binary, -1, kernel_sharpen)
        
        return binary
    
    @staticmethod
    def preprocess_for_paddleocr(image: np.ndarray) -> np.ndarray:
        """
        为 PaddleOCR 优化的轻量预处理，包含：
        - 裁切空白（控制内存峰值）
        - 降噪（先于增强，避免放大噪点）
        - 轻度锐化
        - CLAHE 对比度增强

        Args:
            image: 输入图像，BGR 格式

        Returns:
            预处理后的 BGR 图像（保持彩色格式）
        """
        if image is None or image.size == 0:
            return image

        # 0. 裁切空白：控制 PaddleOCR 识别阶段的内存峰值
        processed = ImagePreprocessor.crop_blank_margins(image)

        # 1. 降噪（先于锐化和 CLAHE，避免后续步骤放大扫描噪点）
        processed = cv2.fastNlMeansDenoisingColored(
            processed, None, 3, 3, 7, 21,
        )

        # 2. 轻度锐化：kernel center=5（原为 9，太重导致竖向表头光晕粘连）
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel_sharpen)

        # 3. CLAHE 对比度增强：
        #    tile 从 8×8 扩大到 16×16，减少块状伪影；
        #    clipLimit 从 2.5 降到 2.0，减少局部过增强。
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

        return processed
    
    @staticmethod
    def crop_blank_margins(
        image: np.ndarray,
        margin: int = 20,
        min_content_ratio: float = 0.005,
    ) -> np.ndarray:
        """裁掉页面上下的大片空白区域。

        水平投影检测内容边界，适用于扫描 PDF 中表格仅占上半页、
        下半截为空白的常见场景（如"承上页"的半页表格）。

        使用 Otsu 自适应阈值，自动适应不同扫描件的纸色。

        Args:
            image: 输入图像（BGR / RGB / 灰度均可）
            margin: 裁切后在内容边界外保留的像素边距
            min_content_ratio: 判定为「有内容行」的最小像素和占比

        Returns:
            裁切后的图像；若无空白区域则返回原图。
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊平滑扫描噪点，避免噪点干扰行投影检测
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Otsu 自适应阈值：自动根据像素分布找到文字/背景的最佳分割点
        otsu_thresh, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        row_sums = binary.sum(axis=1)
        if row_sums.max() == 0:
            logger.info("crop_blank_margins: 全空白页，跳过裁切")
            return image

        content_thresh = row_sums.max() * min_content_ratio
        content_rows = np.where(row_sums > content_thresh)[0]
        if len(content_rows) == 0:
            logger.info("crop_blank_margins: 无内容行，跳过裁切")
            return image

        # 将 content_rows 按间距拆成独立的"内容块"，过滤掉被大片空白
        # 隔开的边角小字（如右下角的文件名、页码），避免它们阻挡裁切。
        blocks = ImagePreprocessor._split_into_blocks(content_rows, image.shape[0])
        if len(blocks) > 1:
            kept = ImagePreprocessor._drop_marginal_blocks(blocks, image.shape[0])
            if kept:
                top = max(0, int(kept[0][0]) - margin)
                bottom = min(image.shape[0], int(kept[-1][1]) + margin)
        else:
            top = max(0, int(content_rows[0]) - margin)
            bottom = min(image.shape[0], int(content_rows[-1]) + margin)

        logger.debug(
            "crop_blank_margins: otsu_thresh=%d, max_row_sum=%.0f, "
            "content_thresh=%.0f, content_rows=%d, blocks=%d, "
            "top=%d, bottom=%d, orig_h=%d",
            otsu_thresh, row_sums.max(), content_thresh,
            len(content_rows), len(blocks), top, bottom, image.shape[0],
        )

        if top > 0 or bottom < image.shape[0]:
            logger.info(
                "裁切空白边距：原始高度 %d → %d（上边界 %d，下边界 %d）",
                image.shape[0], bottom - top, top, bottom,
            )
            return image[top:bottom, :]
        return image

    @staticmethod
    def _split_into_blocks(
        content_rows: np.ndarray,
        image_height: int,
        min_gap_ratio: float = 0.10,
    ) -> list:
        """将内容行按大片空白拆成独立的块。

        当连续内容行之间存在超过 min_gap_ratio 页面高度的空白时，
        在此处切分为不同的块。

        Args:
            content_rows: 被判定为"有内容"的行索引数组
            image_height: 图像总高度（像素）
            min_gap_ratio: 判定为"大片空白"的最小间隙占比

        Returns:
            [(start_row, end_row), ...] 每个元组是一个内容块的首尾行号
        """
        if len(content_rows) <= 1:
            return [(content_rows[0], content_rows[0])] if len(content_rows) == 1 else []

        gaps = np.diff(content_rows)
        min_gap_rows = int(image_height * min_gap_ratio)
        split_idx = np.where(gaps > min_gap_rows)[0]

        if len(split_idx) == 0:
            return [(content_rows[0], content_rows[-1])]

        blocks = []
        block_start = content_rows[0]
        for si in split_idx:
            blocks.append((block_start, content_rows[si]))
            block_start = content_rows[si + 1]
        blocks.append((block_start, content_rows[-1]))
        return blocks

    @staticmethod
    def _drop_marginal_blocks(
        blocks: list,
        image_height: int,
        min_block_ratio: float = 0.03,
    ) -> list:
        """丢弃高度过小的边角内容块（如页码、文件名标注）。

        Args:
            blocks: _split_into_blocks 的输出
            image_height: 图像总高度（像素）
            min_block_ratio: 低于此占比的块视为边角标注

        Returns:
            过滤后的内容块列表；若所有块都被判定为边角，则保留全部（不丢空）
        """
        min_block_rows = int(image_height * min_block_ratio)
        kept = [b for b in blocks if (b[1] - b[0]) >= min_block_rows]

        if not kept:
            # 所有块都很小（可能整页都只有零星标注），保留原样避免裁空
            logger.info(
                "crop_blank_margins: 所有内容块均小于阈值（%d 行），保留全部 %d 个块",
                min_block_rows, len(blocks),
            )
            return blocks

        if len(kept) < len(blocks):
            logger.info(
                "crop_blank_margins: 过滤掉 %d 个边角内容块（阈值 %d 行），保留 %d 个",
                len(blocks) - len(kept), min_block_rows, len(kept),
            )
        return kept

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """检测并校正图像倾斜"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 100:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) < 0.5:
            return image
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated