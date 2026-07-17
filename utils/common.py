"""
项目通用工具函数。
"""

import hashlib
import os

from utils.logging import get_file_logger

logger = get_file_logger(__file__)


def read_text_file(file_path: str) -> str:
    """
    读取文本文件内容，自动探测编码。

    依次尝试 UTF-8、GBK、GB18030，均失败则用 UTF-8 替换模式兜底。
    Windows 上中文 TXT 文件普遍是 GBK 编码，仅假设 UTF-8 会导致
    UnicodeDecodeError（如字节 0xa1）。

    限制：试解码法对 Big5 等编码可能"成功"解出乱码而非报错。
    本项目面向大陆中文语境（UTF-8/GBK/GB18030 已覆盖），不做全量编码嗅探。

    Args:
        file_path: 文件路径

    Returns:
        解码后的文本内容
    """
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            logger.info(
                "文本文件编码探测: %s → %s (%d 字符)",
                os.path.basename(file_path), encoding, len(text),
            )
            return text
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 所有编码均失败，用替换模式兜底——不丢数据，但不可解码字节会被 � 替换
    logger.warning(
        "文本文件编码探测失败，UTF-8 替换模式兜底读取（不可解码字节将变为 �）: %s",
        os.path.basename(file_path),
    )
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def compute_file_md5(file_path: str) -> str:
    """计算文件的 MD5 哈希"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def tokens_to_chars(token_count: int, ratio: float = 0.6) -> int:
    """
    token 数 → 字符数的保守估算。

    DeepSeek tokenizer 对中文字符的分词效率约为每 token 0.5–0.7 字符，
    取 0.6 为保守默认值。用于硬上限保护的截断判断——不是精确 tokenization。

    Args:
        token_count: token 数量
        ratio: 每 token 的平均字符数，默认 0.6

    Returns:
        等效字符数上限
    """
    return int(token_count / ratio)


def extract_doc_name(pdf_path: str) -> str:
    """从 PDF 文件路径提取文档名（不含扩展名）"""
    basename = os.path.basename(pdf_path)
    if basename.lower().endswith('.pdf'):
        return basename[:-4]
    return basename
