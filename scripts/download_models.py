"""预下载所有模型，避免首次运行时卡在下载环节。

用法：
    python download_models.py

下载目标路径由 config.py 的 huggingface_cache_dir 控制。
PaddleOCR 模型路径不可配（见 config.py 中的注释），
如需自定义请使用 Windows mklink /J 创建目录联结。
"""

import os

from config import Config
from dotenv import load_dotenv

load_dotenv()

_HF_MODELS = [
    ("BAAI/bge-m3", "BGE-M3 嵌入模型", "约 2GB"),
    ("BAAI/bge-reranker-base", "BGE-Reranker 二次排序模型", "约 1GB"),
]


def _check_huggingface_model_cached(model_name: str) -> bool:
    """检查指定 HuggingFace 模型是否已缓存。"""
    hub_dir = os.environ.get("HF_HOME", "")
    if not hub_dir:
        hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    model_dir = os.path.join(
        hub_dir, "hub", f"models--{model_name.replace('/', '--')}"
    )
    return os.path.isdir(model_dir)


def download_huggingface_models():
    """下载 BGE-M3 和 BGE-Reranker，已缓存则跳过。"""
    for model_id, name, size in _HF_MODELS:
        if _check_huggingface_model_cached(model_id):
            print(f"[skip] {name} 已缓存")
            continue

        print(f"下载 {name}（{size}）...")
        if "reranker" in model_id:
            from sentence_transformers import CrossEncoder
            CrossEncoder(model_id)
        else:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(model_id)
        print(f"  {name} 就绪")


def download_paddle_models():
    """预加载 PaddleOCR 模型（OCR 引擎初始化时自动触发）。"""
    print("下载 PaddleOCR 模型（约 200MB）...")
    from services.indexing.ocr_engine import SecurePDFProcessor
    SecurePDFProcessor()
    print("  PaddleOCR 模型就绪")


if __name__ == "__main__":
    cfg = Config()
    os.environ["HF_HOME"] = cfg.huggingface_cache_dir

    print("=" * 50)
    print("预下载模型")
    print(f"  HuggingFace → {cfg.huggingface_cache_dir or '默认 (C 盘)'}")
    print(f"  PaddleX     → 不可配 (见 config.py 注释)")
    print("=" * 50)

    download_huggingface_models()
    download_paddle_models()

    print()
    print("全部模型就绪，可以运行 demo_search.py 了")
