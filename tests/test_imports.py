#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证所有依赖是否能正常导入"""

import sys

def test_import(module_name, import_name=None):
    """测试导入模块"""
    try:
        if import_name:
            exec(f"from {module_name} import {import_name}")
        else:
            __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    print(f"Python 版本: {sys.version}")
    print("-" * 50)
    
    # OCR相关
    test_import("paddle")
    test_import("paddleocr", "PaddleOCR")
    test_import("fitz")  # pymupdf
    
    # 图像处理
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
    
    # 向量和检索
    test_import("lancedb")
    test_import("sentence_transformers", "SentenceTransformer")
    test_import("transformers")
    import torch
    print(f"✅ torch: {torch.__version__}")
    test_import("huggingface_hub")
    
    # 工具
    import pandas as pd
    print(f"✅ pandas: {pd.__version__}")
    from pydantic import BaseModel
    print("✅ pydantic")
    from tqdm import tqdm
    print("✅ tqdm")
    
    print("-" * 50)
    print("🎉 所有依赖检查通过！")

if __name__ == "__main__":
    main()