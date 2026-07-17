from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from config import Config

class Embedder:
    """文本向量化"""
    
    def __init__(self, config: Config = Config()):
        # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 已在 pipeline.__init__ 中强制设置
        self.model = SentenceTransformer(
            config.embedding_model, local_files_only=True,
        )
        self.dim = config.embedding_dim
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将文本转换为向量。

        返回形状取决于输入类型：
        - str  → (dim,)  1D 向量
        - list[0] → (n, dim)  2D 矩阵，每行为一条文本的向量
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # 归一化，便于余弦相似度
            show_progress_bar=False,
        )

        if single:
            return embeddings[0]  # (dim,)
        return embeddings          # (n, dim)
    
