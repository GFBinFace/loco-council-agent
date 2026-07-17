from dataclasses import dataclass
from typing import Literal

@dataclass
class Config:
    # 路径配置
    lance_db_dir: str = "./data/lancedb"
    # sqlite_dir 下存储：
    #   - docs.sqlite     : DocManager 文档元数据（files + chunks）
    #   - history.sqlite  : HistoryStore 查询历史（history_sessions）
    sqlite_dir: str = "./data/sqlite"
    # PDF 文件大小上限（MB）。需与 .streamlit/config.toml 的 maxUploadSize 保持一致；
    # 若仅调整后端限制，需同步修改 Streamlit 配置文件的对应值。
    max_pdf_size_mb: int = 200

    # ── OCR配置 ──────────────────────────────────────────
    ocr_dpi: int = 300
    ocr_lang: str = "ch"  # 'ch', 'en'
    enable_preprocessing: bool = True
    # 'PaddleOCR + coordinate', 'PaddleOCR', 'original_text_layer' (默认PaddleOCR + coordinate)
    #   只有“PaddleOCR + coordinate”达到了可用的 OCR 效果。留着其他选项仅为对比。
    extraction_strategy: str = "PaddleOCR + coordinate"  

    # ── 切块配置 ──────────────────────────────────────────
    chunk_soft_max: int = 3000       # Prompt 约束值（LLM 切块时的字符上限，写入 system prompt）
    chunk_hard_max: int = 5000       # 代码强制切分阈值（超出则按规则切分）
    txt_chapter_boundary: int = 100_000  # TXT 短篇/中长篇分界（字符数），用于章节 regex 验证
    # LLM 归纳的章节 regex 验证门槛：每 N 字符至少 1 个标题（宽松验证）。
    txt_chapter_llm_min_density: int = 10000
    # 内置章节规则的验证门槛：每 N 字符至少 1 个标题才算真实可用。
    # 严于 LLM 层的宽松验证——规则是盲猜家族，宁可漏判交给 LLM，
    # 不可误判把正文切碎。
    txt_chapter_rule_min_density: int = 6000
    pages_per_batch: int = 3          # 每批发送给 LLM 的页数
    chapter_map_mode: str = "off"     # "off" | "rule_only" | "rule_then_llm" | "llm_only"
    fetch_page_limit: int = 10        # fetch_page 工具最大调用次数

    # 切块 LLM
    chunking_model: str = "deepseek-chat"
    chunking_api_key: str = ""             # 从环境变量 DEEPSEEK_API_KEY 读取
    chunking_base_url: str = "https://api.deepseek.com"
    chunking_max_retries: int = 2
    chunking_retry_base_delay: float = 3.0  # 秒，实际延迟 = base * (2 ** attempt)

    # 行覆盖校验配置
    coverage_max_batch_retries: int = 2       # 整批驳回最多重试次数
    coverage_direct_fix_threshold: int = 2    # 漏行 ≤ 此数时跳过整批驳回，直接专项修补

    # ── 向量配置 ──────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024  # BGE-M3 输出 1024 维，max_seq_length=8192 tokens
    # 索引时的分批嵌入批大小。批与批之间汇报进度（UI 状态行 + 文件日志），
    # 避免大文档单次 encode 长时间静默、事后无法定位卡点。
    # 注意：BGE-M3 编码长文本时 attention 矩阵 O(n²)，批中每条文本可能数千
    # 个 token，过大批大小会导致峰值内存超物理限制（曾致死机）。
    # 64 → 16：纯 CPU 机器安全值，有 GPU 可适当调大以缩短总耗时。
    #   16G内存,无GPU的机器建议这个配置使用16，设置成64可能会导致内存溢出。
    embedding_batch_size: int = 16

    # ── 模型缓存路径 ──────────────────────────────────────
    # HuggingFace 族模型的缓存目录（BGE-M3、BGE-Reranker 等）。
    # 空字符串 = 使用默认路径（Windows: %USERPROFILE%\.cache\huggingface）。
    # 设值后写入 HF_HOME 环境变量，首次加载模型时自动下载到该目录。
    #
    # 注意：PaddleOCR / PaddleX 模型不支持自定义路径（PaddleX 3.0 硬编码
    # ~/.paddlex，不读取环境变量）。如需搬迁，请在 Windows 命令行执行：
    #     mklink /J %USERPROFILE%\.paddlex 目标路径
    # 这会创建一个 NTFS 目录联结（Junction），对程序透明——
    # 程序以为写入 C 盘，实际文件存储在目标路径。
    huggingface_cache_dir: str = ""  # 空=使用默认路径，设值后写入 HF_HOME 环境变量

    # ── 检索配置 ──────────────────────────────────────────
    hybrid_search_top_k: int = 42        # 混合检索返回的候选数量上限
    crossencoder_top_k: int = 15         # CrossEncoder二次排序收窄后数量上限

    # CrossEncoder 模型
    rerank_model: str = "BAAI/bge-reranker-base"

    # LLM 打分后的分级收网策略：
    # 该配置是分级收网策略的阈值，约束：
    #   - 每个值必须在 0-10 之间
    #   - 必须严格从高到低递减
    #   - 最后一个值为截止分数，不会再考虑更低得分的数据块
    rerank_score_tiers: tuple = (8, 7, 6, 5)

    # ── 分块间隙填充配置 ──────────────────────────────────────
    # 策略一：补填入围 chunk 之间的连续缺失块。0=关闭，1=只填单个间隙（旧版行为），
    # N=补齐至多 N 个连续空缺（catch-up 跨章节长叙事）。
    gap_fill_width: int = 5
    # 策略二：将每个入围 chunk 的邻居也拉入候选池。0=关闭，1=±1，2=±2。
    # gap 填充先创建语义连续块，邻居扩展再在块边缘外扩，不会引入远距离不相干 chunk。
    neighbor_extension_width: int = 1
    # 两种策略引入的额外 chunk 总数上限 = 原始候选数 × ratio。
    # 例：原始 6 个候选，ratio=1.0 → 最多额外引入 6 个 chunk。
    gap_fill_ratio: float = 1.2

    # ── 上下文容量保护 ──────────────────────────────────────

    # 间隙+邻居扩展额外引入的 token 硬上限
    gap_fill_token_limit: int = 20000

    # 送入 LLM 的 RAG 上下文总 token 硬上限（超出部分从中间整 chunk 切除）
    llm_context_rag_token_limit: int = 60000

    # 附带历史上下文时仅保留最近 N 轮
    history_turn_limit: int = 5

    # Rerank LLM
    rerank_llm_model: str = "deepseek-chat"
    rerank_llm_api_key: str = ""             # 从环境变量 DEEPSEEK_API_KEY 读取
    rerank_llm_base_url: str = "https://api.deepseek.com"

    # Answer LLM（最终回答）
    answer_model: str = "deepseek-chat"
    answer_api_key: str = ""                 # 从环境变量 DEEPSEEK_API_KEY 读取
    answer_base_url: str = "https://api.deepseek.com"

    # ── 操作历史配置 ──────────────────────────────────────
    # 滚动上限：operation_history 表最多保留的记录数，超出时自动删除最早记录。
    operation_history_max_rows: int = 10000

    def __post_init__(self):
        """校验配置参数合法性"""
        tiers = self.rerank_score_tiers
        if not all(0 <= v <= 10 for v in tiers):
            raise ValueError(
                f"rerank_score_tiers 每个值必须在 0-10 之间，当前值: {tiers}"
            )
        if not all(tiers[i] > tiers[i + 1] for i in range(len(tiers) - 1)):
            raise ValueError(
                f"rerank_score_tiers 必须严格从高到低递减，当前值: {tiers}"
            )
        if self.operation_history_max_rows < 1:
            raise ValueError(
                f"operation_history_max_rows 必须 ≥ 1，当前值: "
                f"{self.operation_history_max_rows}"
            )
        if not isinstance(self.gap_fill_width, int) or self.gap_fill_width < 0:
            raise ValueError(
                f"gap_fill_width 必须是非负整数，当前值: {self.gap_fill_width}"
            )
        if (
            not isinstance(self.neighbor_extension_width, int)
            or self.neighbor_extension_width < 0
        ):
            raise ValueError(
                "neighbor_extension_width 必须是非负整数，当前值: "
                f"{self.neighbor_extension_width}"
            )

# ============================================================
# 调试功能配置
# ============================================================

class DebugConfig:
    """调试功能 — 所有开关默认关闭。调试数据输出到项目根 temp/ 下。"""

    # ── D1: OCR 缓存 ───────────────────────────────────────
    # 开启后跳过 PaddleOCR，从 temp/ 下预生成的 .md 缓存文件直接加载每页结果。
    # 缓存由 scripts/demo_index.py 中的 test_PaddleOCR_coordinate_ocr 函数生成。
    # 仅在 extraction_strategy == "PaddleOCR + coordinate" 时有意义。
    use_ocr_cache: bool = False
    ocr_cache_dir: str = "./temp/test_PaddleOCR_coordinate_ocr"

# ============================================================
# 日志策略配置
# ============================================================

class LogConfig:
    """日志策略 — 统一管理各模块文件日志的行为参数"""

    # 单个日志文件最大字节容量，默认 10MB
    MAX_BYTES: int = 10 * 1024 * 1024  # 10_485_760

    # 日志输出目录（相对于项目根目录）。业务日志和 debug data log 均输出到此目录。
    LOG_DIR: str = "logs"

    # 记录各环节具体数据的单独日志。
    # "off" | "overwrite" | "append"
    #   "overwrite"=每次运行覆盖旧日志，"append"=追加到旧日志（便于对比多次运行）
    # 日志以 模块名_data.log 命名，如 logs/financial_chunker_data.log
    DEBUG_DATA_MODE: str = "append"

# ============================================================
# Streamlit 前端配置
# ============================================================

class StreamlitConfig:
    """Streamlit 前端专属配置。

    前端做薄做轻：此类隔离前端关切，便于未来多前端并存。
    """

    # 操作历史弹窗单次显示条数（默认展示最新 N 条）。
    # 约束：应 ≤ Config.operation_history_max_rows。
    operation_history_page_size: int = 50
