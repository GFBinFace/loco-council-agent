"""本地模型就绪检查。

主程序启动时的准入检查：确认预装脚本已把模型放到配置的路径下。
本模块只做检查，不做展示、不做下载——各入口（Streamlit / CLI）自行
决定如何呈现结果。

模型路径由 .env 提供的环境变量决定：

    HF_HOME               HuggingFace 族模型（BGE-M3、BGE-Reranker）
    PADDLE_PDX_CACHE_HOME PaddleX 模型（PP-OCRv5）

两个变量都未设置时会回退到各自的默认路径（均在 C 盘用户目录下），
此时检查仍然有效，只是路径不是用户配置的那一个。
"""

import os
from dataclasses import dataclass
from typing import List

from utils import get_file_logger

logger = get_file_logger(__file__)

# 预装模型清单——检查与下载共用同一份定义，避免两处维护后漂移。
# 这里只列"不装就跑不起来"的核心模型：BGE-M3 与 BGE-Reranker 在
# RAGPipeline 构造时同步加载；PP-OCRv5 的检测+识别模型是 OCR 的必需项。
# 其余 PaddleX 模型（版面分析、方向分类等）由 PaddleX 按需加载，
# 缺失不影响启动，故不纳入。
HF_MODELS = ("BAAI/bge-m3", "BAAI/bge-reranker-base")
PADDLE_MODELS = ("PP-OCRv5_server_det", "PP-OCRv5_server_rec")

# 预装是强制前置步骤：模型体积大（数 GB），必须让用户明确知情后再执行
PREINSTALL_HINT = (
    "本地模型未就绪。请先运行一次性预装脚本完成下载：\n"
    "\n"
    "    python scripts/download_models.py\n"
    "\n"
    "该脚本会下载约 3.2GB 模型文件，请确认目标磁盘空间与流量充足。\n"
    "下载目标由 .env 中的 HF_HOME / PADDLE_PDX_CACHE_HOME 指定。"
)


class LocalModelsNotReadyError(RuntimeError):
    """本地模型未就绪——主程序拒绝启动，提示用户执行预装脚本。"""


@dataclass
class ModelCheckItem:
    """单个模型的就绪状态。

    Attributes:
        name: 模型名（HuggingFace 为 repo id，PaddleX 为模型目录名）。
        category: 模型族，取值 "HuggingFace" 或 "PaddleX"。
        path: 检查时实际查找的路径。
        ready: 该模型是否已就位。
    """

    name: str
    category: str
    path: str
    ready: bool


@dataclass
class ModelReadinessReport:
    """本地模型就绪检查的完整结果。

    Attributes:
        items: 全部被检查项的明细。
        hf_home: 本次检查使用的 HuggingFace 缓存根目录。
        paddlex_home: 本次检查使用的 PaddleX 缓存根目录。
        hf_home_from_env: HF_HOME 是否来自环境变量（False 表示回退到默认路径）。
        paddlex_home_from_env: PADDLE_PDX_CACHE_HOME 是否来自环境变量。
    """

    items: List[ModelCheckItem]
    hf_home: str
    paddlex_home: str
    hf_home_from_env: bool
    paddlex_home_from_env: bool

    @property
    def ready(self) -> bool:
        """全部检查项是否都已就位。"""
        return all(item.ready for item in self.items)

    @property
    def missing(self) -> List[ModelCheckItem]:
        """未就位的检查项。"""
        return [item for item in self.items if not item.ready]


def _resolve_hf_home() -> tuple[str, bool]:
    """解析 HuggingFace 缓存根目录。

    回退值与 huggingface_hub 自身的默认值保持一致（~/.cache/huggingface），
    否则会出现"检查说没就位、但库其实能找到"的误报。

    Returns:
        (缓存根目录, 是否来自环境变量)
    """
    from_env = os.environ.get("HF_HOME", "").strip()
    if from_env:
        return from_env, True
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), False


def _resolve_paddlex_home() -> tuple[str, bool]:
    """解析 PaddleX 缓存根目录。

    回退值与 paddlex.utils.cache 的默认值保持一致（~/.paddlex）。
    这里按环境变量自行推算而不导入 paddlex：该模块会在 import 时
    确定缓存路径，导入它会把这份耦合带进来，而这里只需要同一个公式。

    Returns:
        (缓存根目录, 是否来自环境变量)
    """
    from_env = os.environ.get("PADDLE_PDX_CACHE_HOME", "").strip()
    if from_env:
        return from_env, True
    return os.path.join(os.path.expanduser("~"), ".paddlex"), False


def _check_hf_models(hf_home: str) -> List[ModelCheckItem]:
    """检查 HuggingFace 族模型是否已落盘。

    huggingface_hub 的目录结构为 {HF_HOME}/hub/models--{org}--{name}，
    只要该目录存在即视为就位——与预装脚本的判断口径一致。

    Args:
        hf_home: HuggingFace 缓存根目录。

    Returns:
        每个模型一条的检查结果。
    """
    items: List[ModelCheckItem] = []
    for model_id in HF_MODELS:
        model_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
        items.append(
            ModelCheckItem(
                name=model_id,
                category="HuggingFace",
                path=model_dir,
                ready=os.path.isdir(model_dir),
            )
        )
    return items


def _check_paddle_models(paddlex_home: str) -> List[ModelCheckItem]:
    """检查 PaddleX 模型是否已落盘。

    PaddleX 的目录结构为 {CACHE_DIR}/official_models/{模型名}，
    见 paddlex/inference/utils/official_models.py 的 _save_dir 定义。

    Args:
        paddlex_home: PaddleX 缓存根目录。

    Returns:
        每个模型一条的检查结果。
    """
    items: List[ModelCheckItem] = []
    for model_name in PADDLE_MODELS:
        model_dir = os.path.join(paddlex_home, "official_models", model_name)
        items.append(
            ModelCheckItem(
                name=model_name,
                category="PaddleX",
                path=model_dir,
                ready=os.path.isdir(model_dir),
            )
        )
    return items


def ensure_cache_dirs() -> None:
    """备好模型缓存根目录，并清除空值环境变量。

    必须在导入 PaddleX 之前调用，原因有二：

    1. PaddleX 在 import 时就往缓存目录下写 temp/ 等子目录，路径不存在会
       直接抛 FileNotFoundError——那时还没轮到本模块的就绪检查。
    2. 环境变量被设为空字符串时，huggingface_hub 会据此算出相对路径 `hub`，
       把数 GB 模型下载进当前工作目录。空值一律清除，退回各自默认路径。

    Raises:
        OSError: 缓存目录无法创建（典型原因：配置的盘符不存在）。
    """
    for var_name in ("HF_HOME", "PADDLE_PDX_CACHE_HOME"):
        value = os.environ.get(var_name)
        if value is None:
            continue
        if not value.strip():
            del os.environ[var_name]
            logger.warning("%s 为空值，已清除并回退到默认路径", var_name)
            continue
        try:
            os.makedirs(value, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"{var_name} 指向的目录无法创建：{value}\n"
                f"原因：{exc.strerror}\n"
                "请检查 .env 中该路径所在的分区是否存在。"
            ) from None


def check_local_models() -> ModelReadinessReport:
    """检查预装脚本所需的本地模型是否全部就位。

    纯读取，无副作用，可反复调用。

    Returns:
        包含全部检查项明细、以及本次使用的缓存根目录的检查报告。
    """
    hf_home, hf_from_env = _resolve_hf_home()
    paddlex_home, paddlex_from_env = _resolve_paddlex_home()
    items = _check_hf_models(hf_home) + _check_paddle_models(paddlex_home)
    report = ModelReadinessReport(
        items=items,
        hf_home=hf_home,
        paddlex_home=paddlex_home,
        hf_home_from_env=hf_from_env,
        paddlex_home_from_env=paddlex_from_env,
    )
    logger.info(
        "本地模型就绪检查：%d/%d 就位（HF_HOME=%s, PADDLE_PDX_CACHE_HOME=%s）",
        len(items) - len(report.missing), len(items), hf_home, paddlex_home,
    )
    return report


def _format_missing_detail(report: ModelReadinessReport) -> str:
    """把未就位的模型整理成可直接展示的多行文本。

    Args:
        report: 检查报告。

    Returns:
        每行一个缺失模型及其查找路径。
    """
    lines = []
    for item in report.missing:
        lines.append(f"  [缺失] {item.name} ({item.category})")
        lines.append(f"         查找路径：{item.path}")
    if not report.hf_home_from_env:
        lines.append("  提示：未设置 HF_HOME，当前使用默认路径（C 盘用户目录）")
    if not report.paddlex_home_from_env:
        lines.append("  提示：未设置 PADDLE_PDX_CACHE_HOME，当前使用默认路径（C 盘用户目录）")
    return "\n".join(lines)


def ensure_local_models() -> None:
    """确认本地模型全部就位，未就位则拒绝继续。

    供启动路径调用，在加载任何模型之前执行——PaddleOCR 与 BGE 系列
    在缺失时会抛出难以定位的库异常（如"连不上某个 endpoint"），
    这里提前拦截并给出唯一明确的下一步动作。

    Raises:
        LocalModelsNotReadyError: 存在未就位的模型。
    """
    report = check_local_models()
    if report.ready:
        return
    raise LocalModelsNotReadyError(
        f"{PREINSTALL_HINT}\n\n缺失明细：\n{_format_missing_detail(report)}"
    )
