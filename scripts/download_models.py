"""本地模型预装脚本（一次性）。

把本项目运行所需的全部本地模型下载到 .env 指定的路径：

    HF_HOME                BGE-M3（嵌入）、BGE-Reranker（重排）
    PADDLE_PDX_CACHE_HOME  PP-OCRv5 检测与识别模型（OCR）

这是**第一次运行主程序之前的必需步骤**。模型体积大（约 3.2GB 文件加等量
下载流量），必须由用户明确知情后主动执行，而不是在启动时悄悄下载。主程序
启动时会检查模型是否就位（services/model_readiness.py），未就绪则暂停全部
功能并提示执行本脚本。

用法（在项目根目录下）：

    python scripts/download_models.py

执行前会列出将要做的动作、当前哪些模型已就位、以及预计开销，等你确认后
才开始；非交互环境（管道、CI）需显式加 `--yes` 跳过确认。

已下载的模型会跳过，可重复执行。结束时校验落盘位置——若模型没有出现在
配置的路径下（例如某个环境变量未生效），会明确报出来而不是静默成功。
"""

import argparse
import gc
import os
import sys

# 让脚本在 `python scripts/download_models.py` 调用下也能导入项目模块：
# 此时 sys.path[0] 是 scripts/ 而非项目根，直接 import config 会失败。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

# .env 必须在导入任何模型库之前加载——HF_HOME / PADDLE_PDX_CACHE_HOME
# 都在各自库 import 时被读取，之后再设置环境变量无效。
load_dotenv(override=True)

from config import Config
from services.model_readiness import (
    HF_MODELS,
    ModelReadinessReport,
    check_local_models,
    ensure_cache_dirs,
)

_HF_MODEL_LABELS = {
    "BAAI/bge-m3": "BGE-M3 嵌入模型",
    "BAAI/bge-reranker-base": "BGE-Reranker 二次排序模型",
}

# 仅供确认前估算开销展示，不参与任何逻辑判断
_MODEL_SIZES = {
    "BAAI/bge-m3": "约 2GB",
    "BAAI/bge-reranker-base": "约 1GB",
    "PP-OCRv5_server_det": "约 100MB",
    "PP-OCRv5_server_rec": "约 100MB",
}


def _resolve_or_warn(var_name: str, default_path: str) -> str:
    """读取模型缓存路径，未配置时给出醒目警告。

    留空会回退到 C 盘用户目录——数 GB 文件落在系统盘往往不是用户本意，
    所以这里必须显式提醒，而不是默默接受默认值。

    Args:
        var_name: 环境变量名。
        default_path: 该变量未设置时的回退路径。

    Returns:
        实际使用的缓存根目录。
    """
    value = os.environ.get(var_name, "").strip()
    if value:
        return value
    print(f"  ⚠️  未设置 {var_name}")
    print(f"      将使用默认路径：{default_path}")
    print("      数 GB 模型文件会占用该分区空间，建议在 .env 中改到其他盘")
    return default_path


def _print_plan(
    hf_home: str, paddlex_home: str, report: ModelReadinessReport
) -> None:
    """打印将要执行的动作与开销，供用户在确认前评估。

    Args:
        hf_home: HuggingFace 缓存根目录。
        paddlex_home: PaddleX 缓存根目录。
        report: 当前就绪检查结果，用于区分"需下载"与"已存在"。
    """
    pending = report.missing
    ready_count = len(report.items) - len(pending)
    print("=" * 62)
    print("本地模型预装")
    print("=" * 62)
    print()
    print("将执行以下动作：")
    print("  1. 校验 PaddleX 缓存路径是否按配置生效")
    print(f"  2. 下载 HuggingFace 模型 → {hf_home}")
    print(f"  3. 下载 PaddleOCR 模型  → {paddlex_home}")
    print("  4. 校验全部模型是否落到配置路径")
    print()
    print(f"当前状态：{len(report.items)} 个模型中 {ready_count} 个已就位")
    for item in pending:
        print(f"  [需下载] {item.name}（{_MODEL_SIZES.get(item.name, '大小未知')}）")
    for item in report.items:
        if item.ready:
            print(f"  [已存在] {item.name}")
    print()
    if pending:
        print("预计开销：已存在的模型不会重复下载；")
        print("          全部模型合计约需磁盘与下载流量各 3.2GB，视网络可能耗时较久")
    else:
        print("预计开销：无需下载，仅做加载与校验（仍需数十秒）")
    print()
    print("内存说明：触发下载需要构造模型，会临时占用 1~2GB 内存。")
    print("          脚本逐个处理、用完即释放，峰值不累加；退出后内存全部归还。")
    print("          这与主程序无关——主程序如何加载模型是另一件事。")
    print("-" * 62)
    print()


def _confirm(skip: bool) -> bool:
    """请用户确认后再开始，给出退出的机会。

    预装会占用数 GB 磁盘与流量，必须先让用户明确知情。非交互环境（管道、
    CI 等）无法询问，此时要求显式传入 --yes——避免"未经确认就静默下载"，
    这与本脚本"必须由用户主动执行"的定位相悖。

    Args:
        skip: 用户已通过 --yes 显式跳过确认。

    Returns:
        是否继续执行。
    """
    if skip:
        print("（已通过 --yes 跳过确认）")
        return True
    if not sys.stdin.isatty():
        print("非交互环境无法确认，如需继续请加 --yes 参数。")
        return False
    try:
        answer = input("确认开始？[y/N] ")
    except EOFError:
        print()
        print("未收到输入，已取消。")
        return False
    return answer.strip().lower() in ("y", "yes")


def _check_hf_model_cached(model_id: str, hf_home: str) -> bool:
    """判断某个 HuggingFace 模型是否已落盘。

    Args:
        model_id: 形如 "BAAI/bge-m3" 的模型标识。
        hf_home: HuggingFace 缓存根目录。

    Returns:
        对应的 models-- 目录是否存在。
    """
    model_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
    return os.path.isdir(model_dir)


def _download_hf_models(hf_home: str) -> None:
    """下载 HuggingFace 族模型，已缓存的跳过。

    每个模型下载后立即释放——脚本只需要文件落盘，没必要留在内存里。
    这样峰值内存是单个模型的开销而非累加。

    Args:
        hf_home: HuggingFace 缓存根目录。
    """
    for model_id in HF_MODELS:
        label = _HF_MODEL_LABELS.get(model_id, model_id)
        if _check_hf_model_cached(model_id, hf_home):
            print(f"  [跳过] {label} 已存在")
            continue
        print(f"  下载 {label} …")
        if "reranker" in model_id:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_id)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_id)
        del model
        gc.collect()
        print(f"  [完成] {label}")


def _verify_paddlex_cache_dir(expected: str) -> bool:
    """确认 PADDLE_PDX_CACHE_HOME 真的被 PaddleX 读到了。

    PaddleX 在 import 时把缓存根目录固化进模块常量，且该变量未见于任何官方
    文档——属于未公开的实现细节，将来可能变更。所以这里读它实际使用的值来
    验证，而不是假定"设置了就一定生效"。这一条同时保护了落盘位置校验：
    机制失效时模型会悄悄下到 C 盘默认目录。

    Args:
        expected: .env 中配置的期望路径。

    Returns:
        实际生效路径是否与期望一致。
    """
    from paddlex.utils import cache as paddlex_cache
    actual = os.path.abspath(paddlex_cache.CACHE_DIR)
    if actual == os.path.abspath(expected):
        print(f"  [校验] PaddleX 缓存目录已生效：{actual}")
        return True
    print("  ⚠️  PaddleX 未采用配置的缓存路径！")
    print(f"      配置值：{expected}")
    print(f"      实际值：{actual}")
    print("      模型将下载到实际值指向的位置，请检查 PADDLE_PDX_CACHE_HOME")
    return False


def _download_paddle_models(config: Config) -> None:
    """下载 PaddleX 模型（PP-OCRv5 检测与识别）。

    通过访问 SecurePDFProcessor 的 ocr 属性强制完成 PaddleOCR 构造——
    构造过程会解析模型、缺失则下载，最后加载进内存。这里复用生产路径的
    构造参数，保证预装的模型集与主程序实际使用的一致。

    注意：本步骤会连带把模型读入内存（数百 MB 的瞬时占用）。对只运行一次的
    预装脚本而言可以接受——换来的是"预装的模型集与生产完全一致"，比另写一套
    下载 API 更可靠。构造完成后即释放，脚本退出时全部归还给操作系统。

    Args:
        config: 全局配置，提供 ocr_lang、ocr_version 等构造参数。
    """
    from services.indexing.ocr_engine import SecurePDFProcessor
    print("  下载 PP-OCRv5 检测与识别模型 …")
    processor = SecurePDFProcessor(config)
    _ = processor.ocr
    print("  [完成] PP-OCRv5 模型")
    del processor
    # 模型权重可回收，但 paddle 生态的导入开销（实测约 400MB）会作为模块常驻，
    # 释放不掉——脚本退出时随进程一并归还
    gc.collect()


def _report(hf_home: str, paddlex_home: str) -> bool:
    """复检全部模型并就结果给出结论。

    Args:
        hf_home: HuggingFace 缓存根目录。
        paddlex_home: PaddleX 缓存根目录。

    Returns:
        全部模型是否已就位。
    """
    report = check_local_models()
    print()
    print("-" * 62)
    print("落盘校验")
    print("-" * 62)
    for item in report.items:
        state = "就位" if item.ready else "缺失"
        print(f"  [{state}] {item.name} ({item.category})")
        print(f"          {item.path}")
    if report.ready:
        print()
        print("✅ 全部模型已就位，可以启动主程序：streamlit run app.py")
        return True
    print()
    print("🚫 仍有模型未落到配置路径下，请检查上方路径与 .env 配置。")
    print(f"    当前使用的 HF_HOME = {hf_home}")
    print(f"    当前使用的 PADDLE_PDX_CACHE_HOME = {paddlex_home}")
    return False


def main() -> int:
    """执行预装全流程。

    Returns:
        进程退出码：0 表示全部就位；非 0 表示未完成（用户取消、环境有问题
        或仍有模型缺失）。
    """
    parser = argparse.ArgumentParser(
        description="本地模型预装（一次性）。把 BGE 系列与 PaddleOCR 模型"
        "下载到 .env 指定的路径。"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="跳过确认提示，直接开始"
    )
    args = parser.parse_args()

    hf_home = _resolve_or_warn(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    paddlex_home = _resolve_or_warn(
        "PADDLE_PDX_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".paddlex")
    )
    # 缓存目录须在导入 PaddleX 之前备好：它在 import 时就会往缓存目录下写 temp/，
    # 路径不存在会直接抛 FileNotFoundError，用户看到的是无上下文的路径错误
    try:
        ensure_cache_dirs()
    except OSError as exc:
        print(f"🚫 {exc}")
        return 1
    # 先把动作与开销摊开，用户确认后才动手——下载数 GB 前必须给出退出的机会
    _print_plan(hf_home, paddlex_home, check_local_models())
    if not _confirm(args.yes):
        print("已取消，未做任何改动。")
        return 1
    print()
    # 先校验 PaddleX 的路径机制，再下载——机制失效时下载下去的文件位置是错的
    _verify_paddlex_cache_dir(paddlex_home)
    print()
    print("下载 HuggingFace 模型")
    _download_hf_models(hf_home)
    print()
    print("下载 PaddleOCR 模型")
    _download_paddle_models(Config())
    return 0 if _report(hf_home, paddlex_home) else 1


if __name__ == "__main__":
    sys.exit(main())
