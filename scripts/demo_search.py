"""
检索管线端到端演示。

用法：
    python demo_search.py [查询文本]

示例：
    python demo_search.py
    python demo_search.py "长期股权投资变动情况"

前提条件：
    - LanceDB 中已有索引数据（通过 index_pdf 流程入库）
    - 设置了 DEEPSEEK_API_KEY 环境变量（或在 .env 文件中）
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

# 日志：控制台 + 文件
os.makedirs("logs", exist_ok=True)
_file_handler = logging.FileHandler("logs/demo_search.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.basicConfig(
    level=logging.WARNING,  # 根级别 WARNING，屏蔽第三方库噪音
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), _file_handler],
)
# 只放行我们自己的模块 INFO
for _mod in ("pipeline", "retriever", "reranker", "llm_client",
              "doc_manager", "chunkers.financial_table_chunker", "embedder",
              "progress_reporter", "logging_utils"):
    logging.getLogger(_mod).setLevel(logging.INFO)
load_dotenv()

# HF_HOME 必须在 import pipeline 之前设置——
# pipeline 的 import 链会触发 sentence_transformers 导入，该库在导入时读取 HF_HOME。
from config import Config
if Config.huggingface_cache_dir:  # dataclass 默认值 = 类属性
    os.environ["HF_HOME"] = Config.huggingface_cache_dir
from services.pipeline import RAGPipeline
from _types.retrieval_types import ContinueChoice


def progress_reporter(status_line: str | None, log_line: str | None):
    """演示进度回调——管线直接产出中文状态行和状态 log。"""
    if status_line:
        print(f"Status Line: {status_line}")
    if log_line:
        print(f"Log Line: {log_line}")


def handle_user_choice(pipeline: RAGPipeline, result) -> None:
    """处理需要用户选择的场景，模拟 CLI 交互。"""
    if result.pending_decision == "zero_results":
        print()
        print("─" * 60)
        print("知识库中未检索到与您问题相关的资料。")
        print("请选择：")
        print("  A. 不使用知识库资料，由 AI 直接回答")
        print("  B. 放弃本次查询")
        choice = input("请输入 A 或 B: ").strip().upper()
        if choice == "A":
            final = pipeline.continue_search(ContinueChoice.DIRECT_LLM, on_progress=progress_reporter)
        else:
            final = pipeline.continue_search(ContinueChoice.ABANDON, on_progress=progress_reporter)

    elif result.pending_decision == "low_confidence":
        print()
        print("─" * 60)
        print(f"知识库中未找到与您问题高度相关的资料（最高相关性评分 {result.top_score}/10）。")
        print("请选择：")
        print("  A. 基于匹配度较低的文档片段尝试回答（可能不够准确）")
        print("  B. 不使用知识库资料，由 AI 直接回答")
        choice = input("请输入 A 或 B: ").strip().upper()
        if choice == "A":
            final = pipeline.continue_search(ContinueChoice.RAG, on_progress=progress_reporter)
        else:
            final = pipeline.continue_search(ContinueChoice.DIRECT_LLM, on_progress=progress_reporter)
    else:
        print(f"未知的 decision 类型: {result.pending_decision}")
        return

    print()
    print("─" * 60)
    print("📝 最终回答：")
    print("─" * 60)
    print(final.answer)
    print("─" * 60)

    if final.sources:
        print(f"来源 ({len(final.sources)} 个 chunk)：")
        for s in final.sources:
            print(f"  - {s.doc_name or s.doc_id} 页码 {s.page_nums}")

    print()


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "长期股权投资变动情况"

    # 检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("⚠️  未设置 DEEPSEEK_API_KEY 环境变量")
        print("  LLM 打分和生成回答将无法工作（需要 DeepSeek API）")
        print("  你可以设置该变量后重新运行本脚本")
        print()
        print("  PowerShell: $env:DEEPSEEK_API_KEY='your-key'")
        print("  CMD:       set DEEPSEEK_API_KEY=your-key")
        return

    print("=" * 60)
    print(f"🔎 查询: {query}")
    print("=" * 60)
    print()

    # 初始化管线（首次运行会加载 Embedder 和 CrossEncoder 模型）
    print("⏳ 初始化管线（加载 BGE-M3 + BGE-Reranker 模型）…")
    t0 = time.time()
    config = Config()
    pipeline = RAGPipeline(config)
    print(f"✅ 初始化完成，耗时 {time.time() - t0:.1f}s")
    print()

    # 执行检索
    result = pipeline.search(query, on_progress=progress_reporter)
    if result.status == "needs_user_choice":
        handle_user_choice(pipeline, result)
    elif result.status == "success":
        print()
        print("─" * 60)
        print("📝 最终回答：")
        print("─" * 60)
        print(result.answer)
        print("─" * 60)
        # 输出来源数据
        if result.sources:
            print(f"来源 ({len(result.sources)} 个 chunk)：")
            for s in result.sources:
                print(f"  - {s.doc_name or s.doc_id} 页码 {s.page_nums} "
                      f"chunk#{s.chunk_index} LLM得分={s.llm_score}")
        print()
    else:
        print(f"❌ 错误: {result.error_message}")


if __name__ == "__main__":
    main()
