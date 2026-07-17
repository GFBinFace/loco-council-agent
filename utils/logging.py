"""
日志工具模块

提供统一的文件日志工厂函数，各业务模块导入后传入 __file__ 即可
快速构造自己的文件日志记录器。日志策略由 config.LogConfig 统一管理。

所有日志文件统一输出到项目根目录的 logs/ 下，不随模块所在子目录分散。
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import config as _root_config
from config import LogConfig

# 项目根目录——所有日志文件统一落在此目录下的 logs/ 中
_PROJECT_ROOT = os.path.dirname(os.path.abspath(_root_config.__file__))


# ── 内部工具 ──────────────────────────────────────────────

def _caller_name() -> str:
    """返回调用方的函数名（跳过本模块自身）。"""
    return sys._getframe(2).f_code.co_name


def get_file_logger(module_file: str) -> logging.Logger:
    """为指定模块创建文件日志记录器

    约定：
    - 日志文件以模块文件名命名（如 ocr_engine.py → ocr_engine.log）
    - 统一输出到项目根目录的 config.LogConfig.LOG_DIR 下
    - 追加模式写入，单文件上限 10MB，超出后轮转（保留 1 个备份）
    - 单文件上限由 config.LogConfig.MAX_BYTES 控制

    Args:
        module_file: 调用方的 __file__，仅用于推导日志文件名

    Returns:
        配置好的 logging.Logger 实例，propagate=False，仅写入文件
    """
    log_dir = os.path.join(_PROJECT_ROOT, LogConfig.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名 = 模块文件名 + ".log"
    module_name = os.path.splitext(os.path.basename(module_file))[0]
    log_file = os.path.join(log_dir, f"{module_name}.log")

    logger_name = f"{module_name}.file"
    logger_ = logging.getLogger(logger_name)
    logger_.setLevel(logging.INFO)
    logger_.propagate = False

    # 清除已有 handler，保证每次运行全新日志
    logger_.handlers.clear()

    handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=LogConfig.MAX_BYTES,
        backupCount=0,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger_.addHandler(handler)

    return logger_


def get_debug_data_logger(module_file: str) -> logging.Logger:
    """为指定模块创建 debug data 日志记录器。

    日志模式由 LogConfig.DEBUG_DATA_MODE 控制（"overwrite" / "append"）。
    日志文件以 模块名_data.log 命名，与业务日志同在项目根目录 logs/ 下。

    通过 write_debug_data(logger, func_name, msg) 写入各环节关键数据。

    Args:
        module_file: 调用方的 __file__，仅用于推导日志文件名

    Returns:
        配置好的 logging.Logger 实例，DEBUG 级别，propagate=False
    """
    log_dir = os.path.join(_PROJECT_ROOT, LogConfig.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)

    module_name = os.path.splitext(os.path.basename(module_file))[0]
    log_file = os.path.join(log_dir, f"{module_name}_data.log")

    logger_name = f"{module_name}_data"
    logger_ = logging.getLogger(logger_name)
    logger_.setLevel(logging.DEBUG)
    logger_.propagate = False

    logger_.handlers.clear()

    file_mode = "w" if LogConfig.DEBUG_DATA_MODE == "overwrite" else "a"
    handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger_.addHandler(handler)

    return logger_


def write_debug_data(
    logger: logging.Logger | None,
    msg: str,
    func_name: str | None = None,
) -> None:
    """输出单行 debug data log——用于统计摘要或阶段说明等简短信息。

    输出格式：  [函数名] 消息内容

    如需输出大段原始数据（如 OCR 正文、chunk 内容），请使用 write_debug_data_lines()。

    Args:
        logger:    由 get_debug_data_logger 创建的 logger，或 None
        msg:       日志内容（单行摘要）
        func_name: 所属函数名。传 None 时自动从调用栈获取
    """
    if logger:
        if func_name is None:
            func_name = _caller_name()
        logger.debug("[%s] %s", func_name, msg)


def write_debug_data_lines(
    logger: logging.Logger | None,
    lines: list[str],
    func_name: str | None = None,
) -> None:
    """输出多行 debug data log——用于大段原始数据（如 OCR 正文、chunk 全文）。

    输出格式：  [函数名]
              第 1 行
              第 2 行
              ...

    封装了 logger.debug("\\n".join(lines)) 模式，调用方无需手动拼接。

    Args:
        logger:    由 get_debug_data_logger 创建的 logger，或 None
        lines:     多行文本列表，每项为一行
        func_name: 所属函数名。传 None 时自动从调用栈获取
    """
    if logger:
        if func_name is None:
            func_name = _caller_name()
        logger.debug("[%s]\n%s", func_name, "\n".join(lines))
