"""管线进度上报器。

每个 search() / continue_search() / index_pdf() 调用创建一个独立实例，
自带隔离的计时器，支持多线程场景。

协议：
    on_progress(status_line: str | None, log_line: str | None) -> None

调用时机：
    子阶段开始 → report_phase_start(status_line, log_line)
    子阶段结束 → report_phase_end(log_line)（自动追加耗时）
    任务完成   → report_task_end(status_line)

sub-phase 可视化：
    report_phase_start(status="OCR 扫描中… 3/16", log="开始处理第 3/16 页 OCR")
        阶段内工作
    report_phase_end(log="第 3/16 页 OCR 完成，获得 2045 字符")
        → 实际发出: "第 3/16 页 OCR 完成，获得 2045 字符，耗时 15.5s"
"""

import logging
import time
from typing import Callable, Optional


class ProgressReporter:
    """进度上报器——封装计时，产出用户可读的状态行和状态 log。

    上对用户（UI 回调），下对系统（文件日志），是进度汇报的唯一出口。
    """

    _TICK_IDLE = -1.0  # 标记"计时器未启动"

    def __init__(
        self,
        on_progress: Optional[Callable] = None,
        logger: Optional["logging.Logger"] = None,
    ):
        if on_progress is None:
            self._emit: Callable = lambda status_line, log_line: None
        else:
            self._emit = on_progress
        self._logger = logger
        self._tick: float = self._TICK_IDLE

    # ── 公开接口 ──────────────────────────────────────────

    def report_phase_start(
        self, status_line: Optional[str], log_line: Optional[str],
    ) -> None:
        """
        阶段开始：发送状态行 + 状态 log，启动内部计时器。

        status_line 为 None 时仅发送 log_line（用于毫秒级阶段）。
        """
        self._tick = time.time()
        self._emit(status_line, log_line)
        if self._logger and log_line:
            self._logger.info(log_line)

    def report_phase_end(self, log_line: Optional[str]) -> None:
        """
        阶段结束：自动在末尾追加「，耗时 {elapsed}s」后发送 log_line，
        不发状态行。调用方传入的 log_line 不应包含耗时文字。

        Raises:
            RuntimeError: 缺少前置 report_phase_start()。
        """
        if self._tick == self._TICK_IDLE:
            raise RuntimeError(
                "ProgressReporter.report_phase_end() 缺少前置 "
                "report_phase_start()，计时器未启动。"
            )
        elapsed = round(time.time() - self._tick, 1)
        self._tick = self._TICK_IDLE
        if log_line is not None:
            log_line = f"{log_line}，耗时 {elapsed}s"
        self._emit(None, log_line)
        if self._logger and log_line:
            self._logger.info(log_line)

    def report_task_end(self, status_line: Optional[str]) -> None:
        """任务完成（或取消）：仅发送状态行，不发状态 log。"""
        self._tick = self._TICK_IDLE  # 清理计时器
        self._emit(status_line, None)
        if self._logger and status_line:
            self._logger.info(status_line)
