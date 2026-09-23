"""测试 ProgressReporter 的进度上报协议。

重点锁住两件事：
1. 任务收尾的状态 log 行是可选的——默认不发（避免与最后的 phase_end 重复），
   需要给 log 留总结时才传。
2. 阶段收尾自动追加的耗时是人读格式（秒 / 分+秒 / 时+分+秒）。
"""

import pytest

from services.progress_reporter import ProgressReporter


def _make_recorder():
    """构造一个把 (status_line, log_line) 收进列表的上报器。

    Returns:
        (reporter, emitted) —— emitted 为收集列表。
    """
    emitted = []
    reporter = ProgressReporter(on_progress=lambda s, l: emitted.append((s, l)))
    return reporter, emitted


class TestReportTaskEnd:
    """任务收尾：只发状态行。

    完成类的状态 log 由 UI 层在拿到最终结果后自己发——它才握有错误文本、
    耗时等完整信息。两边都发会让同一件事在 log 里出现两遍。
    """

    def test_sends_status_line_only(self):
        reporter, emitted = _make_recorder()

        reporter.report_task_end("索引完成")

        assert emitted == [("索引完成", None)]


class TestReportPhaseEnd:
    """阶段收尾：自动追加人读格式的耗时。"""

    def test_appends_human_readable_duration(self):
        reporter, emitted = _make_recorder()
        reporter.report_phase_start("扫描中", "开始扫描")

        reporter.report_phase_end("扫描完成")

        _, log_line = emitted[-1]
        assert log_line.startswith("扫描完成，耗时 ")
        assert log_line.endswith("秒")

    def test_phase_end_without_prior_start_raises(self):
        reporter, _ = _make_recorder()

        with pytest.raises(RuntimeError):
            reporter.report_phase_end("没有前置 start")
