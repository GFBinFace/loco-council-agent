"""测试 utils/common 的通用工具函数。"""

from utils import format_duration


class TestFormatDuration:
    """时长格式化：秒 / 分+秒 / 时+分+秒 三档。"""

    def test_zero_shows_zero_seconds(self):
        assert format_duration(0) == "0.0秒"

    def test_under_one_minute_keeps_one_decimal(self):
        assert format_duration(15.53) == "15.5秒"

    def test_just_under_one_minute_stays_in_seconds(self):
        assert format_duration(59.9) == "59.9秒"

    def test_exactly_one_minute_switches_to_minutes(self):
        assert format_duration(60) == "1分0秒"

    def test_minute_level_drops_decimal(self):
        """135 秒是索引单页的常见量级，应读作 2分15秒 而不是 135.0秒。"""
        assert format_duration(135.0) == "2分15秒"

    def test_just_under_one_hour_stays_in_minutes(self):
        assert format_duration(3599) == "59分59秒"

    def test_exactly_one_hour_switches_to_hours(self):
        assert format_duration(3600) == "1时0分0秒"

    def test_hour_level_shows_all_three_parts(self):
        assert format_duration(3725) == "1时2分5秒"

    def test_negative_is_clamped_to_zero(self):
        """计时器异常时不应输出负数时长。"""
        assert format_duration(-5) == "0.0秒"
