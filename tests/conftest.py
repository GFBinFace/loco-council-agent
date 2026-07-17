"""pytest 全局配置。

在任何业务模块导入之前，将文件日志重定向到临时目录——
测试中故意 mock 的异常场景（超时、格式错误等）会照常写日志，
若与真实运行共享根 logs/ 目录，会污染用户事后排查用的生产日志。

get_file_logger 在业务模块导入时调用并读取 LogConfig.LOG_DIR，
conftest 先于所有测试模块加载，此处改写即可全局生效。
"""

import tempfile

from config import LogConfig

LogConfig.LOG_DIR = tempfile.mkdtemp(prefix="loco_test_logs_")
