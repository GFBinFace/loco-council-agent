"""pytest 全局配置。

在任何业务模块导入之前完成两件事：

1. 将文件日志重定向到临时目录——测试中故意 mock 的异常场景（超时、
   格式错误等）会照常写日志，若与真实运行共享根 logs/ 目录，会污染
   用户事后排查用的生产日志。
2. 提供占位 API key——LLM 客户端构造时要求凭据非空，但测试全程 mock
   掉真实调用，不应依赖开发者本机的 .env 或系统环境变量。

get_file_logger 在业务模块导入时调用并读取 LogConfig.LOG_DIR，
conftest 先于所有测试模块加载，此处改写即可全局生效。
"""

import os
import tempfile

from config import LogConfig

LogConfig.LOG_DIR = tempfile.mkdtemp(prefix="loco_test_logs_")

# 占位凭据：只为通过客户端构造时的非空校验，测试不会真正发出请求。
# 用 setdefault 而非直接赋值，避免覆盖开发者已有的真实 key。
os.environ.setdefault("DEEPSEEK_API_KEY", "test-placeholder-key")
