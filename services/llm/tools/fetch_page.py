"""
fetch_page 工具定义。

供 chunker 族模块使用：LLM 通过此工具跨页查找表头或上文。
页码 1-based，每次调用计入 fetch_page_limit。
"""

FETCH_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "获取PDF第N页的markdown内容，含行号。"
            "用于查找当前页表格缺失的表头、上文段落的概述等。"
            "最多调用10次，超出后不再执行。页码从1开始。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "页码（1-based），如 fetch_page(3) 获取第3页内容",
                }
            },
            "required": ["page"],
        },
    },
}
