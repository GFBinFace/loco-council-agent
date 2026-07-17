"""
submit_chunks 工具定义。

供 chunker 族模块使用：LLM 在完成分析后必须通过此工具提交切块结果。
参数类型由 API 强制校验，无需担心格式错误。
"""

SUBMIT_CHUNKS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_chunks",
        "description": (
            "提交切块结果。在完成所有分析和必要的 fetch_page 调用后，"
            "必须通过此工具提交最终的 chunk 列表。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "description": "本批次所有页面切出的 chunk 列表，每个 chunk 可跨页",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_type": {
                                "type": "string",
                                "enum": ["text", "table", "mixed"],
                            },
                            "context_summary": {
                                "type": ["string", "null"],
                                "description": "上文概述，找不到或纯文本块填 null",
                            },
                            "header_source": {
                                "type": ["object", "null"],
                                "description": "表头来源，本页含表头或非表格块填 null",
                                "properties": {
                                    "page": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "description": "表头所在页码",
                                    },
                                    "lines": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "表头行号列表",
                                    },
                                },
                                "required": ["page", "lines"],
                            },
                            "segments": {
                                "type": "array",
                                "minItems": 1,
                                "description": "chunk 内容来源，按序拼接即为完整文本。单页 chunk 仅含一个元素",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "page": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "description": "页码（1-based）",
                                        },
                                        "line_range": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "description": "[起始行号, 结束行号]，闭区间",
                                        },
                                    },
                                    "required": ["page", "line_range"],
                                },
                            },
                        },
                        "required": ["chunk_type", "segments"],
                    },
                }
            },
            "required": ["chunks"],
        },
    },
}
