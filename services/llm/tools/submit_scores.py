"""
submit_scores 工具定义。

供 RerankLLM 使用：LLM 完成打分后必须通过此工具提交评分结果。
参数类型由 API 强制校验，无需担心格式错误或字段缺失。
"""

SUBMIT_SCORES_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_scores",
        "description": (
            "提交每个文档片段的相关性评分。"
            "score 为 0-10 的整数，0=完全不相关，10=完全匹配。"
            "必须为输入中的每个片段都提交一条评分。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "description": "每个片段的评分结果，顺序与输入片段一致",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "片段 ID",
                            },
                            "score": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 10,
                                "description": "0-10 的整数分数",
                            },
                            "reason": {
                                "type": "string",
                                "description": "一句话评分理由（中文）",
                            },
                        },
                        "required": ["chunk_id", "score", "reason"],
                    },
                },
            },
            "required": ["scores"],
        },
    },
}
