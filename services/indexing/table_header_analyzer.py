"""
增强的表格表头识别模块
智能识别表头特征，支持复杂表格结构
"""
from typing import List, Dict, Optional, Tuple
import re


class TableHeaderAnalyzer:
    """智能表格表头分析器"""
    
    # 常见表头特征模式
    HEADER_PATTERNS = [
        # 数字、日期、单位等
        r'^\d{4}年\d{1,2}月\d{1,2}日$',  # 日期格式
        r'^\d{4}-\d{1,2}-\d{1,2}$',        # ISO日期
        r'^[￥$€£]\d+,?\d*\.\d{2}$',     # 货币金额
        r'^\d+\.?\d*%$',                  # 百分比
        r'^第\d+$',                        # 序号
        r'^合计$|总计$|小计$',            # 统计术语
    ]
    
    # 关键表头关键词（中文）
    CHINESE_HEADER_KEYWORDS = [
        '名称', '金额', '价值', '增加', '减少', '账面', '减值', '日期', '数量',
        '编号', '时间', '部门', '项目', '金额', '占比', '比例', '增长率', '变动',
        '年初', '年末', '本期', '上年', '预算', '实际', '差异', '完成率'
    ]
    
    # 关键表头关键词（英文）
    ENGLISH_HEADER_KEYWORDS = [
        'Name', 'Amount', 'Value', 'Increase', 'Decrease', 'Book Value', 'Impairment',
        'Date', 'Quantity', 'ID', 'Time', 'Department', 'Project', 'Percentage', 'Ratio',
        'Growth Rate', 'Change', 'Beginning', 'Ending', 'Current', 'Prior', 'Budget',
        'Actual', 'Variance', 'Completion Rate'
    ]
    
    @classmethod
    def detect_header_row(cls, table_content: List[List[str]]) -> Tuple[int, bool]:
        """
        智能检测表头行
        
        Args:
            table_content: 表格内容（二维数组）
            
        Returns:
            (表头行号, 是否确认为表头)
        """
        if not table_content:
            return -1, False
        
        # 默认假设第一行为表头
        if len(table_content) == 1:
            return 0, True
        
        first_row = table_content[0]
        second_row = table_content[1]
        
        # 策略1：检查第一行是否包含表头关键词
        first_row_text = ' '.join(first_row)
        header_keywords_found = 0
        
        for keyword in cls.CHINESE_HEADER_KEYWORDS + cls.ENGLISH_HEADER_KEYWORDS:
            if keyword in first_row_text:
                header_keywords_found += 1
        
        # 如果第一行包含多个表头关键词，确认为表头
        if header_keywords_found >= 2:
            return 0, True
        
        # 策略2：检查第一行是否为特殊格式（如全为日期、数字等）
        for pattern in cls.HEADER_PATTERNS:
            match_count = sum(1 for cell in first_row if re.match(pattern, str(cell).strip()))
            if match_count >= len(first_row) * 0.5:  # 超过一半匹配
                return 0, True
        
        # 策略3：对比第一行和第二行的特征差异
        # 表头通常比数据行更独特
        if len(first_row) != len(second_row):
            return 0, True
        
        # 检查第一行的文本特征
        first_row_cells = [cell.strip() for cell in first_row if cell.strip()]
        second_row_cells = [cell.strip() for cell in second_row if cell.strip()]
        
        # 第一行通常是短文本关键词
        avg_first_length = len(''.join(first_row_cells)) / len(first_row_cells) if first_row_cells else 0
        avg_second_length = len(''.join(second_row_cells)) / len(second_row_cells) if second_row_cells else 0
        
        # 如果第一行明显更短，可能是表头
        if avg_first_length < avg_second_length * 0.7:
            return 0, True
        
        # 默认返回第一行
        return 0, True
    
    @classmethod
    def detect_multi_row_headers(cls, table_content: List[List[str]]) -> Tuple[int, int]:
        """
        检测多行表头
        
        Args:
            table_content: 表格内容（二维数组）
            
        Returns:
            (表头起始行号, 表头行数)
        """
        if len(table_content) < 3:
            return 0, 1
        
        # 分析前几行的特征
        header_start = 0
        header_end = 1
        
        for i in range(1, min(len(table_content), 5)):  # 最多检查前5行
            current_row = table_content[i]
            prev_row = table_content[i-1]
            
            # 检查是否为表头延续特征
            is_header_continuation = False
            
            # 1. 当前行包含合并单元格标记（如空字符串后跟内容）
            has_merge_pattern = any(cell.strip() for cell in current_row if cell)
            
            # 2. 当前行包含表头关键词
            current_text = ' '.join(cell for cell in current_row if cell.strip())
            has_header_keyword = any(keyword in current_text for keyword in 
                                      cls.CHINESE_HEADER_KEYWORDS + cls.ENGLISH_HEADER_KEYWORDS)
            
            # 3. 当前行较短（可能是子表头）
            if has_merge_pattern or has_header_keyword:
                header_end = i + 1
            else:
                break
        
        return header_start, header_end - header_start
    
    @classmethod
    def identify_merged_cells(cls, table_content: List[List[str]], header_row: int) -> Dict:
        """
        识别合并单元格（基于空格和内容模式）
        
        Args:
            table_content: 表格内容
            header_row: 表头行号
            
        Returns:
            合并单元格信息
        """
        merged_cells = {}
        
        if header_row < len(table_content):
            header_row_data = table_content[header_row]
            next_row_data = table_content[header_row + 1] if header_row + 1 < len(table_content) else []
            
            # 检测可能的合并单元格（空单元格）
            for i, cell in enumerate(header_row_data):
                if not cell.strip() and i < len(next_row_data) and next_row_data[i].strip():
                    merged_cells[i] = {
                        'content': next_row_data[i],
                        'merged_from_above': True
                    }
        
        return merged_cells


# 示例用法
if __name__ == "__main__":
    # 测试表格数据
    test_tables = [
        # 简单表格
        [
            ["被投资单位名称", "2024年12月31日", "本期增加", "本期减少"],
            ["公司A", "100万元", "10万元", "5万元"],
            ["公司B", "200万元", "20万元", "10万元"]
        ],
        
        # 多行表头
        [
            ["投资明细", "", "", ""],
            ["名称", "金额", "时间", "状态"],
            ["项目A", "100万", "2024-01", "进行中"],
            ["项目B", "200万", "2024-02", "已完成"]
        ],
        
        # 复杂表头
        [
            ["财务状况", "", "", "", ""],
            ["", "流动资产", "", "", ""],
            ["名称", "金额", "占比", "本期", "上年"],
            ["货币资金", "100万", "10%", "105万", "100万"]
        ]
    ]
    
    analyzer = TableHeaderAnalyzer()
    
    for i, table in enumerate(test_tables):
        print(f"表格 {i+1}:")
        header_row, is_confirmed = analyzer.detect_header_row(table)
        print(f"  表头行号: {header_row}, 确认: {is_confirmed}")
        
        multi_start, multi_rows = analyzer.detect_multi_row_headers(table)
        print(f"  多行表头: 起始行{multi_start}, 共{multi_rows}行")
        print()
    
    analyzer = TableHeaderAnalyzer()
    
    for i, table in enumerate(test_tables):
        print(f"表格 {i+1}:")
        header_row, is_confirmed = analyzer.detect_header_row(table)
        print(f"  表头行号: {header_row}, 确认: {is_confirmed}")
        
        multi_start, multi_rows = analyzer.detect_multi_row_headers(table)
        print(f"  多行表头: 起始行{multi_start}, 共{multi_rows}行")
        print()