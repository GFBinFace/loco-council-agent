# test_color_format.py
import fitz
import cv2
import numpy as np

def verify_color_format(pdf_path: str, page_num: int = 0):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # 获取图片
    mat = fitz.Matrix(200/72, 200/72)
    pix = page.get_pixmap(matrix=mat)
    img_rgb = np.frombuffer(pix.tobytes(), dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    
    # 转换
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 保存对比
    cv2.imwrite("test_rgb.jpg", img_rgb)   # 颜色会偏蓝
    cv2.imwrite("test_bgr.jpg", img_bgr)   # 颜色正常
    
    print("RGB 和 BGR 版本已保存，请对比查看")
    print("- test_rgb.jpg: 颜色可能偏蓝（红蓝通道互换）")
    print("- test_bgr.jpg: 颜色应正常")
    
    doc.close()

# 运行测试
# verify_color_format("your_test.pdf")