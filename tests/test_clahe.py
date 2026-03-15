"""
测试光照预处理相关函数: control_lighting_by_clahe, control_lighting_by_msr
可以在开头选择使用的方法
Created by Gemini
"""
import os
import sys
import cv2
import numpy as np

# 将项目根目录加入到sys.path中，以便正确导入模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.process.size_control import size_process
from utils.process.lighting_process import control_lighting_by_clahe, control_lighting_by_msr

# ==========================================
# 在这里选择你想测试的光照处理方法
# 可选值: 'CLAHE', 'MSR'
# ==========================================
SELECTED_METHOD = 'MSR'

def test_lighting_process():
    # 使用 example 中的图片
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "examples", "picture", "02_H_4", "000000.png")
    
    # 尺寸裁剪为1024
    target_size = 1024
    
    # 加载图片
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"错误: 无法加载图片，请检查路径。 ({img_path})")
        return
        
    print(f"Original image shape: {img.shape}")
    
    print(f"1. Processing image to size: {target_size}x{target_size}...")
    sized_img = size_process(img, target_size)
    print(f"Sized image shape: {sized_img.shape}")
    
    print(f"2. Applying {SELECTED_METHOD}...")
    
    if SELECTED_METHOD == 'CLAHE':
        processed_img = control_lighting_by_clahe(sized_img, clipLimit=2.0, tileGridSize=(8, 8))
    elif SELECTED_METHOD == 'MSR':
        processed_img = control_lighting_by_msr(sized_img)
        processed_img = control_lighting_by_clahe(processed_img)
    else:
        print(f"错误: 不支持的方法 {SELECTED_METHOD}")
        return
        
    print(f"Processed image shape: {processed_img.shape}")
    
    # 将处理前和处理后的图片拼接，方便作对比
    separator = np.ones((target_size, 10, 3), dtype=np.uint8) * 128
    comparison = cv2.hconcat([sized_img, separator, processed_img])

    # 保存测试结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_result")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"lighting_test_comparison_{SELECTED_METHOD}.jpg")
    cv2.imwrite(output_path, comparison)
    print(f"Saved comparison image to: {output_path}")

    # 展示图片对比
    window_name = f"Left: Original (1024x1024) | Right: {SELECTED_METHOD} Processed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, comparison)
    print("已展示对比图片，请在图片窗口按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_lighting_process()
