"""
测试utils.process.size_control中的size_process函数
Created by Gemini
"""
import os
import sys
import cv2

from utils.process.size_control import size_process

def test_size_process():
    # 使用 example 中的图片
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "examples", "picture", "01_H_0.JPG")
    
    # 加载图片
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)
    
    if img is None:
        print("错误: 无法加载图片，请检查路径。")
        return
        
    print(f"Original image shape: {img.shape}")
    
    # 测试处理为 1024*1024 大小
    target_size = 1024
    print(f"Processing image to size: {target_size}x{target_size}...")
    processed_img = size_process(img, target_size)
    
    print(f"Processed image shape: {processed_img.shape}")
    
    # 保存测试结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_result")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "size_control_test_1024.jpg")
    cv2.imwrite(output_path, processed_img)
    print(f"Saved processed image to: {output_path}")

    # 展示图片
    cv2.namedWindow("Processed Image (1024x1024)", cv2.WINDOW_NORMAL)
    cv2.imshow("Processed Image (1024x1024)", processed_img)
    print("已展示图片，请在图片窗口按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_size_process()
