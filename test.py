import time
import cv2
from ultralytics import YOLO


import os
# 强制让 PyTorch 和 ONNX 都看不到 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def benchmark_models(image_path="test.jpg", num_runs=100):
    # 读取一张测试图片 (请确保当前目录下有一张名为 test.jpg 的图片)
    # 也可以替换为从摄像头捕获的一帧
    img = cv2.imread(image_path)
    if img is None:
        print("未找到测试图片 test.jpg，请准备一张图片用于测试。")
        return

    print("正在加载模型...")
    model_pt = YOLO("best.pt")
    model_onnx = YOLO("best.onnx")
    model_openvino = YOLO("best_openvino_model/")
    # ==========================================
    # 1. PyTorch 模型推理测试
    # ==========================================
    print("\n--- 开始 PyTorch (.pt) 推理测试 ---")
    # 预热 (Warmup): 深度学习模型前几次推理需要分配显存/内存，通常较慢
    for _ in range(5):
        model_pt(img, verbose=False)

    start_time = time.time()
    for _ in range(num_runs):
        _ = model_pt(img, verbose=False)
    pt_avg_time = (time.time() - start_time) / num_runs * 1000
    print(f"PyTorch 平均耗时: {pt_avg_time:.2f} ms / 帧")

    # ==========================================
    # 2. ONNX 模型推理测试
    # ==========================================
    print("\n--- 开始 ONNX (.onnx) 推理测试 ---")
    # 预热
    for _ in range(5):
        model_onnx(img, verbose=False)

    start_time = time.time()
    for _ in range(num_runs):
        _ = model_onnx(img, verbose=False)
    onnx_avg_time = (time.time() - start_time) / num_runs * 1000
    print(f"ONNX 平均耗时: {onnx_avg_time:.2f} ms / 帧")

    # ==========================================
    # 2.openvino模型推理测试
    # ==========================================
    print("\n--- 开始 openvino (.onnx) 推理测试 ---")
    # 预热
    for _ in range(5):
        model_openvino(img, verbose=False)

    start_time = time.time()
    for _ in range(num_runs):
        _ = model_openvino(img, verbose=False)
    openvino_avg_time = (time.time() - start_time) / num_runs * 1000
    print(f"openvino 平均耗时: {openvino_avg_time:.2f} ms / 帧")
    # ==========================================
    # 3. 结论输出
    # ==========================================
    speedup = pt_avg_time / onnx_avg_time
    speedup1 = openvino_avg_time / onnx_avg_time
    print(f"\n✅ 测试完成！ONNX 模型比 PyTorch 模型快了大约 {speedup:.2f} 倍。")
    print(f"\n✅ 测试完成！ONNX 模型比 openvino 模型快了大约 {speedup1:.2f} 倍。")
if __name__ == "__main__":
    # 请放一张名为 test.jpg 的图片在同目录下
    benchmark_models("test.jpg", num_runs=100)