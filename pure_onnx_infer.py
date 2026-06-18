import cv2
import numpy as np
import onnxruntime as ort

def preprocess(img):
    """底层图像预处理：缩放、通道变换与归一化"""
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB, HWC2CHW
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def benchmark_true_speed(image_path="test.jpg", num_runs=50):
    img = cv2.imread(image_path)
    if img is None:
        print("请提供 test.jpg")
        return

    # 1. 初始化纯 ORT 引擎
    ort_session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    print("\n--- 开始 纯 ONNXRuntime 推理测试 ---")
    
    # 预热
    blob = preprocess(img)
    for _ in range(3): 
        ort_session.run(None, {input_name: blob})
    
    # 性能压测
    import time
    start = time.time()
    for _ in range(num_runs):
        blob = preprocess(img)
        _ = ort_session.run(None, {input_name: blob})
    
    ort_avg = (time.time() - start) / num_runs * 1000
    print(f"纯 ONNXRuntime 包含预处理的平均耗时: {ort_avg:.2f} ms / 帧")

if __name__ == "__main__":
    benchmark_true_speed("test.jpg")