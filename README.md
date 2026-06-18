# YOLO 边缘计算目标检测

基于 YOLOv8 的边缘计算目标检测项目，支持 PyTorch / ONNX / OpenVINO 三种推理后端，并提供实时 Web 视频流展示。

## 项目结构

```
.
├── app.py                  # Flask Web 应用，摄像头实时推理
├── pure_onnx_infer.py      # 纯 ONNXRuntime 推理性能测试
├── tran.py                 # 模型格式转换（PT → ONNX / OpenVINO）
├── test.py                 # 三种后端推理速度对比测试
├── templates/
│   └── index.html          # Web 前端页面
├── test.jpg                # 测试图片
└── best.pt                 # YOLOv8 训练权重（需自行提供）
```

> ⚠️ 模型文件（`best.pt`、`best.onnx`、`best_openvino_model/`）体积较大，未纳入版本管理，需自行准备。

---

## 环境依赖

```bash
pip install ultralytics flask opencv-python onnxruntime
```

如需使用 OpenVINO 加速：

```bash
pip install openvino
```

---

## 快速开始

### 1. 模型转换

将训练好的 `best.pt` 转换为 ONNX 和 OpenVINO 格式：

```bash
python tran.py
```

转换成功后会生成：
- `best.onnx` — ONNX 格式模型
- `best_openvino_model/` — OpenVINO 格式模型目录

### 2. 推理速度对比

对比 PyTorch、ONNX、OpenVINO 三种后端在 CPU 上的推理速度：

```bash
python test.py
```

示例输出：

```
PyTorch 平均耗时: 120.50 ms / 帧
ONNX 平均耗时:    45.30 ms / 帧
openvino 平均耗时: 38.20 ms / 帧
```

### 3. 纯 ONNXRuntime 性能测试

使用底层 ONNXRuntime 接口（不依赖 ultralytics）进行推理压测：

```bash
python pure_onnx_infer.py
```

### 4. 启动 Web 实时检测

开启摄像头并在浏览器中查看实时推理结果：

```bash
python app.py
```

启动后访问：[http://localhost:5000](http://localhost:5000)

> 默认使用 `best.pt` 模型，置信度阈值为 `0.5`，可在 `app.py` 中修改。

---

## 推理后端对比

| 后端 | 文件格式 | 特点 |
|------|----------|------|
| PyTorch | `best.pt` | 精度高，速度较慢 |
| ONNX | `best.onnx` | 跨平台，速度提升显著 |
| OpenVINO | `best_openvino_model/` | Intel 硬件优化，CPU 推理最快 |

---

## 技术栈

- **模型训练**：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **推理引擎**：ONNXRuntime、OpenVINO
- **Web 框架**：Flask
- **图像处理**：OpenCV

---

## License

MIT
