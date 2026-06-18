# 智能视觉系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.8+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLO-v8-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-purple.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/EasyOCR-1.7+-red.svg" alt="EasyOCR">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📖 项目简介

这是一个基于YOLOv8和MediaPipe的智能视觉分析系统，集成目标检测、人脸识别、车牌识别、手势识别、目标跟踪、批量处理、数据可视化、视频处理、模型训练、报告导出十大功能。项目采用模块化设计，支持图片、视频和摄像头实时检测，可应用于安防监控、智能交通、人机交互等多个场景。

## ✨ 功能特点

- 🎯 **目标检测**：基于YOLOv8模型，支持80+常见物体类别
- 👤 **人脸识别**：基于MediaPipe，支持多人脸检测
- 🚗 **车牌识别**：基于EasyOCR，支持车牌号码识别
- ✋ **手势识别**：基于MediaPipe，支持10种常见手势识别
- 🎯 **目标跟踪**：基于YOLOv8，支持视频目标跟踪
- 📦 **批量处理**：支持批量图片处理和结果导出
- 📊 **数据可视化**：提供检测结果统计图表
- 🎬 **视频处理**：支持视频目标检测和人脸识别
- 🎓 **模型训练**：支持自定义数据集训练
- 📄 **报告导出**：支持HTML、CSV、JSON格式报告导出
- ⚡ **实时处理**：优化推理速度，支持30+ FPS实时检测
- 📷 **多源输入**：支持图片、视频、摄像头等多种输入方式
- 🎨 **可视化界面**：提供友好的Streamlit Web界面
- 🔌 **API接口**：提供RESTful API，便于集成到其他系统

## 🛠️ 技术栈

| 技术        | 用途           |
| ----------- | -------------- |
| Python 3.8+ | 主要编程语言   |
| PyTorch     | 深度学习框架   |
| YOLOv8      | 目标检测模型   |
| MediaPipe   | 人脸和手部检测 |
| EasyOCR     | 文字识别       |
| OpenCV      | 图像处理       |
| Matplotlib  | 数据可视化     |
| Streamlit   | Web界面        |
| FastAPI     | API服务        |

## 📁 项目结构

```
yolov8-object-detection/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── setup.py                     # 安装配置
├── Dockerfile                   # Docker配置
├── docker-compose.yml           # Docker Compose配置
├── nginx.conf                   # Nginx配置
├── config.yaml                  # 应用配置文件
├── app.py                       # Streamlit Web应用
├── api.py                       # FastAPI接口
├── docs/                        # 文档目录
│   ├── installation.md          # 安装指南
│   ├── usage.md                 # 使用说明
│   └── examples.md              # 示例说明
├── examples/                    # 示例代码
│   ├── detect_image.py          # 图片检测示例
│   ├── detect_video.py          # 视频检测示例
│   ├── detect_camera.py         # 摄像头检测示例
│   ├── detect_samples.py        # 示例数据集检测
│   ├── face_detection.py        # 人脸识别示例
│   ├── face_dataset.py          # 人脸数据集示例
│   ├── plate_detection.py       # 车牌识别示例
│   ├── gesture_detection.py     # 手势识别示例
│   └── train_custom.py          # 自定义训练示例
├── src/                         # 源代码目录
│   ├── __init__.py
│   ├── detector.py              # 目标检测器核心类
│   ├── face_detector.py         # 人脸识别器核心类
│   ├── plate_recognizer.py      # 车牌识别器核心类
│   ├── gesture_recognizer.py    # 手势识别器核心类
│   ├── tracker.py               # 目标跟踪器
│   ├── batch_processor.py       # 批量处理器
│   ├── visualizer.py            # 数据可视化器
│   ├── trainer.py               # 模型训练器
│   ├── video_processor.py       # 视频处理器
│   ├── report_generator.py      # 报告生成器
│   ├── config.py                # 配置管理器
│   └── utils.py                 # 工具函数
├── configs/                     # 配置文件
│   ├── default.yaml
│   └── custom.yaml
├── models/                      # 模型目录
│   └── README.md
├── data/                        # 数据目录
│   ├── samples/                 # 示例图片
│   ├── plate/                   # 车牌数据集
│   ├── gesture/                 # 手势数据集
│   ├── face/                    # 人脸数据集
│   └── README.md
└── tests/                       # 测试文件
    └── test_all.py              # 单元测试
```

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/yolov8-object-detection.git
cd yolov8-object-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 使用示例

#### 图片检测

```python
from src.detector import ObjectDetector

detector = ObjectDetector()
results = detector.detect_image("data/samples/test.jpg")
```

#### 视频检测

```python
detector = ObjectDetector()
detector.detect_video("input.mp4", "output.mp4")
```

#### 摄像头实时检测

```python
detector = ObjectDetector()
detector.detect_camera(camera_id=0)
```

#### 人脸识别

```python
from src.face_detector import FaceDetector

# 创建检测器
detector = FaceDetector()

# 检测人脸
results = detector.detect_faces("photo.jpg")
print(f"检测到 {results['face_count']} 张人脸")

# 绘制结果
detector.draw_faces("photo.jpg", "result.jpg")
```

#### 车牌识别

```python
from src.plate_recognizer import LicensePlateRecognizer

# 创建识别器
recognizer = LicensePlateRecognizer()

# 检测车牌
results = recognizer.detect_plate("car.jpg")
print(f"检测到 {results['plate_count']} 个车牌")

# 绘制结果
recognizer.draw_plates("car.jpg", "result.jpg")
```

#### 手势识别

```python
from src.gesture_recognizer import GestureRecognizer

# 创建识别器
recognizer = GestureRecognizer()

# 识别手势
results = recognizer.recognize_gesture("hand.jpg")
print(f"检测到 {results['hand_count']} 只手")

for gesture in results["gestures"]:
    print(f"手势: {gesture['gesture']}")
```

### 实时检测

```bash
# 实时目标检测
python examples/realtime_detection.py

# 实时人脸识别
python examples/realtime_face.py

# 实时手势识别
python examples/realtime_gesture.py

# 实时综合检测（目标+人脸+手势）
python examples/realtime_all.py
```

### 启动Web界面

```bash
streamlit run app.py
```

### 启动API服务

```bash
uvicorn api:app --reload
```

### Docker部署

```bash
# 构建Docker镜像
docker build -t vision-system .

# 运行单个容器
docker run -p 8000:8000 -p 8501:8501 vision-system

# 使用Docker Compose部署
docker-compose up -d
```

### 配置管理

```python
from src.config import ConfigManager

# 创建配置管理器
manager = ConfigManager("config.yaml")

# 加载配置
config = manager.load()

# 获取配置值
model_path = config.model.detection_model
api_port = config.api.port

# 设置配置值
manager.set("model.detection_model", "yolov8s.pt")
manager.set("api.port", 8001)

# 保存配置
manager.save()
```

## 📊 支持的检测类别

<details>
<summary>点击展开查看所有类别（80类）</summary>

| 类别          | 类别          | 类别          | 类别           |
| ------------- | ------------- | ------------- | -------------- |
| person        | bicycle       | car           | motorcycle     |
| airplane      | bus           | train         | truck          |
| boat          | traffic light | fire hydrant  | stop sign      |
| parking meter | bench         | bird          | cat            |
| dog           | horse         | sheep         | cow            |
| elephant      | bear          | zebra         | giraffe        |
| backpack      | umbrella      | handbag       | tie            |
| suitcase      | frisbee       | skis          | snowboard      |
| sports ball   | kite          | baseball bat  | baseball glove |
| skateboard    | surfboard     | tennis racket | bottle         |
| wine glass    | cup           | fork          | knife          |
| spoon         | bowl          | banana        | apple          |
| sandwich      | orange        | broccoli      | carrot         |
| hot dog       | pizza         | donut         | cake           |
| chair         | couch         | potted plant  | bed            |
| dining table  | toilet        | tv            | laptop         |
| mouse         | remote        | keyboard      | cell phone     |
| microwave     | oven          | toaster       | sink           |
| refrigerator  | book          | clock         | vase           |
| scissors      | teddy bear    | hair drier    | toothbrush     |

</details>

## 📈 性能指标

| 模型    | mAP50 | mAP50-95 | 速度(FPS) |
| ------- | ----- | -------- | --------- |
| YOLOv8n | 52.5  | 37.3     | 80+       |
| YOLOv8s | 61.8  | 45.0     | 60+       |
| YOLOv8m | 67.2  | 50.2     | 45+       |
| YOLOv8l | 71.8  | 53.9     | 35+       |
| YOLOv8x | 74.9  | 55.8     | 25+       |

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📝 更新日志

### v1.0.0 (2024-XX-XX)

- 初始版本发布
- 支持图片/视频/摄像头检测
- 提供Streamlit Web界面
- 提供FastAPI接口

## 📧 联系方式

- 邮箱：your.email@example.com
- GitHub：[@yourusername](https://github.com/yourusername)

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8官方实现
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

---

<p align="center">
  如果觉得有用，请给个 ⭐ Star 支持一下！
</p>
