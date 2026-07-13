# 安装指南

## 系统要求

- Python 3.8+
- 操作系统：Windows 10+, macOS 10.15+, Ubuntu 18.04+
- 内存：8GB+（推荐16GB）
- 显卡：NVIDIA GPU（可选，用于加速推理）

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/vision-system.git
cd vision-system
```

### 2. 创建虚拟环境

```bash
# 使用venv
python -m venv venv
source venv/bin/activate # Linux/Mac
# 或
venv\Scripts\activate # Windows

# 使用conda
conda create -n vision python=3.10
conda activate vision
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型

```bash
# YOLOv8模型会在首次使用时自动下载
# 也可以手动下载
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 5. 验证安装

```bash
python -c "
from src.detector import ObjectDetector
from src.face_detector import FaceDetector
from src.gesture_recognizer import GestureRecognizer
print('安装成功!')
"
```

## Docker安装

### 1. 构建镜像

```bash
docker build -t vision-system .
```

### 2. 运行容器

```bash
docker run -p 8000:8000 -p 8501:8501 vision-system
```

### 3. 使用Docker Compose

```bash
docker-compose up -d
```

## 常见问题

### Q: 安装dlib失败怎么办？

A: dlib需要C++编译器。Windows用户需要安装Visual Studio Build Tools。

### Q: GPU加速如何配置？

A: 安装CUDA和cuDNN，然后安装PyTorch GPU版本：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q: 模型下载很慢怎么办？

A: 可以使用镜像源或手动下载模型文件放到`models/`目录。

## 更新日志

### v1.0.0 (2024-XX-XX)

- 初始版本发布
- 支持目标检测、人脸识别、车牌识别、手势识别
- 提供Web界面和API接口
