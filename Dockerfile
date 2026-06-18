# 智能视觉系统 - Docker配置

# 使用Python 3.10基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p models data/samples results reports visualizations

# 下载YOLOv8模型
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 下载MediaPipe模型
RUN python -c "
import urllib.request
import os
os.makedirs('models', exist_ok=True)
# 下载人脸检测模型
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
    'models/face_detector.tflite'
)
# 下载手部检测模型
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'models/hand_landmarker.task'
)
"

# 暴露端口
EXPOSE 8000 8501

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
