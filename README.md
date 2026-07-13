# YOLOv8 智能视觉系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## ✨ 功能特点

| 功能        | 说明                                               |
| ----------- | -------------------------------------------------- |
| 📹 实时监测 | 摄像头实时检测，支持截图、告警                     |
| 📷 目标检测 | YOLOv8 检测 80+ 物体类别，人用绿色框，其他用蓝色框 |
| 👤 人脸识别 | YOLOv8 专用人脸检测模型                            |
| ✋ 手势识别 | MediaPipe HandLandmarker，支持多种手势             |
| 😷 口罩检测 | 检测佩戴状态（正确/不规范/未佩戴）                 |
| 🧠 模型训练 | 支持自定义数据集训练                               |

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动应用

```bash
streamlit run app.py
```

访问 http://localhost:8501

## 📦 模型下载

首次运行会自动下载 YOLOv8 模型。如需手动下载：

```bash
# YOLOv8 目标检测模型
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

# 人脸检测模型（可选）
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-face.pt
```

## 🛠️ 技术栈

- **Python 3.8+**
- **YOLOv8** - 目标检测
- **MediaPipe** - 人脸/手势识别
- **OpenCV** - 图像处理
- **Streamlit** - Web 界面

## 📁 项目结构

```
├── app.py              # 主程序
├── api.py              # API 接口
├── src/                # 核心代码
├── configs/            # 配置文件
├── models/             # 模型目录
├── data/               # 数据目录
└── requirements.txt    # 依赖列表
```

## 📄 License

MIT License
