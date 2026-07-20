# 更新日志

本项目的所有重要更改都将记录在此文件。

格式基于[Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循[语义化版本控制](https://semver.org/lang/zh-CN/)。

## [2.0.0] - 2026-06-17

### 新增

- 自定义物品检测训练（custom_items 数据集，90类合并配置）
- 自动数据集生成工具（generate_dataset.py）
- 多数据集合并工具（merge_datasets.py）
- 统一训练脚本（train_unified.py）
- 自定义物品训练脚本（train_custom_items.py）
- 带数据集的训练脚本（train_with_dataset.py）
- 自定义配置文件（custom_items.yaml, merged_90class.yaml）

### 变更

- 优化 Streamlit Web 界面（app.py 重构）
- 改进模型训练流程

### 技术栈

- Python 3.8+
- PyTorch + YOLOv8
- Streamlit Web 界面

## [1.0.0] - 2024-XX-XX

### 新增

- 目标检测功能（YOLOv8）
- 人脸识别功能（MediaPipe）
- 车牌识别功能（EasyOCR）
- 手势识别功能（MediaPipe）
- 目标跟踪功能（ByteTrack）
- 图像分割功能（YOLOv8）
- 图像分类功能（YOLOv8）
- OCR文字识别功能（EasyOCR）
- 图像增强功能
- 批量处理功能
- 数据可视化功能
- 视频处理功能
- 模型训练功能
- 报告导出功能
- 性能优化功能
- 用户认证系统
- WebSocket实时通信
- 日志系统
- 缓存系统
- 配置管理
- Streamlit Web界面
- FastAPI接口
- Docker部署支持
- CI/CD流程
- 单元测试

### 技术栈

- Python 3.8+
- PyTorch
- YOLOv8
- MediaPipe
- EasyOCR
- OpenCV
- Streamlit
- FastAPI
- Docker

## [0.1.0] - 2024-XX-XX

### 新增

- 项目初始化
- 基础架构搭建

---

## 版本说明

### 版本号格式

- 主版本号：不兼容的API修改
- 次版本号：向下兼容的功能性新增
- 修订号：向下兼容的问题修正

### 变更类型

- **新增**：新功能
- **变更**：对现有功能的变更
- **废弃**：已经不建议使用，未来会移除
- **移除**：已经移除的功能
- **修复**：任何bug修复
- **安全**：安全问题修复
