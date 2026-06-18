# 使用指南

## 快速开始

### 1. 启动Web界面

```bash
streamlit run app.py
```

访问 http://localhost:8501 打开Web界面。

### 2. 启动API服务

```bash
uvicorn api:app --reload
```

访问 http://localhost:8000/docs 查看API文档。

## 功能使用

### 目标检测

```python
from src.detector import ObjectDetector

# 创建检测器
detector = ObjectDetector(model_path='yolov8n.pt')

# 检测图片
results = detector.detect_image('image.jpg')
print(f"检测到 {results['count']} 个物体")

# 检测视频
detector.detect_video('video.mp4', 'output.mp4')

# 实时检测
detector.detect_camera(camera_id=0)
```

### 人脸识别

```python
from src.face_detector import FaceDetector

# 创建检测器
detector = FaceDetector()

# 检测人脸
results = detector.detect_faces('photo.jpg')
print(f"检测到 {results['face_count']} 张人脸")

# 绘制结果
detector.draw_faces('photo.jpg', 'result.jpg')
```

### 车牌识别

```python
from src.plate_recognizer import LicensePlateRecognizer

# 创建识别器
recognizer = LicensePlateRecognizer()

# 检测车牌
results = recognizer.detect_plate('car.jpg')
print(f"检测到 {results['plate_count']} 个车牌")

# 识别车牌号码
results = recognizer.recognize_plate('car.jpg')
for plate in results['plates']:
    print(f"车牌号码: {plate['text']}")
```

### 手势识别

```python
from src.gesture_recognizer import GestureRecognizer

# 创建识别器
recognizer = GestureRecognizer()

# 识别手势
results = recognizer.recognize_gesture('hand.jpg')
print(f"检测到 {results['hand_count']} 只手")

for gesture in results['gestures']:
    print(f"手势: {gesture['gesture']}")
```

## 批量处理

```python
from src.batch_processor import BatchProcessor

# 创建处理器
processor = BatchProcessor()

# 批量检测
results = processor.batch_detect_images(
    'input_folder',
    'output_folder',
    detection_type='object'
)

# 导出结果
processor.export_results_to_json(results, 'results.json')
processor.export_results_to_csv(results, 'results.csv')
```

## 数据可视化

```python
from src.visualizer import Visualizer

# 创建可视化器
visualizer = Visualizer()

# 绘制统计图
chart_path = visualizer.plot_detection_stats(results)
print(f"统计图已生成: {chart_path}")
```

## 报告导出

```python
from src.report_generator import ReportGenerator

# 创建报告生成器
generator = ReportGenerator()

# 生成HTML报告
html_path = generator.generate_html_report(results, '检测报告')

# 生成CSV报告
csv_path = generator.generate_csv_report(results)

# 生成JSON报告
json_path = generator.generate_json_report(results)
```

## 模型训练

```python
from src.trainer import ModelTrainer

# 创建训练器
trainer = ModelTrainer('yolov8n.pt')

# 创建数据集配置
config_path = trainer.create_dataset_config(
    'data/custom_dataset',
    ['person', 'car', 'dog'],
    'configs/custom_dataset.yaml'
)

# 训练模型
results = trainer.train(
    data_path=config_path,
    epochs=100,
    img_size=640,
    batch_size=16
)
```

## 视频处理

```python
from src.video_processor import VideoProcessor

# 创建处理器
processor = VideoProcessor()

# 处理视频
stats = processor.process_video(
    'input.mp4',
    'output.mp4',
    detection_type='object'
)
```

## 配置管理

```python
from src.config import ConfigManager

# 创建配置管理器
manager = ConfigManager('config.yaml')

# 加载配置
config = manager.load()

# 获取配置值
model_path = config.model.detection_model
api_port = config.api.port

# 设置配置值
manager.set('model.detection_model', 'yolov8s.pt')
manager.set('api.port', 8001)

# 保存配置
manager.save()
```

## API使用

### 目标检测

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### 人脸识别

```bash
curl -X POST "http://localhost:8000/api/v1/face/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@photo.jpg"
```

### 车牌识别

```bash
curl -X POST "http://localhost:8000/api/v1/plate/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@car.jpg"
```

### 手势识别

```bash
curl -X POST "http://localhost:8000/api/v1/gesture/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand.jpg"
```

## 实时检测

```bash
# 实时目标检测
python examples/realtime_detection.py

# 实时人脸识别
python examples/realtime_face.py

# 实时手势识别
python examples/realtime_gesture.py

# 实时综合检测
python examples/realtime_all.py
```
