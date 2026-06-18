# API文档

## 概述

智能视觉系统提供RESTful API接口，支持目标检测、人脸识别、车牌识别、手势识别等功能。

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API版本**: v1
- **认证方式**: Bearer Token

## 通用响应格式

### 成功响应

```json
{
  "success": true,
  "data": {...},
  "message": "操作成功"
}
```

### 错误响应

```json
{
  "success": false,
  "error": "错误信息",
  "detail": "详细错误信息"
}
```

## API端点

### 1. 健康检查

**GET** `/health`

检查系统健康状态。

**响应示例**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "modules": {
    "detector": true,
    "face_detector": true,
    "plate_recognizer": true,
    "gesture_recognizer": true
  }
}
```

### 2. 目标检测

**POST** `/api/v1/detect`

检测图片中的物体。

**参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| conf | float | 否 | 置信度阈值，默认0.5 |

**请求示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "conf=0.5"
```

**响应示例**:

```json
{
  "boxes": [[100, 200, 300, 400]],
  "confidences": [0.95],
  "class_ids": [0],
  "class_names": ["person"],
  "count": 1
}
```

### 3. 人脸识别

**POST** `/api/v1/face/detect`

检测图片中的人脸。

**参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| min_confidence | float | 否 | 最小置信度，默认0.5 |

**响应示例**:

```json
{
  "face_count": 1,
  "face_locations": [{"x": 100, "y": 200, "width": 150, "height": 150}],
  "face_confidences": [0.98]
}
```

### 4. 车牌识别

**POST** `/api/v1/plate/detect`

检测图片中的车牌。

**参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| conf | float | 否 | 置信度阈值，默认0.5 |

**响应示例**:

```json
{
  "plate_count": 1,
  "plates": [
    {
      "location": {"x": 100, "y": 200, "width": 200, "height": 50},
      "text": "京A12345",
      "confidence": 0.95
    }
  ]
}
```

### 5. 手势识别

**POST** `/api/v1/gesture/detect`

检测图片中的手势。

**参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| min_confidence | float | 否 | 最小置信度，默认0.5 |

**响应示例**:

```json
{
  "hand_count": 1,
  "gestures": [
    {
      "gesture": "张开手掌",
      "confidence": 0.92
    }
  ]
}
```

### 6. 获取模型列表

**GET** `/api/v1/models`

获取可用模型列表。

**响应示例**:

```json
[
  "yolov8n.pt",
  "yolov8s.pt",
  "yolov8m.pt",
  "yolov8l.pt",
  "yolov8x.pt"
]
```

### 7. 获取类别列表

**GET** `/api/v1/classes`

获取支持的检测类别。

**响应示例**:

```json
[
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane"
]
```

## 错误码

| 错误码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

## 使用示例

### Python

```python
import requests

# 目标检测
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/v1/detect',
        files=files
    )
    result = response.json()
    print(f"检测到 {result['count']} 个物体")
```

### JavaScript

```javascript
// 目标检测
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/v1/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`检测到 ${data.count} 个物体`);
});
```

## 注意事项

1. 图片大小限制：10MB
2. 支持的图片格式：jpg, jpeg, png, bmp
3. 请求频率限制：100次/分钟
4. 建议使用HTTPS连接
