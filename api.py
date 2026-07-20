"""
智能视觉系统 - FastAPI接口（可选）.

独立的RESTful API服务，供第三方系统调用。
非必需组件，主系统通过 app.py (Streamlit) 提供完整功能。

启动方式: python api.py 或 uvicorn api:app --port 8000
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.batch_processor import BatchProcessor

# 导入各个模块
from src.detector import ObjectDetector
from src.face_detector import FaceDetector
from src.gesture_recognizer import GestureRecognizer
from src.plate_recognizer import LicensePlateRecognizer
from src.report_generator import ReportGenerator

# 创建FastAPI应用
app = FastAPI(
    title="智能视觉系统API",
    description="基于YOLOv8和MediaPipe的智能视觉分析系统API接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器实例
detector = None
face_detector = None
plate_recognizer = None
gesture_recognizer = None
batch_processor = None
report_generator = None


# 数据模型
class DetectionResult(BaseModel):
    """检测结果模型."""

    boxes: list[list[float]]
    confidences: list[float]
    class_ids: list[int]
    class_names: list[str]
    count: int


class FaceResult(BaseModel):
    """人脸检测结果模型."""

    face_count: int
    face_locations: list[dict]
    face_confidences: list[float]


class PlateResult(BaseModel):
    """车牌识别结果模型."""

    plate_count: int
    plates: list[dict]


class GestureResult(BaseModel):
    """手势识别结果模型."""

    hand_count: int
    gestures: list[dict]


class HealthResponse(BaseModel):
    """健康检查响应."""

    status: str
    version: str
    modules: dict[str, bool]


class ErrorResponse(BaseModel):
    """错误响应."""

    error: str
    detail: str


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化检测器."""
    global detector, face_detector, plate_recognizer, gesture_recognizer
    global batch_processor, report_generator

    print("正在初始化检测器...")

    # 初始化目标检测器
    detector = ObjectDetector(model_path="yolov8n.pt", conf_threshold=0.5, iou_threshold=0.45)
    print("  ✓ 目标检测器初始化完成")

    # 初始化人脸检测器
    face_detector = FaceDetector(min_detection_confidence=0.5)
    print("  ✓ 人脸检测器初始化完成")

    # 初始化车牌识别器
    plate_recognizer = LicensePlateRecognizer(ocr_languages=["en", "ch_sim"])
    print("  ✓ 车牌识别器初始化完成")

    # 初始化手势识别器
    gesture_recognizer = GestureRecognizer()
    print("  ✓ 手势识别器初始化完成")

    # 初始化批量处理器
    batch_processor = BatchProcessor()
    print("  ✓ 批量处理器初始化完成")

    # 初始化报告生成器
    report_generator = ReportGenerator()
    print("  ✓ 报告生成器初始化完成")

    print("所有检测器初始化完成!")


@app.get("/", response_model=dict)
async def root():
    """根路径."""
    return {"message": "智能视觉系统API", "version": "1.0.0", "docs": "/docs", "redoc": "/redoc"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "modules": {
            "detector": detector is not None,
            "face_detector": face_detector is not None,
            "plate_recognizer": plate_recognizer is not None,
            "gesture_recognizer": gesture_recognizer is not None,
            "batch_processor": batch_processor is not None,
            "report_generator": report_generator is not None,
        },
    }


# ==================== 目标检测接口 ====================


@app.post("/api/v1/detect", response_model=DetectionResult, tags=["目标检测"])
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="置信度阈值"),
    model: str = Query("yolov8n.pt", description="模型名称"),
):
    """检测上传的图片.

    - **file**: 图片文件（支持jpg、png、bmp格式）
    - **conf**: 置信度阈值（0-1）
    - **model**: 模型名称
    """
    # 检查文件类型
    allowed_types = ["image/jpeg", "image/png", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")

    # 保存临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 检测
        results = detector.detect_image(tmp_path, save_result=False, show_result=False)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/v1/detect/batch", response_model=list[DetectionResult], tags=["目标检测"])
async def detect_batch(
    files: list[UploadFile] = File(...), conf: float = Query(0.5, ge=0.0, le=1.0, description="置信度阈值")
):
    """批量检测多张图片.

    - **files**: 图片文件列表
    - **conf**: 置信度阈值（0-1）
    """
    results = []

    for file in files:
        # 检查文件类型
        allowed_types = ["image/jpeg", "image/png", "image/bmp"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")

        # 保存临时文件
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # 检测
            result = detector.detect_image(tmp_path, save_result=False, show_result=False)
            results.append(result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return results


# ==================== 人脸识别接口 ====================


@app.post("/api/v1/face/detect", response_model=FaceResult, tags=["人脸识别"])
async def detect_face(
    file: UploadFile = File(...), min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="最小置信度")
):
    """检测图片中的人脸.

    - **file**: 图片文件
    - **min_confidence**: 最小置信度（0-1）
    """
    # 检查文件类型
    allowed_types = ["image/jpeg", "image/png", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")

    # 保存临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 检测
        results = face_detector.detect_faces(tmp_path)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==================== 车牌识别接口 ====================


@app.post("/api/v1/plate/detect", response_model=PlateResult, tags=["车牌识别"])
async def detect_plate(
    file: UploadFile = File(...), conf: float = Query(0.5, ge=0.0, le=1.0, description="置信度阈值")
):
    """检测图片中的车牌.

    - **file**: 图片文件
    - **conf**: 置信度阈值（0-1）
    """
    # 检查文件类型
    allowed_types = ["image/jpeg", "image/png", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")

    # 保存临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 检测
        results = plate_recognizer.detect_plate(tmp_path)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/v1/plate/recognize", response_model=PlateResult, tags=["车牌识别"])
async def recognize_plate(
    file: UploadFile = File(...), conf: float = Query(0.5, ge=0.0, le=1.0, description="置信度阈值")
):
    """识别图片中的车牌号码.

    - **file**: 图片文件
    - **conf**: 置信度阈值（0-1）
    """
    # 检查文件类型
    allowed_types = ["image/jpeg", "image/png", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")

    # 保存临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 识别
        results = plate_recognizer.recognize_plate(tmp_path)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==================== 手势识别接口 ====================


@app.post("/api/v1/gesture/detect", response_model=GestureResult, tags=["手势识别"])
async def detect_gesture(
    file: UploadFile = File(...), min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="最小置信度")
):
    """检测图片中的手势.

    - **file**: 图片文件
    - **min_confidence**: 最小置信度（0-1）
    """
    # 检查文件类型
    allowed_types = ["image/jpeg", "image/png", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")

    # 保存临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 检测
        results = gesture_recognizer.recognize_gesture(tmp_path)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==================== 报告导出接口 ====================


@app.post("/api/v1/report/generate", tags=["报告导出"])
async def generate_report(
    results: dict,
    title: str = Query("检测报告", description="报告标题"),
    format: str = Query("html", description="报告格式，支持html、csv、json"),
):
    """生成检测报告.

    - **results**: 检测结果
    - **title**: 报告标题
    - **format**: 报告格式
    """
    try:
        if format == "html":
            report_path = report_generator.generate_html_report(results, title)
        elif format == "csv":
            report_path = report_generator.generate_csv_report(results)
        elif format == "json":
            report_path = report_generator.generate_json_report(results)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")

        return FileResponse(
            report_path,
            media_type="text/html" if format == "HTML" else "text/csv" if format == "CSV" else "application/json",
            filename=Path(report_path).name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 模型信息接口 ====================


@app.get("/api/v1/models", response_model=list[str], tags=["模型信息"])
async def list_models():
    """列出可用模型."""
    return ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]


@app.get("/api/v1/classes", response_model=list[str], tags=["模型信息"])
async def list_classes():
    """列出支持的检测类别."""
    return [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


@app.get("/api/v1/gestures", response_model=list[str], tags=["模型信息"])
async def list_gestures():
    """列出支持的手势类别."""
    return ["拳头", "张开手掌", "食指指向前方", "耶/剪刀手", "竖起拇指", "点赞", "小指", "摇滚手势", "其他手势"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
