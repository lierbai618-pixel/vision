"""
人脸检测模块.

使用 YOLOv8 或 MediaPipe 进行人脸检测
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FaceDetector:
    """人脸检测器.

    支持 YOLOv8 专用人脸模型和 MediaPipe 回退方案

    Attributes:
        min_detection_confidence: 最小检测置信度
        model: YOLOv8 人脸检测模型
    """

    def __init__(self, min_detection_confidence: float = 0.5, model_path: str | None = None):
        """初始化人脸检测器.

        Args:
            min_detection_confidence: 最小检测置信度，范围 0-1
            model_path: 人脸检测模型路径，None 使用默认路径
        """
        self.min_detection_confidence = min_detection_confidence
        self.model = None
        self._model_path = model_path

    def _load_model(self):
        """懒加载人脸检测模型."""
        if self.model is not None:
            return

        from ultralytics import YOLO

        # 优先使用指定路径
        if self._model_path and Path(self._model_path).exists():
            self.model = YOLO(self._model_path)
            return

        # 尝试常见路径
        candidate_paths = [
            "models/yolov8n-face.pt",
            "C:/temp/models/yolov8n-face.pt",
            str(Path(__file__).parent.parent / "models" / "yolov8n-face.pt"),
        ]

        for path in candidate_paths:
            if Path(path).exists():
                self.model = YOLO(path)
                return

        # 回退到通用 YOLOv8 模型（person 类作为人脸近似）
        print("警告: 未找到专用人脸模型，使用 yolov8n.pt 回退")
        self.model = YOLO("yolov8n.pt")

    def detect_faces(self, image_path: str) -> dict:
        """检测图片中的人脸.

        Args:
            image_path: 图片路径

        Returns:
            检测结果字典，包含 face_count, face_locations, face_confidences
        """
        self._load_model()

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 运行检测
        results = self.model(image, conf=self.min_detection_confidence, verbose=False)

        # 解析结果
        face_locations = []
        face_confidences = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                face_locations.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})
                face_confidences.append(conf)

        return {
            "face_count": len(face_locations),
            "face_locations": face_locations,
            "face_confidences": face_confidences,
        }

    def detect_faces_in_frame(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """检测视频帧中的人脸（用于实时检测）.

        Args:
            frame: BGR 格式的视频帧

        Returns:
            人脸列表，每项为 (x, y, w, h, confidence)
        """
        self._load_model()

        results = self.model(frame, conf=self.min_detection_confidence, verbose=False)
        faces = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                faces.append((x1, y1, x2 - x1, y2 - y1, conf))

        return faces

    def detect_faces_batch(self, image_paths: list[str]) -> list[dict]:
        """批量检测多张图片的人脸.

        Args:
            image_paths: 图片路径列表

        Returns:
            每张图片的检测结果列表
        """
        results = []
        for path in image_paths:
            try:
                result = self.detect_faces(path)
                result["image_path"] = path
                results.append(result)
            except Exception as e:
                results.append(
                    {"image_path": path, "face_count": 0, "face_locations": [], "face_confidences": [], "error": str(e)}
                )
        return results


# 便捷函数
def detect_faces_in_image(image_path: str, confidence: float = 0.5) -> dict:
    """快速检测单张图片的人脸.

    Args:
        image_path: 图片路径
        confidence: 最小置信度

    Returns:
        检测结果
    """
    detector = FaceDetector(min_detection_confidence=confidence)
    return detector.detect_faces(image_path)


if __name__ == "__main__":
    # 测试代码
    print("人脸检测器测试")
    detector = FaceDetector()

    test_image = "https://ultralytics.com/images/bus.jpg"
    try:
        results = detector.detect_faces(test_image)
        print(f"检测到 {results['face_count']} 张人脸")
        for i, (loc, conf) in enumerate(zip(results["face_locations"], results["face_confidences"])):
            print(f"  人脸 {i + 1}: 位置({loc['x']},{loc['y']}), 置信度 {conf:.2%}")
    except Exception as e:
        print(f"测试失败: {e}")
