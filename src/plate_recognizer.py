"""
车牌识别模块

使用 YOLOv8 检测车牌区域，使用 OCR 识别车牌号码
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class LicensePlateRecognizer:
    """
    车牌识别器

    支持车牌检测和号码识别

    Attributes:
        ocr_languages: OCR 语言列表
        model: YOLOv8 检测模型
    """

    def __init__(self, ocr_languages: Optional[List[str]] = None):
        """
        初始化车牌识别器

        Args:
            ocr_languages: OCR 语言列表，如 ['en', 'ch_sim']
        """
        self.ocr_languages = ocr_languages or ['en']
        self.model = None
        self._ocr_reader = None

    def _load_model(self):
        """懒加载检测模型"""
        if self.model is not None:
            return

        from ultralytics import YOLO

        # 查找车牌检测模型
        candidate_paths = [
            "models/license_plate.pt",
            str(Path(__file__).parent.parent / "models" / "license_plate.pt"),
        ]

        for path in candidate_paths:
            if Path(path).exists():
                self.model = YOLO(path)
                return

        # 回退到通用模型
        print("警告: 未找到专用车牌模型，使用 yolov8n.pt 回退")
        self.model = YOLO("yolov8n.pt")

    def _load_ocr(self):
        """懒加载 OCR 引擎"""
        if self._ocr_reader is not None:
            return

        try:
            import easyocr
            self._ocr_reader = easyocr.Reader(self.ocr_languages, gpu=False)
        except ImportError:
            print("警告: easyocr 未安装，车牌号码识别功能不可用")
            self._ocr_reader = None

    def detect_plate(self, image_path: str) -> Dict:
        """
        检测图片中的车牌

        Args:
            image_path: 图片路径

        Returns:
            检测结果字典，包含 plate_count, plates
        """
        self._load_model()

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 运行检测
        results = self.model(image, conf=0.5, verbose=False)

        plates = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                plates.append({
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'confidence': conf,
                    'text': '',
                })

        return {
            'plate_count': len(plates),
            'plates': plates
        }

    def recognize_plate(self, image_path: str) -> Dict:
        """
        识别图片中的车牌号码

        Args:
            image_path: 图片路径

        Returns:
            识别结果字典，包含 plate_count, plates（含 text 字段）
        """
        # 先检测车牌
        result = self.detect_plate(image_path)

        # 加载 OCR
        self._load_ocr()

        if self._ocr_reader is None:
            return result

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            return result

        # 对每个检测到的车牌区域进行 OCR
        for plate in result['plates']:
            bbox = plate['bbox']
            roi = image[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']]

            if roi.size == 0:
                continue

            try:
                ocr_results = self._ocr_reader.readtext(roi)
                if ocr_results:
                    # 取置信度最高的结果
                    best = max(ocr_results, key=lambda x: x[2])
                    plate['text'] = best[1]
                    plate['ocr_confidence'] = float(best[2])
            except Exception as e:
                plate['text'] = ''
                plate['ocr_error'] = str(e)

        return result


# 便捷函数
def recognize_plate_in_image(image_path: str) -> Dict:
    """
    快速识别单张图片的车牌

    Args:
        image_path: 图片路径

    Returns:
        识别结果
    """
    recognizer = LicensePlateRecognizer()
    return recognizer.recognize_plate(image_path)


if __name__ == '__main__':
    # 测试代码
    print("车牌识别器测试")
    recognizer = LicensePlateRecognizer()

    test_image = "https://ultralytics.com/images/bus.jpg"
    try:
        results = recognizer.detect_plate(test_image)
        print(f"检测到 {results['plate_count']} 个车牌")
        for i, plate in enumerate(results['plates']):
            print(f"  车牌 {i+1}: 置信度 {plate['confidence']:.2%}")
    except Exception as e:
        print(f"测试失败: {e}")
