"""
批量处理模块

支持批量图片检测、人脸检测、手势识别等
"""

import cv2
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class BatchProcessor:
    """
    批量处理器

    支持批量目标检测、人脸检测、车牌识别、手势识别

    Attributes:
        output_dir: 输出目录
    """

    # 支持的图片格式
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, output_dir: str = 'batch_results'):
        """
        初始化批量处理器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def batch_detect_images(
        self,
        input_folder: str,
        output_dir: Optional[str] = None,
        detection_type: str = 'object',
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.5
    ) -> Dict:
        """
        批量检测图片

        Args:
            input_folder: 输入图片文件夹
            output_dir: 输出目录，None 使用默认目录
            detection_type: 检测类型，支持 object/face/plate/gesture
            model_path: 模型路径（object 检测时使用）
            conf_threshold: 置信度阈值

        Returns:
            批量处理结果，包含 total_images, successful, failed, results
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise ValueError(f"文件夹不存在: {input_folder}")

        # 获取所有图片
        images = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS and f.is_file()
        ]

        if not images:
            raise ValueError(f"文件夹中没有图片: {input_folder}")

        # 设置输出目录
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"找到 {len(images)} 张图片，开始批量处理...")
        print(f"检测类型: {detection_type}")

        start_time = time.time()
        results = []
        successful = 0
        failed = 0

        for i, image_path in enumerate(images):
            print(f"处理 [{i+1}/{len(images)}]: {image_path.name}")
            try:
                result = self._detect_single(
                    str(image_path), detection_type, model_path, conf_threshold
                )
                result['image_name'] = image_path.name
                result['image_path'] = str(image_path)
                results.append(result)
                successful += 1
            except Exception as e:
                results.append({
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'error': str(e)
                })
                failed += 1

        total_time = time.time() - start_time

        batch_results = {
            'total_images': len(images),
            'successful': successful,
            'failed': failed,
            'detection_type': detection_type,
            'processing_time': total_time,
            'results': results
        }

        print(f"\n批量处理完成: 成功 {successful}, 失败 {failed}, 耗时 {total_time:.1f}s")
        return batch_results

    def _detect_single(
        self,
        image_path: str,
        detection_type: str,
        model_path: str,
        conf_threshold: float
    ) -> Dict:
        """
        单张图片检测

        Args:
            image_path: 图片路径
            detection_type: 检测类型
            model_path: 模型路径
            conf_threshold: 置信度阈值

        Returns:
            检测结果
        """
        if detection_type == 'object':
            from src.detector import ObjectDetector
            detector = ObjectDetector(model_path, conf_threshold)
            return detector.detect_image(image_path, save_result=False, show_result=False)

        elif detection_type == 'face':
            from src.face_detector import FaceDetector
            detector = FaceDetector(min_detection_confidence=conf_threshold)
            return detector.detect_faces(image_path)

        elif detection_type == 'plate':
            from src.plate_recognizer import LicensePlateRecognizer
            recognizer = LicensePlateRecognizer()
            return recognizer.detect_plate(image_path)

        elif detection_type == 'gesture':
            from src.gesture_recognizer import GestureRecognizer
            recognizer = GestureRecognizer()
            return recognizer.detect_hands(image_path)

        else:
            raise ValueError(f"不支持的检测类型: {detection_type}")


# 便捷函数
def batch_detect(
    input_folder: str,
    output_dir: str = 'batch_results',
    detection_type: str = 'object'
) -> Dict:
    """
    快速批量检测

    Args:
        input_folder: 输入图片文件夹
        output_dir: 输出目录
        detection_type: 检测类型

    Returns:
        批量处理结果
    """
    processor = BatchProcessor(output_dir)
    return processor.batch_detect_images(input_folder, output_dir, detection_type)


if __name__ == '__main__':
    # 测试代码
    print("批量处理器测试")
    processor = BatchProcessor()

    # 检查测试文件夹
    test_folder = "data/samples"
    if Path(test_folder).exists():
        results = processor.batch_detect_images(test_folder)
        print(f"处理完成: {results['successful']}/{results['total_images']} 成功")
    else:
        print(f"测试文件夹不存在: {test_folder}")
