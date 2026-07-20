"""
智能视觉系统 - 单元测试.

测试所有核心模块
"""

import sys
from pathlib import Path

import pytest

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestObjectDetector:
    """目标检测器测试."""

    def setup_method(self):
        """测试前准备."""
        from src.detector import ObjectDetector

        self.detector = ObjectDetector(model_path="yolov8n.pt", conf_threshold=0.5)

    def test_init(self):
        """测试初始化."""
        assert self.detector is not None
        assert self.detector.conf_threshold == 0.5

    def test_detect_image_url(self):
        """测试检测网络图片."""
        results = self.detector.detect_image(
            "https://ultralytics.com/images/bus.jpg", save_result=False, show_result=False
        )

        assert results is not None
        assert "boxes" in results
        assert "confidences" in results
        assert "class_ids" in results
        assert "class_names" in results
        assert "count" in results
        assert results["count"] > 0


class TestFaceDetector:
    """人脸检测器测试."""

    def setup_method(self):
        """测试前准备."""
        from src.face_detector import FaceDetector

        self.detector = FaceDetector(min_detection_confidence=0.5)

    def test_init(self):
        """测试初始化."""
        assert self.detector is not None

    def test_detect_faces(self):
        """测试检测人脸."""
        # 准备测试图片
        test_image = "data/face/Celebrity Faces Dataset/Angelina Jolie/001_fe3347c0.jpg"
        if not Path(test_image).exists():
            pytest.skip("测试图片不存在")

        results = self.detector.detect_faces(test_image)

        assert results is not None
        assert "face_count" in results
        assert "face_locations" in results
        assert "face_confidences" in results
        assert results["face_count"] > 0


class TestGestureRecognizer:
    """手势识别器测试."""

    def setup_method(self):
        """测试前准备."""
        from src.gesture_recognizer import GestureRecognizer

        self.recognizer = GestureRecognizer()

    def test_init(self):
        """测试初始化."""
        assert self.recognizer is not None

    def test_detect_hands(self):
        """测试检测手部."""
        # 准备测试图片
        test_image = "data/gesture/leapGestRecog/01/01_palm/frame_01_01_0001.png"
        if not Path(test_image).exists():
            pytest.skip("测试图片不存在")

        results = self.recognizer.detect_hands(test_image)

        assert results is not None
        assert "hand_count" in results
        assert "hands" in results
        assert results["hand_count"] > 0


class TestBatchProcessor:
    """批量处理器测试."""

    def setup_method(self):
        """测试前准备."""
        from src.batch_processor import BatchProcessor

        self.processor = BatchProcessor()

    def test_init(self):
        """测试初始化."""
        assert self.processor is not None

    def test_batch_detect_images(self):
        """测试批量检测."""
        test_folder = "data/samples"
        if not Path(test_folder).exists():
            pytest.skip("测试文件夹不存在")

        results = self.processor.batch_detect_images(test_folder, "batch_results", detection_type="object")

        assert results is not None
        assert "total_images" in results
        assert "successful" in results
        assert "failed" in results
        assert results["total_images"] > 0


class TestVisualizer:
    """数据可视化器测试."""

    def setup_method(self):
        """测试前准备."""
        from src.visualizer import Visualizer

        self.visualizer = Visualizer()

    def test_init(self):
        """测试初始化."""
        assert self.visualizer is not None

    def test_plot_detection_stats(self):
        """测试绘制统计图."""
        test_results = {
            "count": 5,
            "class_names": ["person", "car", "person", "dog", "car"],
            "confidences": [0.95, 0.87, 0.92, 0.78, 0.89],
        }

        chart_path = self.visualizer.plot_detection_stats(test_results)

        assert chart_path is not None
        assert Path(chart_path).exists()


class TestReportGenerator:
    """报告生成器测试."""

    def setup_method(self):
        """测试前准备."""
        from src.report_generator import ReportGenerator

        self.generator = ReportGenerator()

    def test_init(self):
        """测试初始化."""
        assert self.generator is not None

    def test_generate_html_report(self):
        """测试生成HTML报告."""
        test_results = {
            "total_images": 10,
            "successful": 9,
            "failed": 1,
            "detection_type": "object",
            "results": [{"image_name": "test1.jpg", "count": 3, "class_names": ["person", "car", "dog"]}],
        }

        report_path = self.generator.generate_html_report(test_results)

        assert report_path is not None
        assert Path(report_path).exists()


class TestConfigManager:
    """配置管理器测试."""

    def setup_method(self):
        """测试前准备."""
        from src.config import ConfigManager

        self.manager = ConfigManager("test_config.yaml")

    def test_init(self):
        """测试初始化."""
        assert self.manager is not None

    def test_create_default(self):
        """测试创建默认配置."""
        self.manager.create_default()

        assert Path("test_config.yaml").exists()

    def test_load_config(self):
        """测试加载配置."""
        self.manager.create_default()
        config = self.manager.load()

        assert config is not None
        assert config.model is not None
        assert config.api is not None

    def test_validate_config(self):
        """测试验证配置."""
        self.manager.create_default()
        is_valid = self.manager.validate()

        assert is_valid is True

    def teardown_method(self):
        """测试后清理."""
        config_path = Path("test_config.yaml")
        if config_path.exists():
            config_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
