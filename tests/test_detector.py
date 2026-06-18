"""
检测器测试模块

测试ObjectDetector类的功能
"""

import pytest
import sys
sys.path.append('..')

from src.detector import ObjectDetector


class TestObjectDetector:
    """ObjectDetector测试类"""

    def setup_method(self):
        """测试前准备"""
        self.detector = ObjectDetector(
            model_path='yolov8n.pt',
            conf_threshold=0.5,
            iou_threshold=0.45
        )

    def test_init(self):
        """测试初始化"""
        assert self.detector is not None
        assert self.detector.conf_threshold == 0.5
        assert self.detector.iou_threshold == 0.45

    def test_detect_image_url(self):
        """测试检测网络图片"""
        results = self.detector.detect_image(
            'https://ultralytics.com/images/bus.jpg',
            save_result=False,
            show_result=False
        )

        assert results is not None
        assert 'boxes' in results
        assert 'confidences' in results
        assert 'class_ids' in results
        assert 'class_names' in results
        assert 'count' in results
        assert results['count'] > 0

    def test_detect_image_local(self):
        """测试检测本地图片"""
        # 需要准备测试图片
        # results = self.detector.detect_image('test.jpg')
        # assert results is not None
        pass

    def test_parse_results(self):
        """测试结果解析"""
        # 需要模拟检测结果
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
