"""
集成测试模块

测试各模块之间的集成
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDetectionPipeline:
    """检测流程集成测试"""

    def test_object_detection_pipeline(self):
        """测试目标检测流程"""
        from src.detector import ObjectDetector
        from src.visualizer import Visualizer
        from src.report_generator import ReportGenerator

        # 创建组件
        detector = ObjectDetector()
        visualizer = Visualizer()
        report_generator = ReportGenerator()

        # 测试检测
        test_image = "https://ultralytics.com/images/bus.jpg"
        results = detector.detect_image(test_image, save_result=False, show_result=False)

        assert results is not None
        assert 'count' in results
        assert results['count'] > 0

        # 测试可视化
        chart_path = visualizer.plot_detection_stats(results, 'test_stats.png')
        assert chart_path is not None

        # 测试报告生成
        results['image_name'] = 'bus.jpg'
        batch_results = {
            'total_images': 1,
            'successful': 1,
            'failed': 0,
            'detection_type': 'object',
            'results': [results]
        }
        report_path = report_generator.generate_html_report(batch_results, '测试报告')
        assert report_path is not None

    def test_batch_processing_pipeline(self):
        """测试批量处理流程"""
        from src.batch_processor import BatchProcessor
        from src.visualizer import Visualizer

        # 创建组件
        processor = BatchProcessor()
        visualizer = Visualizer()

        # 测试批量处理
        test_folder = "data/samples"
        if not Path(test_folder).exists():
            pytest.skip("测试文件夹不存在")

        results = processor.batch_detect_images(
            test_folder,
            'test_batch_results',
            detection_type='object'
        )

        assert results is not None
        assert results['total_images'] > 0
        assert results['successful'] > 0

        # 测试可视化
        chart_path = visualizer.plot_batch_results(results, 'test_batch_stats.png')
        assert chart_path is not None


class TestAPIIntegration:
    """API集成测试"""

    def test_api_health_check(self):
        """测试API健康检查"""
        import requests

        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'
        except Exception:
            pytest.skip("API服务未启动")

    def test_api_detection(self):
        """测试API检测"""
        import requests

        try:
            test_image = "data/samples/bus.jpg"
            if not Path(test_image).exists():
                pytest.skip("测试图片不存在")

            with open(test_image, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    'http://localhost:8000/api/v1/detect',
                    files=files,
                    timeout=30
                )

            assert response.status_code == 200
            data = response.json()
            assert 'count' in data
            assert data['count'] > 0

        except Exception:
            pytest.skip("API服务未启动")


class TestDatabaseIntegration:
    """数据库集成测试"""

    def test_detection_storage(self):
        """测试检测结果存储"""
        from src.database import Database

        db = Database('test_integration.db')

        # 插入检测记录
        result = {
            'count': 3,
            'class_names': ['person', 'car', 'dog'],
            'confidences': [0.95, 0.87, 0.92]
        }

        record_id = db.insert_detection(
            user_id='test_user',
            detection_type='object',
            image_path='test.jpg',
            result=result
        )

        assert record_id is not None

        # 获取检测记录
        detections = db.get_detections(user_id='test_user')
        assert len(detections) > 0
        assert detections[0]['detection_type'] == 'object'

        # 清理
        import os
        os.remove('test_integration.db')

    def test_task_management(self):
        """测试任务管理"""
        from src.database import Database

        db = Database('test_integration.db')

        # 创建任务
        task_id = db.insert_task(
            task_id='test_task_001',
            user_id='test_user',
            task_type='detection',
            input_data={'image_path': 'test.jpg'}
        )

        assert task_id == 'test_task_001'

        # 更新任务状态
        db.update_task(task_id, 'completed', output_data={'count': 3})

        # 获取任务
        task = db.get_task(task_id)
        assert task is not None
        assert task['status'] == 'completed'

        # 清理
        import os
        os.remove('test_integration.db')


class TestAuthIntegration:
    """认证集成测试"""

    def test_user_registration_and_login(self):
        """测试用户注册和登录"""
        from src.auth import AuthManager

        auth = AuthManager('test_auth.json')

        # 注册用户
        result = auth.register('testuser', 'test@example.com', 'password123')
        assert result['success'] is True

        # 登录
        result = auth.login('testuser', 'password123')
        assert result['success'] is True
        assert 'session_id' in result

        # 获取用户信息
        session_id = result['session_id']
        user_info = auth.get_user(session_id)
        assert user_info is not None
        assert user_info['username'] == 'testuser'

        # 登出
        result = auth.logout(session_id)
        assert result['success'] is True

        # 清理
        import os
        os.remove('test_auth.json')


class TestResilienceIntegration:
    """高可用集成测试"""

    def test_rate_limiter_integration(self):
        """测试限流器集成"""
        from src.resilience import RateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_second=5, burst_size=5)
        limiter = RateLimiter(config)

        # 测试限流
        success_count = 0
        for _ in range(10):
            if limiter.acquire():
                success_count += 1

        assert success_count <= 5

    def test_circuit_breaker_integration(self):
        """测试熔断器集成"""
        from src.resilience import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        # 触发熔断
        for _ in range(3):
            try:
                breaker.call(lambda: 1/0)
            except:
                pass

        assert breaker.state == 'open'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
