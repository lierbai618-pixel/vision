"""
综合测试模块

包含单元测试、集成测试、性能测试
"""

import pytest
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestObjectDetector:
    """目标检测器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.detector import ObjectDetector
        self.detector = ObjectDetector(model_path='yolov8n.pt')

    def test_init(self):
        """测试初始化"""
        assert self.detector is not None
        assert self.detector.model is not None

    def test_detect_image_url(self):
        """测试检测网络图片"""
        results = self.detector.detect_image(
            'https://ultralytics.com/images/bus.jpg',
            save_result=False,
            show_result=False
        )

        assert results is not None
        assert results['count'] > 0
        assert len(results['class_names']) > 0
        assert len(results['confidences']) > 0

    def test_detect_image_local(self):
        """测试检测本地图片"""
        test_image = "data/samples/bus.jpg"
        if not Path(test_image).exists():
            pytest.skip("测试图片不存在")

        results = self.detector.detect_image(
            test_image,
            save_result=False,
            show_result=False
        )

        assert results is not None
        assert results['count'] > 0

    def test_detect_batch(self):
        """测试批量检测"""
        test_folder = "data/samples"
        if not Path(test_folder).exists():
            pytest.skip("测试文件夹不存在")

        results = self.detector.detect_batch(test_folder, "test_batch_results")

        assert len(results) > 0
        assert all(r['count'] > 0 for r in results)


class TestFaceDetector:
    """人脸检测器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.face_detector import FaceDetector
        self.detector = FaceDetector()

    def test_init(self):
        """测试初始化"""
        assert self.detector is not None

    def test_detect_faces(self):
        """测试检测人脸"""
        test_image = "data/face/Celebrity Faces Dataset/Angelina Jolie/001_fe3347c0.jpg"
        if not Path(test_image).exists():
            pytest.skip("测试图片不存在")

        results = self.detector.detect_faces(test_image)

        assert results is not None
        assert 'face_count' in results
        assert results['face_count'] > 0


class TestGestureRecognizer:
    """手势识别器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.gesture_recognizer import GestureRecognizer
        self.recognizer = GestureRecognizer()

    def test_init(self):
        """测试初始化"""
        assert self.recognizer is not None

    def test_detect_hands(self):
        """测试检测手部"""
        test_image = "data/gesture/leapGestRecog/01/01_palm/frame_01_01_0001.png"
        if not Path(test_image).exists():
            pytest.skip("测试图片不存在")

        results = self.recognizer.detect_hands(test_image)

        assert results is not None
        assert 'hand_count' in results
        assert results['hand_count'] > 0


class TestBatchProcessor:
    """批量处理器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.batch_processor import BatchProcessor
        self.processor = BatchProcessor()

    def test_init(self):
        """测试初始化"""
        assert self.processor is not None

    def test_batch_detect_images(self):
        """测试批量检测"""
        test_folder = "data/samples"
        if not Path(test_folder).exists():
            pytest.skip("测试文件夹不存在")

        results = self.processor.batch_detect_images(
            test_folder,
            'test_batch_results',
            detection_type='object'
        )

        assert results is not None
        assert results['total_images'] > 0
        assert results['successful'] > 0


class TestVisualizer:
    """数据可视化器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.visualizer import Visualizer
        self.visualizer = Visualizer()

    def test_init(self):
        """测试初始化"""
        assert self.visualizer is not None

    def test_plot_detection_stats(self):
        """测试绘制统计图"""
        from src.detector import ObjectDetector

        detector = ObjectDetector()
        results = detector.detect_image(
            'https://ultralytics.com/images/bus.jpg',
            save_result=False,
            show_result=False
        )

        chart_path = self.visualizer.plot_detection_stats(results, 'test_stats.png')

        assert chart_path is not None
        assert Path(chart_path).exists()


class TestReportGenerator:
    """报告生成器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.report_generator import ReportGenerator
        self.generator = ReportGenerator()

    def test_init(self):
        """测试初始化"""
        assert self.generator is not None

    def test_generate_html_report(self):
        """测试生成HTML报告"""
        test_results = {
            'total_images': 10,
            'successful': 9,
            'failed': 1,
            'detection_type': 'object',
            'results': [
                {
                    'image_name': 'test1.jpg',
                    'count': 3,
                    'class_names': ['person', 'car', 'dog']
                }
            ]
        }

        report_path = self.generator.generate_html_report(test_results)

        assert report_path is not None
        assert Path(report_path).exists()


class TestConfigManager:
    """配置管理器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.config import ConfigManager
        self.manager = ConfigManager('test_config.yaml')

    def test_init(self):
        """测试初始化"""
        assert self.manager is not None

    def test_create_default(self):
        """测试创建默认配置"""
        self.manager.create_default()

        assert Path('test_config.yaml').exists()

    def test_load_config(self):
        """测试加载配置"""
        self.manager.create_default()
        config = self.manager.load()

        assert config is not None
        assert config.model is not None
        assert config.api is not None

    def test_validate_config(self):
        """测试验证配置"""
        self.manager.create_default()
        is_valid = self.manager.validate()

        assert is_valid is True

    def teardown_method(self):
        """测试后清理"""
        config_path = Path('test_config.yaml')
        if config_path.exists():
            config_path.unlink()


class TestAuthManager:
    """认证管理器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.auth import AuthManager
        self.auth = AuthManager('test_auth.json')

    def test_register(self):
        """测试注册"""
        result = self.auth.register('testuser', 'test@example.com', 'Password123')
        assert result['success'] is True

    def test_login(self):
        """测试登录"""
        self.auth.register('testuser', 'test@example.com', 'Password123')
        result = self.auth.login('testuser', 'Password123')
        assert result['success'] is True
        assert 'session_id' in result

    def test_get_user(self):
        """测试获取用户"""
        self.auth.register('testuser', 'test@example.com', 'Password123')
        login_result = self.auth.login('testuser', 'Password123')
        session_id = login_result['session_id']

        user_info = self.auth.get_user(session_id)
        assert user_info is not None
        assert user_info['username'] == 'testuser'

    def teardown_method(self):
        """测试后清理"""
        auth_file = Path('test_auth.json')
        if auth_file.exists():
            auth_file.unlink()


class TestDatabase:
    """数据库测试"""

    def setup_method(self):
        """测试前准备"""
        from src.database import Database
        self.db = Database('test.db')

    def test_insert_detection(self):
        """测试插入检测记录"""
        record_id = self.db.insert_detection(
            user_id='test_user',
            detection_type='object',
            image_path='test.jpg',
            result={'count': 3, 'classes': ['person', 'car', 'dog']}
        )

        assert record_id is not None

    def test_get_detections(self):
        """测试获取检测记录"""
        self.db.insert_detection(
            user_id='test_user',
            detection_type='object',
            image_path='test.jpg',
            result={'count': 3}
        )

        detections = self.db.get_detections(user_id='test_user')
        assert len(detections) > 0

    def test_insert_task(self):
        """测试插入任务"""
        task_id = self.db.insert_task(
            task_id='test_task_001',
            user_id='test_user',
            task_type='detection',
            input_data={'image_path': 'test.jpg'}
        )

        assert task_id == 'test_task_001'

    def test_update_task(self):
        """测试更新任务"""
        self.db.insert_task(
            task_id='test_task_001',
            user_id='test_user',
            task_type='detection',
            input_data={'image_path': 'test.jpg'}
        )

        self.db.update_task('test_task_001', 'completed', output_data={'count': 3})

        task = self.db.get_task('test_task_001')
        assert task['status'] == 'completed'

    def teardown_method(self):
        """测试后清理"""
        db_file = Path('test.db')
        if db_file.exists():
            db_file.unlink()


class TestResilience:
    """高可用模块测试"""

    def test_rate_limiter(self):
        """测试限流器"""
        from src.resilience import RateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_second=5, burst_size=5)
        limiter = RateLimiter(config)

        success_count = 0
        for _ in range(10):
            if limiter.acquire():
                success_count += 1

        assert success_count <= 5

    def test_circuit_breaker(self):
        """测试熔断器"""
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

    def test_health_checker(self):
        """测试健康检查"""
        from src.resilience import HealthChecker

        checker = HealthChecker()
        checker.register('test', lambda: True)

        result = checker.check()
        assert result['status'] == 'healthy'


class TestSecurity:
    """安全模块测试"""

    def test_validate_email(self):
        """测试邮箱验证"""
        from src.security import InputValidator

        assert InputValidator.validate_email('test@example.com') is True
        assert InputValidator.validate_email('invalid-email') is False

    def test_validate_username(self):
        """测试用户名验证"""
        from src.security import InputValidator

        assert InputValidator.validate_username('test_user') is True
        assert InputValidator.validate_username('ab') is False

    def test_validate_password(self):
        """测试密码验证"""
        from src.security import InputValidator

        result = InputValidator.validate_password('Password123')
        assert result['valid'] is True

        result = InputValidator.validate_password('weak')
        assert result['valid'] is False

    def test_token_manager(self):
        """测试令牌管理"""
        from src.security import TokenManager

        manager = TokenManager()
        token = manager.generate_token({'user_id': '123'})

        data = manager.verify_token(token)
        assert data is not None
        assert data['user_id'] == '123'


class TestPerformance:
    """性能测试"""

    def test_detection_speed(self):
        """测试检测速度"""
        from src.detector import ObjectDetector

        detector = ObjectDetector()

        start_time = time.perf_counter()
        for _ in range(10):
            results = detector.detect_image(
                'https://ultralytics.com/images/bus.jpg',
                save_result=False,
                show_result=False
            )
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10
        assert avg_time < 2.0  # 平均每张图片处理时间应小于2秒

    def test_batch_processing_speed(self):
        """测试批量处理速度"""
        from src.batch_processor import BatchProcessor

        processor = BatchProcessor()
        test_folder = "data/samples"

        if not Path(test_folder).exists():
            pytest.skip("测试文件夹不存在")

        start_time = time.perf_counter()
        results = processor.batch_detect_images(
            test_folder,
            'test_batch_results',
            detection_type='object'
        )
        end_time = time.perf_counter()

        total_time = end_time - start_time
        assert total_time < 30  # 批量处理时间应小于30秒


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
