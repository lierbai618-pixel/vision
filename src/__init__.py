"""
智能视觉系统 - 源代码包（延迟导入，加速启动）
"""

_LAZY_IMPORTS = {
    'ObjectDetector': ('.detector', 'ObjectDetector'),
    'VideoProcessor': ('.video_processor', 'VideoProcessor'),
    'Visualizer': ('.visualizer', 'Visualizer'),
    'ModelTrainer': ('.trainer', 'ModelTrainer'),
    'ReportGenerator': ('.report_generator', 'ReportGenerator'),
    'ConfigManager': ('.config', 'ConfigManager'),
    'load_config': ('.config', 'load_config'),
    'create_default_config': ('.config', 'create_default_config'),
    'FaceDetector': ('.face_detector', 'FaceDetector'),
    'GestureRecognizer': ('.gesture_recognizer', 'GestureRecognizer'),
    'LicensePlateRecognizer': ('.plate_recognizer', 'LicensePlateRecognizer'),
    'BatchProcessor': ('.batch_processor', 'BatchProcessor'),
    'ObjectTracker': ('.tracker', 'ObjectTracker'),
    'AuthManager': ('.auth', 'AuthManager'),
    'Database': ('.database', 'Database'),
    'RateLimiter': ('.resilience', 'RateLimiter'),
    'CircuitBreaker': ('.resilience', 'CircuitBreaker'),
    'HealthChecker': ('.resilience', 'HealthChecker'),
    'InputValidator': ('.security', 'InputValidator'),
    'TokenManager': ('.security', 'TokenManager'),
}

__version__ = '1.0.0'


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
