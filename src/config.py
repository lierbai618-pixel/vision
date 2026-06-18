"""
配置管理模块

提供配置文件的读取、写入、验证功能
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """模型配置"""
    detection_model: str = 'yolov8n.pt'
    face_confidence: float = 0.5
    plate_confidence: float = 0.5
    gesture_confidence: float = 0.5


@dataclass
class WebConfig:
    """Web界面配置"""
    host: str = '0.0.0.0'
    port: int = 8501
    debug: bool = False


@dataclass
class PathConfig:
    """路径配置"""
    data_dir: str = 'data'
    model_dir: str = 'models'
    result_dir: str = 'results'
    report_dir: str = 'reports'
    log_dir: str = 'logs'


@dataclass
class DatasetConfig:
    """数据集配置"""
    train_dir: str = 'train'
    val_dir: str = 'val'
    test_dir: str = 'test'
    image_extensions: list = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])


@dataclass
class ApiConfig:
    """API 配置"""
    host: str = '0.0.0.0'
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ['*'])


@dataclass
class AppConfig:
    """应用配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    web: WebConfig = field(default_factory=WebConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """从字典创建配置"""
        config = cls()
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'web' in data:
            config.web = WebConfig(**data['web'])
        if 'api' in data:
            config.api = ApiConfig(**data['api'])
        if 'paths' in data:
            config.paths = PathConfig(**data['paths'])
        if 'dataset' in data:
            config.dataset = DatasetConfig(**data['dataset'])
        return config


class ConfigManager:
    """
    配置管理器

    支持读取、写入、验证配置文件

    Attributes:
        config_path: 配置文件路径
        config: 应用配置
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = AppConfig()

        # 如果配置文件存在，加载配置
        if self.config_path.exists():
            self.load()

    def load(self) -> AppConfig:
        """
        加载配置文件

        Returns:
            应用配置
        """
        if not self.config_path.exists():
            print(f"配置文件不存在: {self.config_path}")
            return self.config

        # 根据文件扩展名选择加载方式
        suffix = self.config_path.suffix.lower()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if suffix == '.yaml' or suffix == '.yml':
                    data = yaml.safe_load(f)
                elif suffix == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {suffix}")

            if data:
                self.config = AppConfig.from_dict(data)
                print(f"配置已加载: {self.config_path}")

        except Exception as e:
            print(f"加载配置文件失败: {e}")

        return self.config

    def save(self, config_path: Optional[str] = None) -> None:
        """
        保存配置文件

        Args:
            config_path: 配置文件路径，None使用默认路径
        """
        if config_path:
            save_path = Path(config_path)
        else:
            save_path = self.config_path

        # 创建目录
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为字典
        data = self.config.to_dict()

        # 根据文件扩展名选择保存方式
        suffix = save_path.suffix.lower()

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if suffix == '.yaml' or suffix == '.yml':
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                elif suffix == '.json':
                    json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    raise ValueError(f"不支持的配置文件格式: {suffix}")

            print(f"配置已保存: {save_path}")

        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config.to_dict()

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值

        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        data = self.config.to_dict()

        # 导航到目标位置
        current = data
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # 设置值
        current[keys[-1]] = value

        # 更新配置
        self.config = AppConfig.from_dict(data)

    def validate(self) -> bool:
        """
        验证配置

        Returns:
            是否有效
        """
        try:
            # 检查模型文件是否存在
            model_path = Path(self.config.model.detection_model)
            if not model_path.exists():
                print(f"警告: 模型文件不存在: {model_path}")

            # 检查数据目录
            data_dir = Path(self.config.paths.data_dir)
            if not data_dir.exists():
                print(f"警告: 数据目录不存在: {data_dir}")

            # 检查端口范围
            if not (1024 <= self.config.web.port <= 65535):
                print(f"警告: Web端口无效: {self.config.web.port}")
                return False

            return True

        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

    def create_default(self) -> None:
        """
        创建默认配置文件
        """
        self.config = AppConfig()
        self.save()
        print(f"默认配置已创建: {self.config_path}")

    def print_config(self) -> None:
        """
        打印配置信息
        """
        print("\n当前配置:")
        print("=" * 50)

        data = self.config.to_dict()
        for section, values in data.items():
            print(f"\n[{section}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {values}")

        print("=" * 50)


# 便捷函数
def load_config(config_path: str = 'config.yaml') -> AppConfig:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        应用配置
    """
    manager = ConfigManager(config_path)
    return manager.config


def create_default_config(config_path: str = 'config.yaml') -> None:
    """
    创建默认配置文件

    Args:
        config_path: 配置文件路径
    """
    manager = ConfigManager(config_path)
    manager.create_default()


if __name__ == '__main__':
    # 测试代码
    print("测试配置管理器...")

    # 创建配置管理器
    manager = ConfigManager('config.yaml')

    # 打印配置
    manager.print_config()

    # 创建默认配置
    manager.create_default()

    # 验证配置
    is_valid = manager.validate()
    print(f"\n配置验证: {'通过' if is_valid else '失败'}")
