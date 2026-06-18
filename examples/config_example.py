"""
配置管理示例

演示如何使用配置管理功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager, create_default_config


def main():
    """配置管理示例"""

    print("=" * 50)
    print("配置管理示例")
    print("=" * 50)

    # 创建默认配置
    print("\n1. 创建默认配置")
    print("-" * 40)
    create_default_config('my_config.yaml')
    print("  配置文件已创建: my_config.yaml")

    # 加载配置
    print("\n2. 加载配置")
    print("-" * 40)
    manager = ConfigManager('my_config.yaml')
    config = manager.load()

    print(f"  模型路径: {config.model.detection_model}")
    print(f"  API端口: {config.api.port}")
    print(f"  Web端口: {config.web.port}")

    # 修改配置
    print("\n3. 修改配置")
    print("-" * 40)
    manager.set('model.detection_model', 'yolov8s.pt')
    manager.set('api.port', 8001)
    manager.set('web.port', 8502)

    print(f"  新模型路径: {config.model.detection_model}")
    print(f"  新API端口: {config.api.port}")
    print(f"  新Web端口: {config.web.port}")

    # 保存配置
    print("\n4. 保存配置")
    print("-" * 40)
    manager.save('my_config_modified.yaml')
    print("  配置已保存: my_config_modified.yaml")

    # 验证配置
    print("\n5. 验证配置")
    print("-" * 40)
    is_valid = manager.validate()
    print(f"  配置验证: {'通过' if is_valid else '失败'}")

    # 打印配置
    print("\n6. 打印配置")
    print("-" * 40)
    manager.print_config()

    # 清理
    import os
    os.remove('my_config.yaml')
    os.remove('my_config_modified.yaml')

    print("\n配置管理示例完成!")


if __name__ == '__main__':
    main()
