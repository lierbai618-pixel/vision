"""
API客户端示例.

演示如何使用API接口进行检测
"""

from pathlib import Path

import requests


class VisionAPIClient:
    """视觉系统API客户端.

    Attributes:
        base_url: API基础URL
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """初始化API客户端.

        Args:
            base_url: API基础URL
        """
        self.base_url = base_url

    def health_check(self) -> dict:
        """健康检查.

        Returns:
            健康状态
        """
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def detect_object(self, image_path: str, conf: float = 0.5) -> dict:
        """目标检测.

        Args:
            image_path: 图片路径
            conf: 置信度阈值

        Returns:
            检测结果
        """
        with open(image_path, "rb") as f:
            files = {"file": f}
            params = {"conf": conf}
            response = requests.post(f"{self.base_url}/api/v1/detect", files=files, params=params)
        return response.json()

    def detect_face(self, image_path: str) -> dict:
        """人脸识别.

        Args:
            image_path: 图片路径

        Returns:
            检测结果
        """
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{self.base_url}/api/v1/face/detect", files=files)
        return response.json()

    def detect_plate(self, image_path: str) -> dict:
        """车牌识别.

        Args:
            image_path: 图片路径

        Returns:
            检测结果
        """
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{self.base_url}/api/v1/plate/detect", files=files)
        return response.json()

    def detect_gesture(self, image_path: str) -> dict:
        """手势识别.

        Args:
            image_path: 图片路径

        Returns:
            检测结果
        """
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{self.base_url}/api/v1/gesture/detect", files=files)
        return response.json()

    def list_models(self) -> list:
        """列出可用模型.

        Returns:
            模型列表
        """
        response = requests.get(f"{self.base_url}/api/v1/models")
        return response.json()

    def list_classes(self) -> list:
        """列出支持的类别.

        Returns:
            类别列表
        """
        response = requests.get(f"{self.base_url}/api/v1/classes")
        return response.json()


def main():
    """API客户端示例."""
    print("=" * 50)
    print("API客户端示例")
    print("=" * 50)

    # 创建客户端
    client = VisionAPIClient()

    # 健康检查
    print("\n1. 健康检查")
    print("-" * 40)
    try:
        health = client.health_check()
        print(f"  状态: {health['status']}")
        print(f"  版本: {health['version']}")
    except Exception as e:
        print(f"  错误: {e}")
        print("  请确保API服务已启动: uvicorn api:app --reload")
        return

    # 列出模型
    print("\n2. 列出模型")
    print("-" * 40)
    models = client.list_models()
    for model in models:
        print(f"  - {model}")

    # 列出类别
    print("\n3. 列出类别")
    print("-" * 40)
    classes = client.list_classes()
    print(f"  共 {len(classes)} 个类别")

    # 目标检测
    print("\n4. 目标检测")
    print("-" * 40)
    test_image = "data/samples/bus.jpg"
    if Path(test_image).exists():
        result = client.detect_object(test_image)
        print(f"  检测到 {result['count']} 个物体")
        for class_name in result["class_names"]:
            print(f"    - {class_name}")
    else:
        print(f"  测试图片不存在: {test_image}")

    print("\nAPI客户端示例完成!")


if __name__ == "__main__":
    main()
