"""
人脸识别示例.

演示如何使用FaceDetector进行人脸检测和识别
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import urllib.request

from src.face_detector import FaceDetector


def download_face_image(url: str, filename: str) -> str:
    """下载测试图片."""
    filepath = Path("data/faces") / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        print(f"下载 {filename}...")
        urllib.request.urlretrieve(url, str(filepath))

    return str(filepath)


def main():
    """人脸识别示例."""
    # 创建检测器
    detector = FaceDetector(
        tolerance=0.6,
        model="hog",  # 使用HOG模型，CPU友好
    )

    print("=" * 50)
    print("人脸识别示例")
    print("=" * 50)

    # 测试图片URL（使用公开的人脸图片）
    test_images = [
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/250px-Gatto_europeo4.jpg",
            "cat.jpg",
        ),
    ]

    # 下载并检测
    for url, filename in test_images:
        print(f"\n检测: {filename}")
        print("-" * 40)

        try:
            filepath = download_face_image(url, filename)
            results = detector.detect_faces(filepath)

            print(f"  检测到 {results['face_count']} 张人脸")

            if results["face_count"] > 0:
                for i, location in enumerate(results["face_locations"]):
                    print(
                        f"  人脸 {i + 1}: x={location['x']}, y={location['y']}, "
                        f"宽={location['width']}, 高={location['height']}"
                    )
        except Exception as e:
            print(f"  错误: {e}")

    # 演示人脸比对功能
    print("\n" + "=" * 50)
    print("人脸比对功能演示")
    print("=" * 50)
    print("\n使用方法:")
    print("1. 添加已知人脸:")
    print("   detector.add_known_face('path/to/face.jpg', '张三')")
    print("\n2. 识别未知人脸:")
    print("   results = detector.recognize_faces('path/to/photo.jpg')")
    print("\n3. 保存/加载人脸编码:")
    print("   detector.save_encodings('encodings.pkl')")
    print("   detector.load_encodings('encodings.pkl')")


if __name__ == "__main__":
    main()
