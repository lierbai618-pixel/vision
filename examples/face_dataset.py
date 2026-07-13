"""
人脸识别数据集示例.

使用Kaggle数据集进行人脸识别
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.face_detector import FaceDetector


def main():
    """人脸识别数据集示例."""
    # 创建检测器
    detector = FaceDetector()

    # 数据集路径
    dataset_path = Path("data/face/Celebrity Faces Dataset")

    if not dataset_path.exists():
        print("数据集不存在，请先下载")
        return

    # 获取人脸类别（名人文件夹）
    face_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    print("=" * 50)
    print("人脸识别数据集示例")
    print("=" * 50)
    print(f"数据集路径: {dataset_path}")
    print(f"名人类别数量: {len(face_dirs)}")

    # 显示名人类别
    print("\n名人类别:")
    for face_dir in face_dirs[:10]:  # 只显示前10个
        face_name = face_dir.name
        image_count = len(list(face_dir.glob("*.jpg")))
        print(f"  - {face_name}: {image_count} 张图片")

    if len(face_dirs) > 10:
        print(f"  ... 还有 {len(face_dirs) - 10} 个类别")

    # 测试每个类别的第一张图片
    print("\n" + "=" * 50)
    print("测试检测")
    print("=" * 50)

    for face_dir in face_dirs[:3]:  # 只测试前3个类别
        face_name = face_dir.name
        images = list(face_dir.glob("*.jpg"))

        if images:
            image_path = images[0]
            print(f"\n名人: {face_name}")
            print(f"图片: {image_path.name}")
            print("-" * 40)

            try:
                results = detector.detect_faces(str(image_path))
                print(f"  检测到 {results['face_count']} 张人脸")

                if results["face_count"] > 0:
                    for j, (location, confidence) in enumerate(
                        zip(results["face_locations"], results["face_confidences"])
                    ):
                        print(
                            f"  人脸 {j + 1}: x={location['x']}, y={location['y']}, "
                            f"宽={location['width']}, 高={location['height']}, "
                            f"置信度={confidence:.2%}"
                        )
            except Exception as e:
                print(f"  错误: {e}")

    print("\n" + "=" * 50)
    print("测试完成!")


if __name__ == "__main__":
    main()
