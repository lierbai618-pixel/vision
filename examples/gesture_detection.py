"""
手势识别示例.

使用Kaggle数据集进行手势识别
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.gesture_recognizer import GestureRecognizer


def main():
    """手势识别示例."""
    # 创建识别器
    recognizer = GestureRecognizer()

    # 数据集路径
    dataset_path = Path("data/gesture/leapGestRecog")

    if not dataset_path.exists():
        print("数据集不存在，请先下载")
        return

    # 获取手势类别（数字文件夹）
    gesture_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()])

    print("=" * 50)
    print("手势识别示例")
    print("=" * 50)
    print(f"数据集路径: {dataset_path}")
    print(f"手势类别数量: {len(gesture_dirs)}")

    # 手势类别名称映射
    gesture_names = {
        "01": "手掌",
        "02": "L形",
        "03": "拳头",
        "04": "移动拳头",
        "05": "拇指",
        "06": "食指",
        "07": "OK手势",
        "08": "移动手掌",
        "09": "C形",
        "10": "向下指",
    }

    # 显示手势类别
    print("\n手势类别:")
    for gesture_dir in gesture_dirs:
        # 获取子目录
        sub_dirs = [d for d in gesture_dir.iterdir() if d.is_dir()]
        total_images = sum(len(list(d.glob("*.png"))) for d in sub_dirs)
        gesture_name = gesture_names.get(gesture_dir.name, gesture_dir.name)
        print(f"  - {gesture_dir.name} ({gesture_name}): {total_images} 张图片")

    # 测试每个类别的第一张图片
    print("\n" + "=" * 50)
    print("测试识别")
    print("=" * 50)

    for gesture_dir in gesture_dirs[:3]:  # 只测试前3个类别
        # 获取第一个子目录
        sub_dirs = [d for d in gesture_dir.iterdir() if d.is_dir()]
        if sub_dirs:
            gesture_name = gesture_names.get(gesture_dir.name, gesture_dir.name)
            images = list(sub_dirs[0].glob("*.png"))

            if images:
                image_path = images[0]
                print(f"\n手势: {gesture_name}")
                print(f"图片: {image_path.name}")
                print("-" * 40)

                try:
                    results = recognizer.detect_hands(str(image_path))
                    print(f"  检测到 {results['hand_count']} 只手")

                    if results["hand_count"] > 0:
                        for j, hand in enumerate(results["hands"]):
                            print(f"  手 {j + 1}: {hand['landmark_count']} 个关键点")
                except Exception as e:
                    print(f"  错误: {e}")

    print("\n" + "=" * 50)
    print("测试完成!")


if __name__ == "__main__":
    main()
