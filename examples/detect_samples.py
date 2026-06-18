"""
示例图片检测

检测下载的示例图片
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detector import ObjectDetector


def main():
    """检测示例图片"""

    # 创建检测器
    detector = ObjectDetector(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.45
    )

    # 获取示例图片
    sample_dir = Path('data/samples')
    images = list(sample_dir.glob('*.jpg'))

    if not images:
        print("没有找到示例图片，请先运行下载脚本")
        return

    print(f"找到 {len(images)} 张示例图片\n")

    # 检测每张图片
    for image_path in images:
        print(f"检测: {image_path.name}")
        print("-" * 40)

        results = detector.detect_image(
            str(image_path),
            save_result=True,
            show_result=False,
            output_dir='results'
        )

        print(f"  检测到 {results['count']} 个物体")
        print(f"  类别: {results['class_names']}")
        print(f"  置信度: {[f'{s:.2f}' for s in results['confidences']]}")
        print()

    print("检测完成! 结果保存在 results/ 目录")


if __name__ == '__main__':
    main()
