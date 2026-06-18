"""
图片检测示例

演示如何使用ObjectDetector进行图片目标检测
"""

import sys
sys.path.append('..')

from src.detector import ObjectDetector


def main():
    """图片检测示例"""

    # 创建检测器
    detector = ObjectDetector(
        model_path='yolov8n.pt',  # 使用yolov8n模型，速度快
        conf_threshold=0.5,        # 置信度阈值
        iou_threshold=0.45         # IOU阈值
    )

    # 检测图片
    print("开始检测图片...")
    results = detector.detect_image(
        image_path='https://ultralytics.com/images/bus.jpg',
        save_result=True,          # 保存结果
        show_result=False,         # 不显示结果（服务器环境）
        output_dir='results'       # 输出目录
    )

    # 打印结果
    print("\n检测结果:")
    print(f"  检测到 {results['count']} 个物体")
    print(f"  类别: {results['class_names']}")
    print(f"  置信度: {[f'{s:.2f}' for s in results['confidences']]}")


if __name__ == '__main__':
    main()
