"""
摄像头实时检测示例

演示如何使用ObjectDetector进行摄像头实时目标检测
"""

import sys
sys.path.append('..')

from src.detector import ObjectDetector


def main():
    """摄像头实时检测示例"""

    # 创建检测器
    detector = ObjectDetector(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.45
    )

    # 摄像头检测
    print("启动摄像头实时检测...")
    print("按 'q' 退出")
    detector.detect_camera(
        camera_id=0,        # 摄像头ID，0为默认摄像头
        show_result=True     # 显示结果
    )


if __name__ == '__main__':
    main()
