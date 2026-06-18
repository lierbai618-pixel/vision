"""
视频检测示例

演示如何使用ObjectDetector进行视频目标检测
"""

import sys
sys.path.append('..')

from src.detector import ObjectDetector


def main():
    """视频检测示例"""

    # 创建检测器
    detector = ObjectDetector(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.45
    )

    # 检测视频
    print("开始检测视频...")
    stats = detector.detect_video(
        video_path='input.mp4',      # 输入视频路径
        output_path='output.mp4',    # 输出视频路径
        show_result=False            # 不显示结果
    )

    # 打印统计信息
    print("\n检测统计:")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  FPS: {stats['fps']}")
    print(f"  总检测数: {stats['total_detections']}")
    print(f"  平均每帧检测数: {stats['total_detections']/stats['total_frames']:.2f}")


if __name__ == '__main__':
    main()
