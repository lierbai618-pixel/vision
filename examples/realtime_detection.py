"""
实时目标检测示例

使用摄像头进行实时目标检测
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import time
from src.detector import ObjectDetector


def main():
    """实时目标检测"""

    # 创建检测器
    detector = ObjectDetector(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.45
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("=" * 50)
    print("实时目标检测")
    print("=" * 50)
    print("按 'q' 退出")
    print("按 's' 保存截图")
    print("按 'f' 切换全屏")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n摄像头信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")

    # 创建窗口
    window_name = 'Real-time Object Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    fps_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 计算FPS
        frame_start = time.time()

        # 检测
        results = detector.model(frame, conf=0.5, verbose=False)

        # 绘制结果
        annotated_frame = results[0].plot()

        # 计算FPS
        frame_end = time.time()
        fps = 1.0 / (frame_end - frame_start)
        fps_list.append(fps)

        # 显示FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示检测数量
        if results[0].boxes is not None:
            det_count = len(results[0].boxes)
            cv2.putText(annotated_frame, f"Objects: {det_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示帧
        cv2.imshow(window_name, annotated_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存截图
            screenshot_path = f"screenshot_{frame_count:06d}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"截图已保存: {screenshot_path}")
        elif key == ord('f'):
            # 切换全屏
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if prop == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        frame_count += 1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 统计信息
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print(f"\n检测统计:")
    print(f"  总帧数: {frame_count}")
    print(f"  总时间: {total_time:.2f}秒")
    print(f"  平均FPS: {avg_fps:.2f}")
    print(f"  最小FPS: {min(fps_list):.2f}")
    print(f"  最大FPS: {max(fps_list):.2f}")


if __name__ == '__main__':
    main()
