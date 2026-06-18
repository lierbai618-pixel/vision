"""
实时人脸识别示例

使用摄像头进行实时人脸检测
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import time
from src.face_detector import FaceDetector


def main():
    """实时人脸识别"""

    # 创建检测器
    detector = FaceDetector(min_detection_confidence=0.5)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("=" * 50)
    print("实时人脸识别")
    print("=" * 50)
    print("按 'q' 退出")
    print("按 's' 保存截图")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n摄像头信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")

    # 创建窗口
    window_name = 'Real-time Face Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 计算FPS
        frame_start = time.time()

        # 检测人脸
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用MediaPipe检测
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detector.detect(mp_image)

        # 绘制结果
        face_count = 0
        if result.detections:
            face_count = len(result.detections)
            for detection in result.detections:
                # 获取边界框
                bbox = detection.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height

                # 绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 绘制置信度
                confidence = detection.categories[0].score
                label = f"{confidence:.2%}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 计算FPS
        frame_end = time.time()
        fps = 1.0 / (frame_end - frame_start)

        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示人脸数量
        cv2.putText(frame, f"Faces: {face_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示帧
        cv2.imshow(window_name, frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存截图
            screenshot_path = f"face_screenshot_{frame_count:06d}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"截图已保存: {screenshot_path}")

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


if __name__ == '__main__':
    main()
