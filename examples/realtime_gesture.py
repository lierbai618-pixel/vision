"""
实时手势识别示例

使用摄像头进行实时手势识别
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import time
import mediapipe as mp
from src.gesture_recognizer import GestureRecognizer


def main():
    """实时手势识别"""

    # 创建识别器
    recognizer = GestureRecognizer()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("=" * 50)
    print("实时手势识别")
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
    window_name = 'Real-time Gesture Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 计算FPS
        frame_start = time.time()

        # 检测手势
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = recognizer.detector.detect(mp_image)

        # 绘制结果
        hand_count = 0
        gesture_text = ""

        if result.hand_landmarks:
            hand_count = len(result.hand_landmarks)

            for hand_landmarks in result.hand_landmarks:
                # 绘制关键点
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # 识别手势
                landmarks = []
                for landmark in hand_landmarks:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                gesture = recognizer._classify_gesture(landmarks)
                gesture_text = gesture

                # 显示手势名称
                cv2.putText(frame, gesture, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 计算FPS
        frame_end = time.time()
        fps = 1.0 / (frame_end - frame_start)

        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示手部数量
        cv2.putText(frame, f"Hands: {hand_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示帧
        cv2.imshow(window_name, frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存截图
            screenshot_path = f"gesture_screenshot_{frame_count:06d}.jpg"
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
