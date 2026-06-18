"""
实时综合检测示例

使用摄像头同时进行目标检测、人脸识别、手势识别
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import time
import mediapipe as mp
from src.detector import ObjectDetector
from src.face_detector import FaceDetector
from src.gesture_recognizer import GestureRecognizer


def main():
    """实时综合检测"""

    # 创建检测器
    print("正在初始化检测器...")
    detector = ObjectDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    face_detector = FaceDetector(min_detection_confidence=0.5)
    gesture_recognizer = GestureRecognizer()
    print("检测器初始化完成!")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("=" * 50)
    print("实时综合检测")
    print("=" * 50)
    print("按 'q' 退出")
    print("按 's' 保存截图")
    print("按 '1' 切换目标检测")
    print("按 '2' 切换人脸识别")
    print("按 '3' 切换手势识别")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n摄像头信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")

    # 创建窗口
    window_name = 'Real-time Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 检测开关
    enable_object_detection = True
    enable_face_detection = True
    enable_gesture_detection = True

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 计算FPS
        frame_start = time.time()

        # 目标检测
        if enable_object_detection:
            results = detector.model(frame, conf=0.5, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = f"{detector.model.names[cls]}: {conf:.2%}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 人脸识别
        if enable_face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            face_result = face_detector.detector.detect(mp_image)

            if face_result.detections:
                for detection in face_result.detections:
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    w = bbox.width
                    h = bbox.height
                    conf = detection.categories[0].score

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Face: {conf:.2%}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 手势识别
        if enable_gesture_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            gesture_result = gesture_recognizer.detector.detect(mp_image)

            if gesture_result.hand_landmarks:
                for hand_landmarks in gesture_result.hand_landmarks:
                    # 绘制关键点
                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    # 识别手势
                    landmarks = []
                    for landmark in hand_landmarks:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    gesture = gesture_recognizer._classify_gesture(landmarks)
                    cv2.putText(frame, gesture, (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 计算FPS
        frame_end = time.time()
        fps = 1.0 / (frame_end - frame_start)

        # 显示状态
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示检测开关状态
        status_y = 150
        cv2.putText(frame, f"Object: {'ON' if enable_object_detection else 'OFF'}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if enable_object_detection else (0, 0, 255), 2)
        cv2.putText(frame, f"Face: {'ON' if enable_face_detection else 'OFF'}", (10, status_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if enable_face_detection else (0, 0, 255), 2)
        cv2.putText(frame, f"Gesture: {'ON' if enable_gesture_detection else 'OFF'}", (10, status_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if enable_gesture_detection else (0, 0, 255), 2)

        # 显示帧
        cv2.imshow(window_name, frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存截图
            screenshot_path = f"screenshot_{frame_count:06d}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"截图已保存: {screenshot_path}")
        elif key == ord('1'):
            enable_object_detection = not enable_object_detection
            print(f"目标检测: {'开启' if enable_object_detection else '关闭'}")
        elif key == ord('2'):
            enable_face_detection = not enable_face_detection
            print(f"人脸识别: {'开启' if enable_face_detection else '关闭'}")
        elif key == ord('3'):
            enable_gesture_detection = not enable_gesture_detection
            print(f"手势识别: {'开启' if enable_gesture_detection else '关闭'}")

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
