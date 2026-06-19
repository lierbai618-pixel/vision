"""
手势识别模块.

使用 MediaPipe HandLandmarker 进行手势识别
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class GestureRecognizer:
    """手势识别器.

    使用 MediaPipe HandLandmarker 检测手部关键点并识别手势

    Attributes:
        num_hands: 最大检测手数
        min_detection_confidence: 最小检测置信度
    """

    # 手势类别
    GESTURES = {
        "fist": "拳头",
        "open_palm": "张开手掌",
        "pointing": "食指指向前方",
        "peace": "耶/剪刀手",
        "thumbs_up": "竖起拇指",
    }

    def __init__(self, num_hands: int = 2, min_detection_confidence: float = 0.3, min_tracking_confidence: float = 0.3):
        """初始化手势识别器.

        Args:
            num_hands: 最大检测手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小追踪置信度
        """
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._detector = None

    def _load_detector(self):
        """懒加载 MediaPipe 检测器."""
        if self._detector is not None:
            return

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # 查找模型文件
            model_path = self._find_model()
            if model_path is None:
                raise RuntimeError("未找到 hand_landmarker.task 模型文件")

            base = python.BaseOptions(model_asset_path=model_path)
            opts = vision.HandLandmarkerOptions(
                base_options=base,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._detector = vision.HandLandmarker.create_from_options(opts)

        except ImportError:
            raise RuntimeError("MediaPipe 未安装，请运行: pip install mediapipe")

    def _find_model(self) -> str | None:
        """查找 hand_landmarker.task 模型文件."""
        candidate_paths = [
            "models/hand_landmarker.task",
            str(Path(__file__).parent.parent / "models" / "hand_landmarker.task"),
            "C:/temp/models/hand_landmarker.task",
        ]
        for path in candidate_paths:
            if Path(path).exists():
                return path
        return None

    def _classify_gesture(self, landmarks) -> str:
        """根据手部关键点判断手势.

        Args:
            landmarks: MediaPipe 手部关键点列表

        Returns:
            手势名称
        """
        # 指尖和指间关节的 y 坐标比较
        tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指的指尖
        pips = [3, 6, 10, 14, 18]  # 对应的指间关节
        fingers_up = []

        for tip, pip in zip(tips, pips):
            if tip == 4:  # 拇指特殊处理（水平方向）
                fingers_up.append(landmarks[tip].x < landmarks[pip].x)
            else:
                fingers_up.append(landmarks[tip].y < landmarks[pip].y)

        count = sum(fingers_up)

        if count == 0:
            return self.GESTURES["fist"]
        if count == 5:
            return self.GESTURES["open_palm"]
        if count == 1 and fingers_up[1]:
            return self.GESTURES["pointing"]
        if count == 2 and fingers_up[1] and fingers_up[2]:
            return self.GESTURES["peace"]
        if count == 1 and fingers_up[0]:
            return self.GESTURES["thumbs_up"]

        return f"{count}根手指"

    def detect_hands(self, image_path: str) -> dict:
        """检测图片中的手部.

        Args:
            image_path: 图片路径

        Returns:
            检测结果字典，包含 hand_count, hands
        """
        self._load_detector()

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 转换为 RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe 检测
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        hands = []
        if result.hand_landmarks:
            _h, _w = image.shape[:2]
            for hand_lms in result.hand_landmarks:
                gesture = self._classify_gesture(hand_lms)
                landmarks = []
                for lm in hand_lms:
                    landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
                hands.append({"gesture": gesture, "landmarks": landmarks})

        return {"hand_count": len(hands), "hands": hands}

    def detect_hands_in_frame(self, frame: np.ndarray) -> tuple[int, list[dict]]:
        """检测视频帧中的手部（用于实时检测）.

        Args:
            frame: BGR 格式的视频帧

        Returns:
            (手部数量, 手部信息列表)
        """
        self._load_detector()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        hands = []
        if result.hand_landmarks:
            _h, _w = frame.shape[:2]
            for hand_lms in result.hand_landmarks:
                gesture = self._classify_gesture(hand_lms)
                hands.append({"gesture": gesture, "landmarks": hand_lms})

        return len(hands), hands


# 便捷函数
def detect_gestures_in_image(image_path: str) -> dict:
    """快速检测单张图片的手势.

    Args:
        image_path: 图片路径

    Returns:
        检测结果
    """
    recognizer = GestureRecognizer()
    return recognizer.detect_hands(image_path)


if __name__ == "__main__":
    # 测试代码
    print("手势识别器测试")

    try:
        recognizer = GestureRecognizer()
        test_image = "https://ultralytics.com/images/bus.jpg"
        results = recognizer.detect_hands(test_image)
        print(f"检测到 {results['hand_count']} 只手")
        for i, hand in enumerate(results["hands"]):
            print(f"  手 {i + 1}: {hand['gesture']}")
    except Exception as e:
        print(f"测试失败: {e}")
