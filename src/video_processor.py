"""
视频处理模块.

支持视频目标检测、人脸识别
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class VideoProcessor:
    """视频处理器（目标检测、人脸识别）."""

    def __init__(self):
        self.detector = None
        self.face_model = None

    def _load_detectors(self):
        if self.detector is None:
            from src.detector import ObjectDetector

            self.detector = ObjectDetector()
        if self.face_model is None:
            from ultralytics import YOLO

            face_model_path = Path(__file__).parent.parent / "models" / "yolov8n-face.pt"
            if not face_model_path.exists():
                face_model_path = "yolov8n.pt"
            self.face_model = YOLO(str(face_model_path))

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        detection_type: str = "object",
        show_result: bool = False,
        skip_frames: int = 0,
    ) -> dict:
        self._load_detectors()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stats = {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "detection_type": detection_type,
            "processing_time": 0,
        }

        start_time = time.time()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                if out:
                    out.write(frame)
                frame_count += 1
                continue

            annotated = self._detect_and_draw(frame, detection_type)
            if out:
                out.write(annotated)

            if show_result:
                try:
                    cv2.imshow("Video Processing", annotated)
                except:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

        cap.release()
        if out:
            out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

        stats["processing_time"] = time.time() - start_time
        stats["processed_frames"] = frame_count
        return stats

    def _detect_and_draw(self, frame: np.ndarray, detection_type: str) -> np.ndarray:
        annotated = frame.copy()

        if detection_type == "object":
            results = self.detector.model(frame, conf=0.5, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = self.detector.model.names[cls]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, f"{name} {conf:.0%}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

        elif detection_type == "face":
            results = self.face_model(frame, conf=0.25, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        annotated, f"Face {conf:.0%}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )

        return annotated

    def extract_frames(self, video_path: str, output_folder: str, frame_interval: int = 1) -> list[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                fp = output_folder / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(fp), frame)
                frame_paths.append(str(fp))
            frame_count += 1
        cap.release()
        return frame_paths


def process_video_file(video_path: str, output_path: str | None = None, detection_type: str = "object") -> dict:
    processor = VideoProcessor()
    return processor.process_video(video_path, output_path, detection_type)
