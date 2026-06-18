"""
目标追踪模块.

使用 YOLOv8 内置追踪器进行多目标追踪
"""

from __future__ import annotations

from pathlib import Path

import cv2


class ObjectTracker:
    """目标追踪器.

    使用 YOLOv8 内置追踪器（如 BoT-SORT、ByteTrack）进行多目标追踪

    Attributes:
        model: YOLOv8 模型
        tracker_type: 追踪器类型
    """

    def __init__(
        self, model_path: str = "yolov8n.pt", tracker_type: str = "bytetrack.yaml", conf_threshold: float = 0.5
    ):
        """初始化目标追踪器.

        Args:
            model_path: 模型路径
            tracker_type: 追踪器配置文件，支持 'bytetrack.yaml' 或 'botsort.yaml'
            conf_threshold: 置信度阈值
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold

    def track_video(
        self, video_path: str, output_path: str | None = None, show_result: bool = False, save_result: bool = True
    ) -> dict:
        """追踪视频中的目标.

        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            show_result: 是否显示结果
            save_result: 是否保存结果

        Returns:
            追踪统计信息
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if save_result and output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stats = {"total_frames": total_frames, "fps": fps, "tracked_ids": set(), "frame_tracks": [], "total_tracks": 0}

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 使用 YOLOv8 追踪
            results = self.model.track(
                frame, conf=self.conf_threshold, tracker=self.tracker_type, persist=True, verbose=False
            )

            # 收集追踪信息
            frame_ids = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                for tid in track_ids:
                    stats["tracked_ids"].add(int(tid))
                    frame_ids.append(int(tid))

            stats["frame_tracks"].append(len(frame_ids))

            # 绘制结果
            annotated = results[0].plot()

            if out:
                out.write(annotated)

            if show_result:
                try:
                    cv2.imshow("Tracking", annotated)
                except Exception:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧")

        cap.release()
        if out:
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        stats["total_tracks"] = len(stats["tracked_ids"])
        stats["tracked_ids"] = list(stats["tracked_ids"])
        stats["processed_frames"] = frame_count

        print(f"追踪完成: {stats['total_tracks']} 个目标, {frame_count} 帧")
        return stats

    def track_camera(self, camera_id: int = 0, output_path: str | None = None, show_result: bool = True) -> None:
        """摄像头实时追踪.

        Args:
            camera_id: 摄像头 ID
            output_path: 输出视频路径
            show_result: 是否显示结果
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")

        print("摄像头追踪已启动，按 'q' 退出")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame, conf=self.conf_threshold, tracker=self.tracker_type, persist=True, verbose=False
            )

            annotated = results[0].plot()

            if show_result:
                try:
                    cv2.imshow("Camera Tracking", annotated)
                except Exception:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("摄像头追踪已结束")


if __name__ == "__main__":
    # 测试代码
    print("目标追踪器测试")
    tracker = ObjectTracker()
    print(f"追踪器类型: {tracker.tracker_type}")
    print("追踪器初始化成功")
