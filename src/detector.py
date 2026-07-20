"""
YOLOv8目标检测器核心模块.

提供图片、视频、摄像头等多种检测功能
"""

from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO


class ObjectDetector:
    """YOLOv8目标检测器.

    支持图片、视频、摄像头等多种输入方式的目标检测

    Attributes:
        model: YOLO模型实例
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        device: 运算设备
    """

    def __init__(
        self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5, iou_threshold: float = 0.45, device: str = ""
    ):
        """初始化检测器.

        Args:
            model_path: 模型路径，支持'yolov8n.pt', 'yolov8s.pt'等
            conf_threshold: 置信度阈值，范围0-1
            iou_threshold: IOU阈值，用于NMS
            device: 运算设备，''自动选择，'cpu'使用CPU，'0'使用GPU
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def detect_image(
        self, image_path: str, save_result: bool = True, show_result: bool = False, output_dir: str = "results"
    ) -> dict:
        """检测单张图片.

        Args:
            image_path: 图片路径
            save_result: 是否保存结果
            show_result: 是否显示结果
            output_dir: 输出目录

        Returns:
            检测结果字典，包含boxes, classes, scores等
        """
        # 运行检测
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)

        # 解析结果
        result = results[0]
        detections = self._parse_results(result)

        # 保存结果
        if save_result:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(output_dir) / Path(image_path).name
            result.save(str(output_path))
            print(f"结果已保存到: {output_path}")

        # 显示结果
        if show_result:
            result.show()

        return detections

    def detect_video(self, video_path: str, output_path: str | None = None, show_result: bool = False) -> dict:
        """检测视频.

        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            show_result: 是否显示结果

        Returns:
            检测统计信息
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入器
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 统计信息
        stats = {"total_frames": total_frames, "fps": fps, "detections_per_frame": [], "total_detections": 0}

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)

            # 获取检测数量
            detections = len(results[0].boxes)
            stats["detections_per_frame"].append(detections)
            stats["total_detections"] += detections

            # 绘制结果
            annotated_frame = results[0].plot()

            # 保存帧
            if output_path:
                out.write(annotated_frame)

            # 显示帧
            if show_result:
                try:
                    cv2.imshow("Detection", annotated_frame)
                except:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧")

        # 释放资源
        cap.release()
        if output_path:
            out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

        print(f"视频检测完成，共处理 {frame_count} 帧")
        return stats

    def detect_camera(self, camera_id: int = 0, output_path: str | None = None, show_result: bool = True) -> None:
        """摄像头实时检测.

        Args:
            camera_id: 摄像头ID
            output_path: 输出视频路径（可选）
            show_result: 是否显示结果
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")

        print("摄像头已打开，按 'q' 退出")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)

            # 绘制结果
            annotated_frame = results[0].plot()

            # 显示帧
            if show_result:
                try:
                    cv2.imshow("Camera Detection", annotated_frame)
                except:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # 释放资源
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("摄像头检测已结束")

    def detect_batch(self, image_folder: str, output_dir: str = "batch_results") -> list[dict]:
        """批量检测图片.

        Args:
            image_folder: 图片文件夹路径
            output_dir: 输出目录

        Returns:
            所有图片的检测结果列表
        """
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise ValueError(f"文件夹不存在: {image_folder}")

        # 支持的图片格式
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        # 获取所有图片
        images = [f for f in image_folder.iterdir() if f.suffix.lower() in image_extensions]

        if not images:
            raise ValueError(f"文件夹中没有图片: {image_folder}")

        print(f"找到 {len(images)} 张图片")

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 批量检测
        results = []
        for i, image_path in enumerate(images):
            print(f"检测 [{i + 1}/{len(images)}]: {image_path.name}")
            detection = self.detect_image(str(image_path), save_result=True, output_dir=output_dir)
            results.append(detection)

        print(f"批量检测完成，结果保存在: {output_dir}")
        return results

    def _parse_results(self, result) -> dict:
        """解析检测结果.

        Args:
            result: YOLO检测结果

        Returns:
            解析后的结果字典
        """
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[cls_id] for cls_id in class_ids]

        return {
            "boxes": boxes.tolist(),
            "confidences": confidences.tolist(),
            "class_ids": class_ids.tolist(),
            "class_names": class_names,
            "count": len(boxes),
        }


# 便捷函数
def detect_single_image(image_path: str, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5) -> dict:
    """快速检测单张图片.

    Args:
        image_path: 图片路径
        model_path: 模型路径
        conf_threshold: 置信度阈值

    Returns:
        检测结果
    """
    detector = ObjectDetector(model_path, conf_threshold)
    return detector.detect_image(image_path)


if __name__ == "__main__":
    # 测试代码
    detector = ObjectDetector()

    # 检测示例图片
    results = detector.detect_image("https://ultralytics.com/images/bus.jpg", save_result=False)
    print(f"检测到 {results['count']} 个物体")
    print(f"类别: {results['class_names']}")
