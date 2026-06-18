"""
模型训练模块

支持YOLOv8自定义数据集训练
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, List
from ultralytics import YOLO


class ModelTrainer:
    """
    模型训练器

    支持YOLOv8自定义数据集训练

    Attributes:
        model_path: 预训练模型路径
        data_path: 数据集配置文件路径
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        初始化训练器

        Args:
            model_path: 预训练模型路径
        """
        self.model_path = model_path
        self.model = None
        self.training_results = None

    def create_dataset_config(
        self,
        dataset_path: str,
        class_names: List[str],
        output_path: str = 'dataset.yaml'
    ) -> str:
        """
        创建数据集配置文件

        Args:
            dataset_path: 数据集根目录
            class_names: 类别名称列表
            output_path: 输出配置文件路径

        Returns:
            配置文件路径
        """
        dataset_path = Path(dataset_path)

        # 检查目录结构
        train_images = dataset_path / 'train' / 'images'
        val_images = dataset_path / 'val' / 'images'

        if not train_images.exists():
            raise ValueError(f"训练集图片目录不存在: {train_images}")
        if not val_images.exists():
            raise ValueError(f"验证集图片目录不存在: {val_images}")

        # 创建配置
        config = {
            'path': str(dataset_path.absolute()),
            'train': str(train_images.relative_to(dataset_path)),
            'val': str(val_images.relative_to(dataset_path)),
            'names': {i: name for i, name in enumerate(class_names)}
        }

        # 保存配置
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"数据集配置已保存: {output_path}")
        return str(output_path)

    def train(
        self,
        data_path: str,
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        device: str = '',
        project: str = 'runs/train',
        name: str = 'exp',
        save_period: int = -1,
        workers: int = 8,
        patience: int = 100
    ) -> Dict:
        """
        训练模型

        Args:
            data_path: 数据集配置文件路径
            epochs: 训练轮数
            img_size: 图片尺寸
            batch_size: 批次大小
            learning_rate: 学习率
            device: 设备，''自动选择，'cpu'使用CPU，'0'使用GPU
            project: 项目目录
            name: 实验名称
            save_period: 保存周期，-1表示只保存最佳和最新
            workers: 数据加载线程数
            patience: 早停耐心值

        Returns:
            训练结果
        """
        # 加载模型
        self.model = YOLO(self.model_path)

        # 开始训练
        print(f"开始训练...")
        print(f"  模型: {self.model_path}")
        print(f"  数据集: {data_path}")
        print(f"  轮数: {epochs}")
        print(f"  图片尺寸: {img_size}")
        print(f"  批次大小: {batch_size}")

        results = self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            lr0=learning_rate,
            device=device,
            project=project,
            name=name,
            save_period=save_period,
            workers=workers,
            patience=patience,
            verbose=True
        )

        # 保存训练结果
        self.training_results = {
            'model_path': str(Path(project) / name / 'weights' / 'best.pt'),
            'results_dir': str(Path(project) / name),
            'epochs': epochs,
            'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else epochs
        }

        print(f"\n训练完成!")
        print(f"  最佳模型: {self.training_results['model_path']}")
        print(f"  结果目录: {self.training_results['results_dir']}")

        return self.training_results

    def validate(self, data_path: str = None) -> Dict:
        """
        验证模型

        Args:
            data_path: 数据集配置文件路径，None使用训练时的数据集

        Returns:
            验证结果
        """
        if self.model is None:
            raise ValueError("模型未加载，请先训练或加载模型")

        print("开始验证...")
        metrics = self.model.val(data=data_path)

        results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }

        print(f"\n验证结果:")
        print(f"  mAP50: {results['mAP50']:.4f}")
        print(f"  mAP50-95: {results['mAP50-95']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")

        return results

    def export_model(
        self,
        format: str = 'onnx',
        output_path: str = None
    ) -> str:
        """
        导出模型

        Args:
            format: 导出格式，支持'onnx', 'torchscript', 'tflite'等
            output_path: 输出路径

        Returns:
            导出模型路径
        """
        if self.model is None:
            raise ValueError("模型未加载，请先训练或加载模型")

        print(f"导出模型为 {format} 格式...")
        export_path = self.model.export(format=format)

        if output_path:
            # 复制到指定路径
            import shutil
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(export_path, output_path)
            export_path = output_path

        print(f"模型已导出: {export_path}")
        return str(export_path)

    def load_model(self, model_path: str) -> None:
        """
        加载模型

        Args:
            model_path: 模型路径
        """
        self.model = YOLO(model_path)
        print(f"模型已加载: {model_path}")

    def predict(self, image_path: str, conf: float = 0.5) -> Dict:
        """
        使用模型进行预测

        Args:
            image_path: 图片路径
            conf: 置信度阈值

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未加载，请先训练或加载模型")

        results = self.model(image_path, conf=conf)

        # 解析结果
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[cls_id] for cls_id in class_ids]

        return {
            'boxes': boxes.tolist(),
            'confidences': confidences.tolist(),
            'class_ids': class_ids.tolist(),
            'class_names': class_names,
            'count': len(boxes)
        }


# 便捷函数
def train_model(
    data_path: str,
    model_path: str = 'yolov8n.pt',
    epochs: int = 100
) -> Dict:
    """
    快速训练模型

    Args:
        data_path: 数据集配置文件路径
        model_path: 预训练模型路径
        epochs: 训练轮数

    Returns:
        训练结果
    """
    trainer = ModelTrainer(model_path)
    return trainer.train(data_path, epochs)


if __name__ == '__main__':
    # 测试代码
    trainer = ModelTrainer()

    # 创建数据集配置
    class_names = ['person', 'car', 'dog']
    config_path = trainer.create_dataset_config(
        'data/custom_dataset',
        class_names,
        'configs/custom_dataset.yaml'
    )
    print(f"配置文件: {config_path}")
