"""
自定义数据集训练示例

演示如何使用YOLOv8训练自定义数据集
"""

from ultralytics import YOLO


def main():
    """自定义数据集训练示例"""

    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 训练配置
    train_args = {
        'data': 'coco8.yaml',      # 数据集配置文件
        'epochs': 100,              # 训练轮数
        'imgsz': 640,               # 图片尺寸
        'batch': 16,                # 批次大小
        'device': '',               # 设备，''自动选择
        'workers': 8,               # 数据加载线程数
        'project': 'runs/train',    # 项目目录
        'name': 'custom_model',     # 实验名称
        'exist_ok': False,          # 是否覆盖已有实验
        'pretrained': True,         # 使用预训练权重
        'optimizer': 'auto',        # 优化器
        'verbose': True,            # 详细输出
        'seed': 0,                  # 随机种子
        'deterministic': True,      # 确定性模式
    }

    # 开始训练
    print("开始训练...")
    results = model.train(**train_args)

    # 打印训练结果
    print("\n训练完成!")
    print(f"最佳模型保存在: {results.save_dir}")

    # 验证模型
    print("\n开始验证...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # 导出模型
    print("\n导出模型...")
    model.export(format='onnx')  # 导出为ONNX格式


def train_with_custom_dataset():
    """
    使用自定义数据集训练

    数据集结构:
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
    """

    # 创建数据集配置文件
    data_yaml = """
    # 数据集配置
    path: ./dataset  # 数据集根目录
    train: train/images  # 训练集图片路径
    val: val/images      # 验证集图片路径

    # 类别
    names:
      0: person
      1: car
      2: dog
    """

    # 保存配置文件
    with open('custom_dataset.yaml', 'w') as f:
        f.write(data_yaml)

    # 加载模型
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data='custom_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16
    )

    return results


if __name__ == '__main__':
    main()
