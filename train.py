# -*- coding: utf-8 -*-
"""
YOLOv8 一键训练脚本

支持数据集:
  - coco128: COCO 128张图片子集（80类）
  - coco128-seg: COCO 128张图片分割子集

使用方法:
  python train.py                    # 默认使用 coco128 + yolov8s.pt
  python train.py --data coco128     # 指定数据集
  python train.py --model yolov8m.pt # 使用更大模型
  python train.py --epochs 50        # 指定训练轮数
"""

import argparse
import os
import sys
from pathlib import Path

# 确保工作目录正确
os.chdir(Path(__file__).parent)
sys.path.insert(0, '.')

from ultralytics import YOLO


def get_dataset_config(dataset_name: str) -> str:
    """获取数据集配置文件路径"""
    config_dir = Path('configs')
    config_file = config_dir / f'{dataset_name}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f'数据集配置不存在: {config_file}')
    return str(config_file)


def train(args):
    """执行训练"""
    print('=' * 50)
    print('  YOLOv8 目标检测模型训练')
    print('=' * 50)

    # 加载预训练模型
    model_path = args.model
    print(f'\n[1/4] 加载预训练模型: {model_path}')
    model = YOLO(model_path)

    # 获取数据集配置
    data_config = get_dataset_config(args.data)
    print(f'[2/4] 数据集配置: {data_config}')

    # 训练参数
    print(f'[3/4] 开始训练...')
    print(f'  - 训练轮数: {args.epochs}')
    print(f'  - 图片尺寸: {args.imgsz}')
    print(f'  - 批次大小: {args.batch}')
    print(f'  - 学习率: {args.lr}')
    print(f'  - 设备: {args.device if args.device else "自动"}')
    print()

    results = model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=max(args.epochs // 5, 1),
        workers=args.workers,
        verbose=True,
        # 数据增强参数
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    # 输出结果
    best_model = Path(args.project) / args.name / 'weights' / 'best.pt'
    print(f'\n[4/4] 训练完成!')
    print(f'  - 最佳模型: {best_model}')
    print(f'  - 训练目录: {Path(args.project) / args.name}')
    print(f'\n将 best.pt 复制到项目根目录即可使用:')
    print(f'  copy "{best_model}" "models/yolov8_custom.pt"')

    return str(best_model)


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')

    # 数据集
    parser.add_argument('--data', type=str, default='coco128',
                        choices=['coco128', 'coco128-seg'],
                        help='数据集名称 (默认: coco128)')

    # 模型
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='预训练模型 (默认: yolov8s.pt)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图片尺寸 (默认: 640)')
    parser.add_argument('--batch', type=int, default=8,
                        help='批次大小 (默认: 8)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='初始学习率 (默认: 0.01)')
    parser.add_argument('--device', type=str, default='',
                        help='设备: 空=自动, cpu=CPU, 0=GPU')
    parser.add_argument('--patience', type=int, default=30,
                        help='早停耐心值 (默认: 30)')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')

    # 输出
    parser.add_argument('--project', type=str, default='runs/train',
                        help='训练输出目录')
    parser.add_argument('--name', type=str, default='custom_detect',
                        help='实验名称')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
