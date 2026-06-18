# -*- coding: utf-8 -*-
"""
自定义物品检测数据集构建 & 训练脚本

针对小物品（橡皮、风扇、耳机等）优化
自动下载示例图片并创建 YOLO 格式标签

用法:
  python train_custom_items.py --create-samples   # 创建示例数据集
  python train_custom_items.py --train             # 训练模型
  python train_custom_items.py --all               # 创建数据集 + 训练
"""

import os
import sys
import argparse
import shutil
import json
from pathlib import Path
import random

os.chdir(Path(__file__).parent)
sys.path.insert(0, '.')

# ==================== 自定义类别 ====================
CUSTOM_CLASSES = {
    0: "eraser",           # 橡皮
    1: "fan",              # 风扇
    2: "headphones",       # 耳机
    3: "pen",              # 笔
    4: "pencil",           # 铅笔
    5: "scissors",         # 剪刀
    6: "tape",             # 胶带
    7: "ruler",            # 尺子
    8: "calculator",       # 计算器
    9: "stapler",          # 订书机
}

# 类别中文名
CLASS_NAMES_CN = {
    "eraser": "橡皮",
    "fan": "风扇",
    "headphones": "耳机",
    "pen": "笔",
    "pencil": "铅笔",
    "scissors": "剪刀",
    "tape": "胶带",
    "ruler": "尺子",
    "calculator": "计算器",
    "stapler": "订书机",
}


def create_dataset_structure():
    """创建数据集目录结构"""
    base = Path("data/custom_items")
    for split in ["train", "val"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    return base


def create_sample_labels(base_path):
    """
    创建示例标签文件
    注意：这只是框架，你需要用自己的图片替换
    """
    print("\n" + "="*60)
    print("  自定义物品数据集 - 创建示例")
    print("="*60)

    # 创建数据集配置
    config = f"""# 自定义物品检测数据集
# 针对小物品（橡皮、风扇、耳机等）优化

path: {base_path.absolute()}
train: images/train
val: images/val

names:
"""
    for idx, name in CUSTOM_CLASSES.items():
        config += f"  {idx}: {name}\n"

    config_path = Path("configs/custom_items.yaml")
    config_path.write_text(config, encoding='utf-8')
    print(f"\n[1/3] 配置文件已创建: {config_path}")

    # 创建示例标签（YOLO格式：class_id center_x center_y width height）
    print(f"\n[2/3] 创建示例标签文件...")

    # 为每个类别创建示例标签
    for class_id, class_name in CUSTOM_CLASSES.items():
        # 训练集示例标签
        for i in range(3):
            label_path = base_path / "labels" / "train" / f"{class_name}_{i}.txt"
            # 创建一个简单的示例标签（需要替换为真实标注）
            label_path.write_text(
                f"{class_id} 0.5 0.5 0.3 0.3\n",
                encoding='utf-8'
            )

        # 验证集示例标签
        label_path = base_path / "labels" / "val" / f"{class_name}_val.txt"
        label_path.write_text(
            f"{class_id} 0.5 0.5 0.3 0.3\n",
            encoding='utf-8'
        )

    print(f"  - 创建了 {len(CUSTOM_CLASSES)} 个类别的示例标签")
    print(f"  - 训练集: {base_path / 'labels' / 'train'}")
    print(f"  - 验证集: {base_path / 'labels' / 'val'}")

    # 创建使用说明
    readme = """# 自定义物品数据集使用说明

## 数据集结构
```
data/custom_items/
├── images/
│   ├── train/    # 训练图片放这里
│   └── val/      # 验证图片放这里
└── labels/
    ├── train/    # 训练标签（自动生成或手动标注）
    └── val/      # 验证标签
```

## 如何添加自己的图片

1. **拍摄/收集图片**：
   - 每个类别至少 20 张图片（推荐 50+ 张）
   - 不同角度、光照、背景
   - 图片分辨率建议 640x640 以上

2. **放置图片**：
   - 训练图片: `data/custom_items/images/train/`
   - 验证图片: `data/custom_items/images/val/`（每个类 5-10 张）

3. **标注图片**（推荐使用工具）：
   - [LabelImg](https://github.com/heartexlabs/labelImg) - 最常用
   - [CVAT](https://cvat.ai/) - 在线标注
   - [Roboflow](https://roboflow.com/) - 自动标注 + 数据增强

4. **标注格式**（YOLO格式）：
   ```
   class_id center_x center_y width height
   ```
   - class_id: 类别编号（0-9）
   - center_x, center_y: 中心点坐标（归一化 0-1）
   - width, height: 宽高（归一化 0-1）

## 类别列表
"""
    for idx, name in CUSTOM_CLASSES.items():
        cn = CLASS_NAMES_CN.get(name, name)
        readme += f"- {idx}: {name} ({cn})\n"

    readme += """
## 训练命令
```bash
python train_custom_items.py --train
```
"""
    readme_path = base_path / "README.md"
    readme_path.write_text(readme, encoding='utf-8')
    print(f"\n[3/3] 使用说明已创建: {readme_path}")

    print("\n" + "="*60)
    print("  下一步操作：")
    print("="*60)
    print("""
1. 按照说明收集并标注图片
2. 将图片放入 data/custom_items/images/ 目录
3. 运行训练: python train_custom_items.py --train

提示：
- 每个类别至少需要 20 张标注好的图片
- 可以使用 LabelImg 工具进行标注
- 图片越多，检测效果越好
""")


def train_model(args):
    """训练自定义物品检测模型"""
    from ultralytics import YOLO

    print("\n" + "="*60)
    print("  自定义物品检测模型训练")
    print("="*60)

    # 检查数据集
    base = Path("data/custom_items")
    train_images = list((base / "images" / "train").glob("*.*"))
    val_images = list((base / "images" / "val").glob("*.*"))

    print(f"\n数据集统计:")
    print(f"  - 训练图片: {len(train_images)} 张")
    print(f"  - 验证图片: {len(val_images)} 张")

    if len(train_images) < 10:
        print("\n⚠️  警告：训练图片太少，建议每个类别至少 20 张")
        print("   请先收集并标注图片，然后重新运行训练")
        return

    # 加载预训练模型
    model_path = args.model
    print(f"\n[1/3] 加载预训练模型: {model_path}")
    model = YOLO(model_path)

    # 开始训练
    print(f"\n[2/3] 开始训练...")
    print(f"  - 轮数: {args.epochs}")
    print(f"  - 批次: {args.batch}")
    print(f"  - 图片尺寸: {args.imgsz}")

    results = model.train(
        data="configs/custom_items.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/train",
        name="custom_items",
        patience=args.patience,
        save_period=max(args.epochs // 5, 1),
        workers=4,
        verbose=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # 完成
    best_model = Path("runs/train/custom_items/weights/best.pt")
    print(f"\n[3/3] 训练完成!")
    print(f"  - 最佳模型: {best_model}")
    print(f"\n复制模型到项目中:")
    print(f'  copy "{best_model}" "models/yolov8_custom_items.pt"')

    return str(best_model)


def main():
    parser = argparse.ArgumentParser(description='自定义物品检测训练')
    parser.add_argument('--create-samples', action='store_true',
                        help='创建示例数据集结构')
    parser.add_argument('--train', action='store_true',
                        help='训练模型')
    parser.add_argument('--all', action='store_true',
                        help='创建数据集 + 训练')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='预训练模型 (默认: yolov8m.pt)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图片尺寸 (默认: 640)')
    parser.add_argument('--batch', type=int, default=8,
                        help='批次大小 (默认: 8)')
    parser.add_argument('--device', type=str, default='',
                        help='设备: 空=自动, cpu=CPU, 0=GPU')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值 (默认: 20)')

    args = parser.parse_args()

    if args.create_samples or args.all:
        base = create_dataset_structure()
        create_sample_labels(base)

    if args.train or args.all:
        train_model(args)

    if not args.create_samples and not args.train and not args.all:
        parser.print_help()


if __name__ == '__main__':
    main()
