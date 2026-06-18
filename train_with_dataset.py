"""
下载数据集并训练模型.

支持的数据集:
  - coco2017: COCO 2017 完整数据集 (~118K训练图片, 80类)
  - coco2017-val: COCO 2017 验证集 (~5K图片, 快速验证)
  - openimages: OpenImages V7 (~1.7M图片, 600类)
  - visdrone: VisDrone 无人机视角 (~6K图片, 10类)
  - custom: 自定义数据集 (放在 data/custom/ 目录)

用法:
  python train_with_dataset.py                       # 下载coco2017-val并训练
  python train_with_dataset.py --dataset coco2017    # 下载完整COCO并训练
  python train_with_dataset.py --dataset visdrone    # 下载VisDrone并训练
  python train_with_dataset.py --dataset custom      # 使用自定义数据集
  python train_with_dataset.py --epochs 100 --model yolov8m.pt
"""

import argparse
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, ".")

from ultralytics import YOLO


def download_dataset(dataset_name):
    """下载数据集."""
    import urllib.request
    import zipfile

    data_dir = Path("data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "coco2017":
        print("正在下载 COCO 2017 完整数据集...")
        print("注意: COCO完整数据集约 19GB, 请确保磁盘空间充足")
        # 使用 ultralytics 内置下载
        return "coco.yaml"  # ultralytics 内置配置

    elif dataset_name == "coco2017-val":
        print("正在下载 COCO 2017 验证集...")
        return "coco.yaml"

    elif dataset_name == "openimages":
        print("正在下载 OpenImages V7 数据集...")
        return "open-images-v7.yaml"

    elif dataset_name == "visdrone":
        print("正在下载 VisDrone 数据集...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone.zip"
        target_dir = data_dir / "VisDrone"
        if target_dir.exists():
            print("  VisDrone 已存在, 跳过下载")
        else:
            print("  正在下载...")
            zip_path = data_dir / "VisDrone.zip"
            urllib.request.urlretrieve(url, str(zip_path))
            with zipfile.ZipFile(str(zip_path), "r") as z:
                z.extractall(str(data_dir))
            zip_path.unlink()
            print("  下载完成!")
        # 创建配置文件
        cfg_path = Path("configs/visdrone.yaml")
        cfg_content = f"""# VisDrone 数据集配置
path: {target_dir}
train: train/images
val: val/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
        cfg_path.write_text(cfg_content, encoding="utf-8")
        return str(cfg_path)

    elif dataset_name == "custom":
        custom_dir = Path("data/custom")
        if not custom_dir.exists():
            print(f"自定义数据集目录不存在: {custom_dir}")
            print("请按以下结构放置数据:")
            print("  data/custom/images/train/  # 训练图片")
            print("  data/custom/labels/train/  # 标签文件 (YOLO格式)")
            print("  data/custom/images/val/    # 验证图片")
            print("  data/custom/labels/val/    # 验证标签")
            sys.exit(1)
        cfg_path = Path("configs/custom.yaml")
        cfg_content = f"""# 自定义数据集配置
path: {custom_dir.absolute()}
train: images/train
val: images/val

# 请修改为你的类别
names:
  0: object
"""
        cfg_path.write_text(cfg_content, encoding="utf-8")
        return str(cfg_path)

    return dataset_name


def train(args):
    """执行训练."""
    print("=" * 60)
    print("  YOLOv8 数据集训练 - 提高物品检测准确率")
    print("=" * 60)

    # 1. 加载预训练模型
    print(f"\n[1/5] 加载预训练模型: {args.model}")
    model = YOLO(args.model)

    # 2. 下载/准备数据集
    print(f"\n[2/5] 准备数据集: {args.dataset}")
    data_config = download_dataset(args.dataset)

    # 3. 显示训练配置
    print("\n[3/5] 训练配置:")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 基础模型: {args.model}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 图片尺寸: {args.imgsz}")
    print(f"  - 批次大小: {args.batch}")
    print(f"  - 设备: {args.device if args.device else '自动(GPU优先)'}")

    # 4. 开始训练
    print("\n[4/5] 开始训练...")
    model.train(
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
        fraction=args.fraction,
        # 数据增强
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # 5. 完成
    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n[5/5] 训练完成!")
    print(f"  - 最佳模型: {best_model}")
    print(f"  - 训练目录: {Path(args.project) / args.name}")
    print("\n将训练好的模型复制到项目中使用:")
    print(f'  copy "{best_model}" "models/"')
    print("\n然后在 app.py 中选择该模型即可使用")

    return str(best_model)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 数据集训练")

    parser.add_argument(
        "--dataset",
        type=str,
        default="coco2017-val",
        choices=["coco2017", "coco2017-val", "openimages", "visdrone", "custom"],
        help="数据集 (默认: coco2017-val)",
    )

    parser.add_argument("--model", type=str, default="yolov8m.pt", help="预训练模型 (默认: yolov8m.pt)")

    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸 (默认: 640)")
    parser.add_argument("--batch", type=int, default=8, help="批次大小 (默认: 8, GPU显存不足请调小)")
    parser.add_argument("--lr", type=float, default=0.01, help="初始学习率 (默认: 0.01)")
    parser.add_argument("--device", type=str, default="", help="设备: 空=自动, cpu=CPU, 0=GPU")
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值 (默认: 30)")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数 (默认: 4)")

    parser.add_argument("--fraction", type=float, default=1.0, help="数据集使用比例 (默认: 1.0, 快速训练用0.1)")

    parser.add_argument("--project", type=str, default="runs/train", help="训练输出目录")
    parser.add_argument("--name", type=str, default="improved", help="实验名称")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
