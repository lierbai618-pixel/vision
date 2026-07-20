"""
合并 COCO + 自定义物品数据集为统一 90 类模型
COCO 80类 (0-79) + 自定义10类 (80-89).
"""

import os
import shutil
from pathlib import Path

os.chdir(Path(__file__).parent)

# 自定义类别 (ID 80-89)
CUSTOM_CLASSES = {
    80: "eraser",  # 橡皮
    81: "fan",  # 风扇
    82: "headphones",  # 耳机
    83: "pen",  # 笔
    84: "pencil",  # 铅笔
    85: "tape",  # 胶带
    86: "ruler",  # 尺子
    87: "calculator",  # 计算器
    88: "stapler",  # 订书机
    89: "scissors2",  # 剪刀(自定义，与COCO的scissors区分)
}

# 合并后的完整类别表
ALL_CLASSES = {}
# COCO 80类
coco_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
for i, name in enumerate(coco_names):
    ALL_CLASSES[i] = name
for i, name in CUSTOM_CLASSES.items():
    ALL_CLASSES[i] = name


def create_merged_config():
    """创建合并数据集配置."""
    coco_path = Path("D:/项目/datasets/coco").absolute()
    custom_path = Path("data/custom_items").absolute()

    config = f"""# 合并数据集: COCO 80类 + 自定义物品10类 = 90类
# 用于统一物品检测模型训练

path: {coco_path}
train: images/train2017
val: images/val2017

names:
"""
    for idx in sorted(ALL_CLASSES.keys()):
        config += f"  {idx}: {ALL_CLASSES[idx]}\n"

    Path("configs/merged_90class.yaml").write_text(config, encoding="utf-8")
    print("配置文件: configs/merged_90class.yaml")
    print(f"COCO 数据路径: {coco_path}")
    print(f"自定义数据路径: {custom_path}")
    return config


def relabel_custom_items():
    """将自定义物品标签从 0-9 重映射到 80-89."""
    label_dir = Path("data/custom_items/labels")
    mapping = {0: 80, 1: 81, 2: 82, 3: 83, 4: 84, 5: 89, 6: 85, 7: 86, 8: 87, 9: 88}

    count = 0
    for split in ["train", "val"]:
        for lbl_file in (label_dir / split).glob("*.txt"):
            lines = lbl_file.read_text(encoding="utf-8").strip().split("\n")
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_id = int(parts[0])
                    new_id = mapping.get(old_id, old_id)
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
            if new_lines:
                lbl_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                count += 1
    print(f"重映射了 {count} 个标签文件")


def copy_custom_to_coco():
    """将自定义图片和标签复制到 COCO 数据目录中."""
    coco_train_img = Path("D:/项目/datasets/coco/images/train2017")
    coco_train_lbl = Path("D:/项目/datasets/coco/labels/train2017")
    coco_val_img = Path("D:/项目/datasets/coco/images/val2017")
    coco_val_lbl = Path("D:/项目/datasets/coco/labels/val2017")

    custom_base = Path("data/custom_items")

    for split, dst_img, dst_lbl in [("train", coco_train_img, coco_train_lbl), ("val", coco_val_img, coco_val_lbl)]:
        src_img = custom_base / "images" / split
        src_lbl = custom_base / "labels" / split

        img_count = 0
        for img_file in src_img.glob("*.jpg"):
            dst = dst_img / f"custom_{img_file.name}"
            if not dst.exists():
                shutil.copy2(img_file, dst)
                img_count += 1

            lbl_file = src_lbl / (img_file.stem + ".txt")
            if lbl_file.exists():
                dst_l = dst_lbl / f"custom_{lbl_file.name}"
                if not dst_l.exists():
                    shutil.copy2(lbl_file, dst_l)

        print(f"  {split}: 复制了 {img_count} 张自定义图片")


if __name__ == "__main__":
    print("=" * 50)
    print("  合并数据集: COCO 80类 + 自定义10类")
    print("=" * 50)

    print("\n[1/3] 重映射自定义标签 ID...")
    relabel_custom_items()

    print("\n[2/3] 复制自定义图片到 COCO 目录...")
    copy_custom_to_coco()

    print("\n[3/3] 创建合并配置文件...")
    create_merged_config()

    print("\n完成! 可以开始训练了。")
