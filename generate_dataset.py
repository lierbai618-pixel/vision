"""
生成合成训练数据集
用于快速启动自定义物品检测训练.

生成简单但有效的训练图片，包含：
- 不同形状、颜色、大小的物品
- 随机背景
- 自动生成 YOLO 格式标签
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np

os.chdir(Path(__file__).parent)

# 类别定义
CLASSES = {
    0: {"name": "eraser", "cn": "橡皮", "color": (255, 200, 100), "shape": "rect"},
    1: {"name": "fan", "cn": "风扇", "color": (200, 200, 255), "shape": "circle"},
    2: {"name": "headphones", "cn": "耳机", "color": (100, 100, 100), "shape": "circle"},
    3: {"name": "pen", "cn": "笔", "color": (0, 0, 200), "shape": "line"},
    4: {"name": "pencil", "cn": "铅笔", "color": (255, 255, 0), "shape": "line"},
    5: {"name": "scissors", "cn": "剪刀", "color": (150, 150, 150), "shape": "x"},
    6: {"name": "tape", "cn": "胶带", "color": (200, 200, 200), "shape": "circle"},
    7: {"name": "ruler", "cn": "尺子", "color": (255, 255, 100), "shape": "rect"},
    8: {"name": "calculator", "cn": "计算器", "color": (50, 50, 50), "shape": "rect"},
    9: {"name": "stapler", "cn": "订书机", "color": (0, 100, 200), "shape": "rect"},
}


def draw_eraser(img, x, y, w, h):
    """绘制橡皮."""
    color = (random.randint(200, 255), random.randint(150, 220), random.randint(50, 150))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # 添加文字标记
    cv2.putText(img, "ERASER", (x + 5, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def draw_fan(img, x, y, w, h):
    """绘制风扇."""
    center = (x + w // 2, y + h // 2)
    radius = min(w, h) // 2 - 5
    # 绘制扇叶
    for angle in range(0, 360, 72):
        int(center[0] + radius * np.cos(np.radians(angle)))
        int(center[1] + radius * np.sin(np.radians(angle)))
        cv2.ellipse(img, center, (radius, radius // 3), angle, 0, 360, (200, 200, 255), -1)
    # 中心圆
    cv2.circle(img, center, radius // 4, (100, 100, 100), -1)
    cv2.circle(img, center, radius, (0, 0, 0), 2)


def draw_headphones(img, x, y, w, h):
    """绘制耳机."""
    center = (x + w // 2, y + h // 2)
    # 头带
    cv2.ellipse(img, (center[0], y + h // 4), (w // 2 - 10, h // 3), 0, 180, 360, (50, 50, 50), 8)
    # 左耳罩
    cv2.ellipse(img, (x + 15, center[1]), (15, 25), 0, 0, 360, (80, 80, 80), -1)
    cv2.ellipse(img, (x + 15, center[1]), (15, 25), 0, 0, 360, (0, 0, 0), 2)
    # 右耳罩
    cv2.ellipse(img, (x + w - 15, center[1]), (15, 25), 0, 0, 360, (80, 80, 80), -1)
    cv2.ellipse(img, (x + w - 15, center[1]), (15, 25), 0, 0, 360, (0, 0, 0), 2)


def draw_pen(img, x, y, w, h):
    """绘制笔."""
    color = (random.randint(0, 100), random.randint(0, 100), random.randint(150, 255))
    # 笔身
    cv2.rectangle(img, (x, y + h // 4), (x + w - 10, y + 3 * h // 4), color, -1)
    # 笔尖
    pts = np.array([[x + w - 10, y + h // 4], [x + w, y + h // 2], [x + w - 10, y + 3 * h // 4]])
    cv2.fillPoly(img, [pts], (200, 200, 200))
    cv2.rectangle(img, (x, y + h // 4), (x + w - 10, y + 3 * h // 4), (0, 0, 0), 2)


def draw_pencil(img, x, y, w, h):
    """绘制铅笔."""
    # 笔身（木色）
    cv2.rectangle(img, (x, y + h // 4), (x + w - 15, y + 3 * h // 4), (100, 180, 255), -1)
    # 笔尖
    pts = np.array([[x + w - 15, y + h // 4], [x + w, y + h // 2], [x + w - 15, y + 3 * h // 4]])
    cv2.fillPoly(img, [pts], (50, 50, 50))
    # 橡皮头
    cv2.rectangle(img, (x, y + h // 4), (x + 10, y + 3 * h // 4), (200, 150, 255), -1)
    cv2.rectangle(img, (x, y + h // 4), (x + w - 15, y + 3 * h // 4), (0, 0, 0), 2)


def draw_scissors(img, x, y, w, h):
    """绘制剪刀."""
    (x + w // 2, y + h // 2)
    # 两个刀片
    cv2.line(img, (x, y), (x + w, y + h), (150, 150, 150), 4)
    cv2.line(img, (x + w, y), (x, y + h), (150, 150, 150), 4)
    # 把手
    cv2.circle(img, (x + 10, y + 10), 12, (200, 50, 50), -1)
    cv2.circle(img, (x + w - 10, y + 10), 12, (200, 50, 50), -1)
    cv2.circle(img, (x + 10, y + 10), 12, (0, 0, 0), 2)
    cv2.circle(img, (x + w - 10, y + 10), 12, (0, 0, 0), 2)


def draw_tape(img, x, y, w, h):
    """绘制胶带."""
    center = (x + w // 2, y + h // 2)
    # 外圈
    cv2.circle(img, center, min(w, h) // 2 - 5, (200, 200, 200), -1)
    # 内圈（空心）
    cv2.circle(img, center, min(w, h) // 4, (180, 180, 180), -1)
    cv2.circle(img, center, min(w, h) // 2 - 5, (0, 0, 0), 2)
    cv2.circle(img, center, min(w, h) // 4, (0, 0, 0), 2)


def draw_ruler(img, x, y, w, h):
    """绘制尺子."""
    # 尺子主体
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 100), -1)
    # 刻度线
    for i in range(0, w, w // 10):
        line_h = h // 3 if i % (w // 5) == 0 else h // 6
        cv2.line(img, (x + i, y), (x + i, y + line_h), (0, 0, 0), 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)


def draw_calculator(img, x, y, w, h):
    """绘制计算器."""
    # 主体
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    # 屏幕
    cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + h // 3), (200, 255, 200), -1)
    # 按钮
    for row in range(3):
        for col in range(4):
            bx = x + 10 + col * (w - 20) // 4
            by = y + h // 3 + 10 + row * (h - h // 3 - 20) // 3
            cv2.rectangle(img, (bx, by), (bx + 15, by + 10), (200, 200, 200), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)


def draw_stapler(img, x, y, w, h):
    """绘制订书机."""
    # 底座
    cv2.rectangle(img, (x, y + h // 2), (x + w, y + h), (0, 100, 200), -1)
    # 上盖
    cv2.rectangle(img, (x + 5, y + h // 4), (x + w - 5, y + h // 2 + 5), (0, 80, 180), -1)
    # 铰链
    cv2.circle(img, (x + 15, y + h // 2), 8, (150, 150, 150), -1)
    cv2.rectangle(img, (x, y + h // 2), (x + w, y + h), (0, 0, 0), 2)


# 绘制函数映射
DRAW_FUNCTIONS = {
    0: draw_eraser,
    1: draw_fan,
    2: draw_headphones,
    3: draw_pen,
    4: draw_pencil,
    5: draw_scissors,
    6: draw_tape,
    7: draw_ruler,
    8: draw_calculator,
    9: draw_stapler,
}


def generate_random_background(width, height):
    """生成随机背景."""
    bg_type = random.choice(["solid", "gradient", "noise"])

    if bg_type == "solid":
        color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        img = np.full((height, width, 3), color, dtype=np.uint8)
    elif bg_type == "gradient":
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            ratio = y / height
            color = (
                int(200 * ratio + 55 * (1 - ratio)),
                int(180 * ratio + 75 * (1 - ratio)),
                int(160 * ratio + 95 * (1 - ratio)),
            )
            img[y, :] = color
    else:
        img = np.random.randint(150, 255, (height, width, 3), dtype=np.uint8)

    return img


def generate_image_with_objects(image_size=640, num_objects=None):
    """生成包含物品的图片和标签."""
    if num_objects is None:
        num_objects = random.randint(1, 4)

    # 生成背景
    img = generate_random_background(image_size, image_size)

    labels = []
    used_regions = []

    for _ in range(num_objects):
        # 随机选择类别
        class_id = random.randint(0, len(CLASSES) - 1)

        # 随机大小和位置
        obj_w = random.randint(60, 150)
        obj_h = random.randint(60, 150)
        obj_x = random.randint(10, image_size - obj_w - 10)
        obj_y = random.randint(10, image_size - obj_h - 10)

        # 检查重叠
        overlap = False
        for rx, ry, rw, rh in used_regions:
            if obj_x < rx + rw and obj_x + obj_w > rx and obj_y < ry + rh and obj_y + obj_h > ry:
                overlap = True
                break

        if overlap:
            continue

        # 绘制物品
        draw_func = DRAW_FUNCTIONS.get(class_id)
        if draw_func:
            draw_func(img, obj_x, obj_y, obj_w, obj_h)
            used_regions.append((obj_x, obj_y, obj_w, obj_h))

            # YOLO 格式标签
            center_x = (obj_x + obj_w / 2) / image_size
            center_y = (obj_y + obj_h / 2) / image_size
            norm_w = obj_w / image_size
            norm_h = obj_h / image_size
            labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    return img, labels


def generate_dataset(num_train=200, num_val=50, image_size=640):
    """生成完整数据集."""
    print("=" * 60)
    print("  生成合成训练数据集")
    print("=" * 60)

    base = Path("data/custom_items")

    # 清理旧数据
    for split in ["train", "val"]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        # 清理旧文件
        for f in img_dir.glob("*.jpg"):
            f.unlink()
        for f in lbl_dir.glob("*.txt"):
            f.unlink()

    # 生成训练集
    print(f"\n[1/2] 生成训练集 ({num_train} 张)...")
    for i in range(num_train):
        img, labels = generate_image_with_objects(image_size)
        img_path = base / "images" / "train" / f"train_{i:04d}.jpg"
        lbl_path = base / "labels" / "train" / f"train_{i:04d}.txt"
        cv2.imwrite(str(img_path), img)
        if labels:
            lbl_path.write_text("\n".join(labels), encoding="utf-8")
        if (i + 1) % 50 == 0:
            print(f"  - 已生成 {i + 1}/{num_train} 张")

    # 生成验证集
    print(f"\n[2/2] 生成验证集 ({num_val} 张)...")
    for i in range(num_val):
        img, labels = generate_image_with_objects(image_size)
        img_path = base / "images" / "val" / f"val_{i:04d}.jpg"
        lbl_path = base / "labels" / "val" / f"val_{i:04d}.txt"
        cv2.imwrite(str(img_path), img)
        if labels:
            lbl_path.write_text("\n".join(labels), encoding="utf-8")

    print("\n数据集生成完成!")
    print(f"  - 训练集: {num_train} 张")
    print(f"  - 验证集: {num_val} 张")
    print(f"  - 类别数: {len(CLASSES)}")
    print(f"  - 保存位置: {base}")

    # 显示类别统计
    print("\n类别列表:")
    for idx, info in CLASSES.items():
        print(f"  {idx}: {info['name']} ({info['cn']})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=200, help="训练图片数量")
    parser.add_argument("--val", type=int, default=50, help="验证图片数量")
    parser.add_argument("--size", type=int, default=640, help="图片尺寸")
    args = parser.parse_args()
    generate_dataset(args.train, args.val, args.size)
