# 自定义物品数据集使用说明

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
- 0: eraser (橡皮)
- 1: fan (风扇)
- 2: headphones (耳机)
- 3: pen (笔)
- 4: pencil (铅笔)
- 5: scissors (剪刀)
- 6: tape (胶带)
- 7: ruler (尺子)
- 8: calculator (计算器)
- 9: stapler (订书机)

## 训练命令
```bash
python train_custom_items.py --train
```
