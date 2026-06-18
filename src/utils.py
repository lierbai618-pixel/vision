"""
工具函数模块

提供通用的图片处理、路径管理、格式校验等工具函数
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def load_image(image_path: str) -> np.ndarray:
    """
    加载图片

    Args:
        image_path: 图片路径（本地路径或URL）

    Returns:
        BGR 格式的图片数组

    Raises:
        ValueError: 图片无法读取时
    """
    if image_path.startswith(('http://', 'https://')):
        # 下载网络图片
        import tempfile
        import urllib.request
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                urllib.request.urlretrieve(image_path, tmp.name)
                image = cv2.imread(tmp.name)
                Path(tmp.name).unlink(missing_ok=True)
        except Exception as e:
            raise ValueError(f"无法下载图片: {e}")
    else:
        image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    return image


def safe_path(path: str) -> str:
    """
    安全化路径（处理中文路径）

    Args:
        path: 原始路径

    Returns:
        安全路径
    """
    p = Path(path)
    if p.exists():
        return str(p)

    # 尝试创建目录
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)

    return str(p)


def get_image_extensions() -> set:
    """
    获取支持的图片扩展名

    Returns:
        扩展名集合
    """
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}


def is_image_file(path: str) -> bool:
    """
    判断文件是否为图片

    Args:
        path: 文件路径

    Returns:
        是否为图片文件
    """
    return Path(path).suffix.lower() in get_image_extensions()


def list_images(folder: str, recursive: bool = False) -> List[str]:
    """
    列出文件夹中的所有图片

    Args:
        folder: 文件夹路径
        recursive: 是否递归查找

    Returns:
        图片路径列表
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    extensions = get_image_extensions()
    images = []

    if recursive:
        for ext in extensions:
            images.extend(str(p) for p in folder_path.rglob(f'*{ext}'))
            images.extend(str(p) for p in folder_path.rglob(f'*{ext.upper()}'))
    else:
        for f in folder_path.iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                images.append(str(f))

    return sorted(set(images))


def resize_image(
    image: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720
) -> np.ndarray:
    """
    等比缩放图片

    Args:
        image: 原始图片
        max_width: 最大宽度
        max_height: 最大高度

    Returns:
        缩放后的图片
    """
    h, w = image.shape[:2]

    if w <= max_width and h <= max_height:
        return image

    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def crop_roi(
    image: np.ndarray,
    x: int, y: int,
    width: int, height: int
) -> np.ndarray:
    """
    裁剪感兴趣区域

    Args:
        image: 原始图片
        x, y: 左上角坐标
        width, height: 区域大小

    Returns:
        裁剪后的图片
    """
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + width)
    y2 = min(h, y + height)
    return image[y1:y2, x1:x2]


def draw_bounding_box(
    image: np.ndarray,
    x: int, y: int,
    width: int, height: int,
    label: str = '',
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    绘制边界框和标签

    Args:
        image: 原始图片
        x, y: 左上角坐标
        width, height: 框大小
        label: 标签文本
        color: 颜色 (B, G, R)
        thickness: 线条粗细

    Returns:
        绘制后的图片
    """
    img = image.copy()
    cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y), color, -1)
        cv2.putText(img, label, (x, y - 2), font, font_scale, (255, 255, 255), 1)

    return img


def create_output_dir(base_dir: str = 'results') -> Path:
    """
    创建输出目录

    Args:
        base_dir: 基础目录

    Returns:
        输出目录路径
    """
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_timestamp() -> str:
    """
    获取时间戳字符串

    Returns:
        格式化的时间戳
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == '__main__':
    # 测试代码
    print("工具函数测试")
    print(f"支持的图片格式: {get_image_extensions()}")
    print(f"是否为图片: {is_image_file('test.jpg')}")
    print(f"时间戳: {get_timestamp()}")
