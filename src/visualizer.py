"""
数据可视化模块

提供检测结果的统计图表和可视化
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json


class Visualizer:
    """
    数据可视化器

    提供检测结果的统计图表和可视化

    Attributes:
        output_dir: 输出目录
    """

    def __init__(self, output_dir: str = 'visualizations'):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_detection_stats(
        self,
        results: Dict,
        output_name: str = 'detection_stats.png'
    ) -> str:
        """
        绘制检测统计图表

        Args:
            results: 检测结果
            output_name: 输出文件名

        Returns:
            输出文件路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("警告: matplotlib未安装，无法生成图表")
            print("请运行: pip install matplotlib")
            return ""

        # 统计各类别数量
        class_counts = {}
        if 'class_names' in results:
            for class_name in results['class_names']:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if not class_counts:
            print("没有检测数据可绘制")
            return ""

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 柱状图
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        bars = axes[0].bar(classes, counts, color='steelblue')
        axes[0].set_title('检测类别分布', fontsize=14)
        axes[0].set_xlabel('类别')
        axes[0].set_ylabel('数量')
        axes[0].tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(count), ha='center', va='bottom')

        # 饼图
        axes[1].pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('类别占比', fontsize=14)

        plt.tight_layout()

        # 保存图表
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"统计图表已保存: {output_path}")
        return str(output_path)

    def plot_batch_results(
        self,
        batch_results: Dict,
        output_name: str = 'batch_stats.png'
    ) -> str:
        """
        绘制批量检测结果统计

        Args:
            batch_results: 批量检测结果
            output_name: 输出文件名

        Returns:
            输出文件路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("警告: matplotlib未安装，无法生成图表")
            return ""

        # 统计各类别数量
        type_counts = {}
        detection_type = batch_results.get('detection_type', 'object')

        for result in batch_results.get('results', []):
            if 'error' not in result:
                if detection_type == 'object':
                    for class_name in result.get('class_names', []):
                        type_counts[class_name] = type_counts.get(class_name, 0) + 1
                elif detection_type == 'face':
                    count = result.get('face_count', 0)
                    type_counts['人脸'] = type_counts.get('人脸', 0) + count
                elif detection_type == 'plate':
                    count = result.get('plate_count', 0)
                    type_counts['车牌'] = type_counts.get('车牌', 0) + count
                elif detection_type == 'gesture':
                    count = result.get('hand_count', 0)
                    type_counts['手部'] = type_counts.get('手部', 0) + count

        if not type_counts:
            print("没有检测数据可绘制")
            return ""

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 成功率饼图
        successful = batch_results.get('successful', 0)
        failed = batch_results.get('failed', 0)
        axes[0, 0].pie([successful, failed], labels=['成功', '失败'],
                       autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        axes[0, 0].set_title('处理成功率', fontsize=14)

        # 2. 类别分布柱状图
        classes = list(type_counts.keys())
        counts = list(type_counts.values())

        bars = axes[0, 1].bar(classes, counts, color='steelblue')
        axes[0, 1].set_title('检测类别分布', fontsize=14)
        axes[0, 1].set_xlabel('类别')
        axes[0, 1].set_ylabel('数量')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           str(count), ha='center', va='bottom')

        # 3. 检测数量分布
        detection_counts = []
        for result in batch_results.get('results', []):
            if 'error' not in result:
                if detection_type == 'object':
                    detection_counts.append(result.get('count', 0))
                elif detection_type == 'face':
                    detection_counts.append(result.get('face_count', 0))
                elif detection_type == 'plate':
                    detection_counts.append(result.get('plate_count', 0))
                elif detection_type == 'gesture':
                    detection_counts.append(result.get('hand_count', 0))

        if detection_counts:
            axes[1, 0].hist(detection_counts, bins=20, color='steelblue', edgecolor='black')
            axes[1, 0].set_title('检测数量分布', fontsize=14)
            axes[1, 0].set_xlabel('检测数量')
            axes[1, 0].set_ylabel('图片数量')

        # 4. 置信度分布（如果有）
        confidences = []
        for result in batch_results.get('results', []):
            if 'error' not in result:
                if 'confidences' in result:
                    confidences.extend(result['confidences'])

        if confidences:
            axes[1, 1].hist(confidences, bins=20, color='steelblue', edgecolor='black')
            axes[1, 1].set_title('置信度分布', fontsize=14)
            axes[1, 1].set_xlabel('置信度')
            axes[1, 1].set_ylabel('数量')
        else:
            axes[1, 1].text(0.5, 0.5, '无置信度数据', ha='center', va='center')
            axes[1, 1].set_title('置信度分布', fontsize=14)

        plt.tight_layout()

        # 保存图表
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"批量统计图表已保存: {output_path}")
        return str(output_path)

    def create_detection_summary_image(
        self,
        image_path: str,
        results: Dict,
        output_name: str = 'summary.png'
    ) -> str:
        """
        创建检测结果摘要图片

        Args:
            image_path: 原始图片路径
            results: 检测结果
            output_name: 输出文件名

        Returns:
            输出文件路径
        """
        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 创建摘要图片
        h, w = image.shape[:2]

        # 创建信息面板
        panel_height = 200
        panel = np.ones((panel_height, w, 3), dtype=np.uint8) * 255

        # 添加信息
        y_offset = 30
        cv2.putText(panel, "Detection Summary", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        y_offset += 40
        if 'count' in results:
            cv2.putText(panel, f"Objects: {results['count']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 30

        if 'class_names' in results:
            classes = set(results['class_names'])
            cv2.putText(panel, f"Classes: {', '.join(classes)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 30

        if 'confidences' in results and results['confidences']:
            avg_conf = np.mean(results['confidences'])
            cv2.putText(panel, f"Avg Confidence: {avg_conf:.2%}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 合并图片
        combined = np.vstack([image, panel])

        # 保存
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), combined)

        print(f"摘要图片已保存: {output_path}")
        return str(output_path)

    def export_results_json(self, results: Dict, output_name: str = 'results.json') -> str:
        """
        导出结果为JSON

        Args:
            results: 检测结果
            output_name: 输出文件名

        Returns:
            输出文件路径
        """
        # 转换numpy数组
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        results_serializable = convert_numpy(results)

        output_path = self.output_dir / output_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)

        print(f"结果已导出: {output_path}")
        return str(output_path)


# 便捷函数
def visualize_detection_results(results: Dict, output_dir: str = 'visualizations') -> str:
    """
    快速可视化检测结果

    Args:
        results: 检测结果
        output_dir: 输出目录

        Returns:
            图表路径
    """
    visualizer = Visualizer(output_dir)
    return visualizer.plot_detection_stats(results)


if __name__ == '__main__':
    # 测试代码
    visualizer = Visualizer()

    # 测试数据
    test_results = {
        'count': 5,
        'class_names': ['person', 'car', 'person', 'dog', 'car'],
        'confidences': [0.95, 0.87, 0.92, 0.78, 0.89]
    }

    # 绘制统计图
    chart_path = visualizer.plot_detection_stats(test_results)
    if chart_path:
        print(f"统计图已生成: {chart_path}")

    # 导出JSON
    json_path = visualizer.export_results_json(test_results)
    print(f"JSON已导出: {json_path}")
