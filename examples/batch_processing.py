"""
批量处理示例.

演示如何批量处理图片并生成报告
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.batch_processor import BatchProcessor
from src.report_generator import ReportGenerator
from src.visualizer import Visualizer


def main():
    """批量处理示例."""
    print("=" * 50)
    print("批量处理示例")
    print("=" * 50)

    # 创建处理器
    processor = BatchProcessor()
    visualizer = Visualizer()
    report_generator = ReportGenerator()

    # 批量目标检测
    print("\n1. 批量目标检测")
    print("-" * 40)

    results = processor.batch_detect_images("data/samples", "batch_results/object", detection_type="object")

    print(f"  总图片数: {results['total_images']}")
    print(f"  成功处理: {results['successful']}")
    print(f"  处理失败: {results['failed']}")

    # 生成统计图
    print("\n2. 生成统计图")
    print("-" * 40)

    chart_path = visualizer.plot_batch_results(results, "batch_stats.png")
    print(f"  统计图: {chart_path}")

    # 生成报告
    print("\n3. 生成报告")
    print("-" * 40)

    report_paths = report_generator.generate_all_reports(results, "批量检测报告")
    for format_type, path in report_paths.items():
        print(f"  {format_type}: {path}")

    # 导出结果
    print("\n4. 导出结果")
    print("-" * 40)

    processor.export_results_to_json(results, "batch_results/results.json")
    processor.export_results_to_csv(results, "batch_results/results.csv")
    processor.generate_report(results, "batch_results/report.md")

    print("\n批量处理完成!")


if __name__ == "__main__":
    main()
