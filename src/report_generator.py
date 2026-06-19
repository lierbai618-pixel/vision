"""
报告生成模块.

支持生成PDF、Excel、HTML格式的检测报告
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


class ReportGenerator:
    """报告生成器.

    支持生成PDF、Excel、HTML格式的检测报告

    Attributes:
        output_dir: 输出目录
    """

    def __init__(self, output_dir: str = "reports"):
        """初始化报告生成器.

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, results: dict, title: str = "检测报告", output_name: str | None = None) -> str:
        """生成HTML报告.

        Args:
            results: 检测结果
            title: 报告标题
            output_name: 输出文件名

        Returns:
            报告路径
        """
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"report_{timestamp}.html"

        # 统计信息
        total = results.get("total_images", 0)
        successful = results.get("successful", 0)
        failed = results.get("failed", 0)
        detection_type = results.get("detection_type", "object")

        # 计算各类别数量
        type_counts = {}
        for result in results.get("results", []):
            if "error" not in result:
                if detection_type == "object":
                    for class_name in result.get("class_names", []):
                        type_counts[class_name] = type_counts.get(class_name, 0) + 1
                elif detection_type == "face":
                    count = result.get("face_count", 0)
                    type_counts["人脸"] = type_counts.get("人脸", 0) + count
                elif detection_type == "plate":
                    count = result.get("plate_count", 0)
                    type_counts["车牌"] = type_counts.get("车牌", 0) + count
                elif detection_type == "gesture":
                    count = result.get("hand_count", 0)
                    type_counts["手部"] = type_counts.get("手部", 0) + count

        # 生成HTML
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
        }}
        .summary-item {{
            text-align: center;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            min-width: 150px;
        }}
        .summary-item h3 {{
            margin: 0;
            color: #666;
            font-size: 14px;
        }}
        .summary-item p {{
            margin: 10px 0 0 0;
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p style="text-align: center; color: #666;">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <div class="summary-item">
                <h3>总图片数</h3>
                <p>{total}</p>
            </div>
            <div class="summary-item">
                <h3>成功处理</h3>
                <p>{successful}</p>
            </div>
            <div class="summary-item">
                <h3>处理失败</h3>
                <p>{failed}</p>
            </div>
            <div class="summary-item">
                <h3>成功率</h3>
                <p>{successful / total * 100:.1f}%</p>
            </div>
        </div>

        <h2>检测类型: {detection_type}</h2>

        <h2>检测结果统计</h2>
        <table>
            <thead>
                <tr>
                    <th>类别</th>
                    <th>数量</th>
                    <th>占比</th>
                </tr>
            </thead>
            <tbody>
"""

        total_count = sum(type_counts.values())
        for class_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_count * 100 if total_count > 0 else 0
            html += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <h2>详细结果</h2>
        <table>
            <thead>
                <tr>
                    <th>图片名称</th>
                    <th>检测数量</th>
                    <th>详细信息</th>
                </tr>
            </thead>
            <tbody>
"""

        for result in results.get("results", []):
            if "error" in result:
                html += f"""
                <tr>
                    <td>{result["image_name"]}</td>
                    <td>0</td>
                    <td>错误: {result["error"]}</td>
                </tr>
"""
            else:
                if detection_type == "object":
                    count = result.get("count", 0)
                    classes = ", ".join(result.get("class_names", []))
                    html += f"""
                <tr>
                    <td>{result["image_name"]}</td>
                    <td>{count}</td>
                    <td>{classes}</td>
                </tr>
"""
                elif detection_type == "face":
                    count = result.get("face_count", 0)
                    html += f"""
                <tr>
                    <td>{result["image_name"]}</td>
                    <td>{count}</td>
                    <td>检测到 {count} 张人脸</td>
                </tr>
"""
                elif detection_type == "plate":
                    count = result.get("plate_count", 0)
                    html += f"""
                <tr>
                    <td>{result["image_name"]}</td>
                    <td>{count}</td>
                    <td>检测到 {count} 个车牌</td>
                </tr>
"""
                elif detection_type == "gesture":
                    count = result.get("hand_count", 0)
                    html += f"""
                <tr>
                    <td>{result["image_name"]}</td>
                    <td>{count}</td>
                    <td>检测到 {count} 只手</td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <div class="footer">
            <p>由智能视觉系统生成</p>
        </div>
    </div>
</body>
</html>
"""

        # 保存HTML
        output_path = self.output_dir / output_name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"HTML报告已生成: {output_path}")
        return str(output_path)

    def generate_csv_report(self, results: dict, output_name: str | None = None) -> str:
        """生成CSV报告.

        Args:
            results: 检测结果
            output_name: 输出文件名

        Returns:
            报告路径
        """
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"report_{timestamp}.csv"

        detection_type = results.get("detection_type", "object")

        # 保存CSV
        output_path = self.output_dir / output_name
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow(["图片名称", "检测类型", "检测数量", "详细信息"])

            # 写入数据
            for result in results.get("results", []):
                if "error" in result:
                    writer.writerow([result["image_name"], detection_type, 0, f"错误: {result['error']}"])
                else:
                    if detection_type == "object":
                        count = result.get("count", 0)
                        details = ", ".join(result.get("class_names", []))
                    elif detection_type == "face":
                        count = result.get("face_count", 0)
                        details = f"{count}张人脸"
                    elif detection_type == "plate":
                        count = result.get("plate_count", 0)
                        details = f"{count}个车牌"
                    elif detection_type == "gesture":
                        count = result.get("hand_count", 0)
                        details = f"{count}只手"
                    else:
                        count = 0
                        details = ""

                    writer.writerow([result["image_name"], detection_type, count, details])

        print(f"CSV报告已生成: {output_path}")
        return str(output_path)

    def generate_json_report(self, results: dict, output_name: str | None = None) -> str:
        """生成JSON报告.

        Args:
            results: 检测结果
            output_name: 输出文件名

        Returns:
            报告路径
        """
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"report_{timestamp}.json"

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

        # 添加元数据
        report = {
            "metadata": {"generated_at": datetime.now().isoformat(), "version": "1.0.0"},
            "results": results_serializable,
        }

        # 保存JSON
        output_path = self.output_dir / output_name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"JSON报告已生成: {output_path}")
        return str(output_path)

    def generate_all_reports(self, results: dict, title: str = "检测报告") -> dict[str, str]:
        """生成所有格式的报告.

        Args:
            results: 检测结果
            title: 报告标题

        Returns:
            各格式报告路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_paths = {
            "html": self.generate_html_report(results, title, f"report_{timestamp}.html"),
            "csv": self.generate_csv_report(results, f"report_{timestamp}.csv"),
            "json": self.generate_json_report(results, f"report_{timestamp}.json"),
        }

        print("\n所有报告已生成:")
        for format_type, path in report_paths.items():
            print(f"  {format_type}: {path}")

        return report_paths


# 便捷函数
def generate_detection_report(results: dict, output_dir: str = "reports") -> dict[str, str]:
    """快速生成检测报告.

    Args:
        results: 检测结果
        output_dir: 输出目录

    Returns:
        各格式报告路径
    """
    generator = ReportGenerator(output_dir)
    return generator.generate_all_reports(results)


if __name__ == "__main__":
    # 测试代码
    generator = ReportGenerator()

    # 测试数据
    test_results = {
        "total_images": 10,
        "successful": 9,
        "failed": 1,
        "detection_type": "object",
        "results": [
            {"image_name": "test1.jpg", "count": 3, "class_names": ["person", "car", "dog"]},
            {"image_name": "test2.jpg", "count": 2, "class_names": ["person", "cat"]},
            {"image_name": "test3.jpg", "error": "无法读取图片"},
        ],
    }

    # 生成报告
    report_paths = generator.generate_all_reports(test_results)
    print(f"报告已生成: {report_paths}")
