"""
实时监测示例

使用实时监测模块进行视频流监测
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime_monitor import RealtimeMonitor, MonitorConfig


def main():
    """实时监测示例"""

    print("=" * 50)
    print("实时监测示例")
    print("=" * 50)

    # 配置监测器
    config = MonitorConfig(
        camera_id=0,
        fps=30,
        width=640,
        height=480,
        detection_types=['object', 'face', 'gesture'],
        save_video=False
    )

    # 创建监测器
    monitor = RealtimeMonitor(config)

    # 定义回调函数
    def on_frame(frame, data):
        """帧回调函数"""
        # 可以在这里处理每一帧
        # 例如：保存检测结果、发送通知等
        pass

    # 启动监测
    try:
        monitor.start(callback=on_frame)
    except KeyboardInterrupt:
        print("\n用户中断")
        monitor.stop()
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == '__main__':
    main()
