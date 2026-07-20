"""
数据库模块.

使用 SQLite 存储检测记录和任务信息
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class Database:
    """SQLite 数据库管理器.

    支持检测记录和任务的存储、查询

    Attributes:
        db_path: 数据库文件路径
    """

    def __init__(self, db_path: str = "vision_system.db"):
        """初始化数据库.

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    detection_type TEXT NOT NULL,
                    image_path TEXT,
                    result TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    input_data TEXT,
                    output_data TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.commit()

    def insert_detection(
        self, user_id: str, detection_type: str, image_path: str = "", result: dict | None = None
    ) -> int:
        """插入检测记录.

        Args:
            user_id: 用户 ID
            detection_type: 检测类型（object/face/plate/gesture）
            image_path: 图片路径
            result: 检测结果

        Returns:
            记录 ID
        """
        result_json = json.dumps(result, ensure_ascii=False) if result else "{}"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO detections (user_id, detection_type, image_path, result) VALUES (?, ?, ?, ?)",
                (user_id, detection_type, image_path, result_json),
            )
            conn.commit()
            return cursor.lastrowid

    def get_detections(
        self, user_id: str | None = None, detection_type: str | None = None, limit: int = 100
    ) -> list[dict]:
        """获取检测记录.

        Args:
            user_id: 用户 ID，None 表示所有用户
            detection_type: 检测类型，None 表示所有类型
            limit: 返回记录数上限

        Returns:
            检测记录列表
        """
        query = "SELECT * FROM detections WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if detection_type:
            query += " AND detection_type = ?"
            params.append(detection_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            record = dict(row)
            if record.get("result"):
                try:
                    record["result"] = json.loads(record["result"])
                except json.JSONDecodeError:
                    pass
            results.append(record)

        return results

    def insert_task(self, task_id: str, user_id: str, task_type: str, input_data: dict | None = None) -> str:
        """插入任务.

        Args:
            task_id: 任务 ID
            user_id: 用户 ID
            task_type: 任务类型
            input_data: 输入数据

        Returns:
            任务 ID
        """
        input_json = json.dumps(input_data, ensure_ascii=False) if input_data else "{}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, user_id, task_type, input_data) VALUES (?, ?, ?, ?)",
                (task_id, user_id, task_type, input_json),
            )
            conn.commit()
        return task_id

    def update_task(self, task_id: str, status: str, output_data: dict | None = None) -> None:
        """更新任务状态.

        Args:
            task_id: 任务 ID
            status: 新状态
            output_data: 输出数据
        """
        output_json = json.dumps(output_data, ensure_ascii=False) if output_data else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, output_data = ?, updated_at = ? WHERE task_id = ?",
                (status, output_json, time.time(), task_id),
            )
            conn.commit()

    def get_task(self, task_id: str) -> dict | None:
        """获取任务信息.

        Args:
            task_id: 任务 ID

        Returns:
            任务信息，不存在返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()

        if row is None:
            return None

        task = dict(row)
        for key in ["input_data", "output_data"]:
            if task.get(key):
                try:
                    task[key] = json.loads(task[key])
                except json.JSONDecodeError:
                    pass
        return task

    def get_tasks(self, user_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
        """获取任务列表.

        Args:
            user_id: 用户 ID
            status: 任务状态
            limit: 返回记录数上限

        Returns:
            任务列表
        """
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            task = dict(row)
            for key in ["input_data", "output_data"]:
                if task.get(key):
                    try:
                        task[key] = json.loads(task[key])
                    except json.JSONDecodeError:
                        pass
            results.append(task)

        return results

    def delete_detection(self, detection_id: int) -> bool:
        """删除检测记录.

        Args:
            detection_id: 记录 ID

        Returns:
            是否删除成功
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_statistics(self, user_id: str | None = None) -> dict:
        """获取统计信息.

        Args:
            user_id: 用户 ID

        Returns:
            统计信息
        """
        with sqlite3.connect(self.db_path) as conn:
            if user_id:
                det_count = conn.execute("SELECT COUNT(*) FROM detections WHERE user_id = ?", (user_id,)).fetchone()[0]
                task_count = conn.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ?", (user_id,)).fetchone()[0]
            else:
                det_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
                task_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        return {"total_detections": det_count, "total_tasks": task_count}


if __name__ == "__main__":
    # 测试代码
    print("数据库模块测试")
    db = Database("test_temp.db")

    # 插入检测记录
    record_id = db.insert_detection(
        user_id="test_user",
        detection_type="object",
        image_path="test.jpg",
        result={"count": 3, "classes": ["person", "car", "dog"]},
    )
    print(f"插入检测记录: ID={record_id}")

    # 查询检测记录
    detections = db.get_detections(user_id="test_user")
    print(f"查询到 {len(detections)} 条记录")

    # 插入任务
    task_id = db.insert_task(
        task_id="test_task_001", user_id="test_user", task_type="detection", input_data={"image_path": "test.jpg"}
    )
    print(f"插入任务: {task_id}")

    # 更新任务
    db.update_task(task_id, "completed", output_data={"count": 3})

    # 查询任务
    task = db.get_task(task_id)
    print(f"任务状态: {task['status']}")

    # 统计
    stats = db.get_statistics()
    print(f"统计: {stats}")

    # 清理
    Path("test_temp.db").unlink(missing_ok=True)
