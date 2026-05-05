"""
数据库模型定义
使用SQLite存储用户数据、学习记录、错题等
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class DatabaseManager:
    """数据库管理类"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # 默认数据库路径
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base_dir, "data", "edu_system.db")

        self.db_path = db_path
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        """初始化数据库表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'student',
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """)

            # 学习记录表（问答交互）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

            # 练习记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exercise_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    question TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    user_answer TEXT NOT NULL,
                    correct_answer TEXT NOT NULL,
                    is_correct INTEGER NOT NULL,
                    time_spent INTEGER,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

            # 错题表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wrong_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    question TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    correct_answer TEXT NOT NULL,
                    user_answer TEXT NOT NULL,
                    knowledge_point TEXT,
                    explanation TEXT,
                    created_at TEXT NOT NULL,
                    review_count INTEGER DEFAULT 0,
                    is_mastered INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

            # 知识点统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_point_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    knowledge_point TEXT NOT NULL,
                    wrong_count INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, subject, knowledge_point)
                )
            """)

            conn.commit()


# 全局数据库实例
_db_instance: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """获取数据库实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance