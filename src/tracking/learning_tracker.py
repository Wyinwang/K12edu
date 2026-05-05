"""
学习进度跟踪模块
记录用户的学习行为数据，生成学习报告
"""
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.models import get_db
from config.settings import SUBJECTS


class LearningTracker:
    """学习进度跟踪器"""

    def __init__(self):
        self.db = get_db()

    def record_qa_interaction(
        self,
        user_id: int,
        subject: str,
        question: str,
        answer: str
    ) -> bool:
        """
        记录问答交互

        Args:
            user_id: 用户ID
            subject: 学科
            question: 用户问题
            answer: 系统回答

        Returns:
            是否记录成功
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO learning_records (user_id, subject, question, answer, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, subject, question, answer, datetime.now().isoformat()))
                conn.commit()
                return True
        except Exception as e:
            print(f"记录问答交互失败: {e}")
            return False

    def record_exercise_result(
        self,
        user_id: int,
        subject: str,
        question: str,
        question_type: str,
        user_answer: str,
        correct_answer: str,
        is_correct: bool,
        time_spent: Optional[int] = None
    ) -> bool:
        """
        记录练习作答结果

        Args:
            user_id: 用户ID
            subject: 学科
            question: 题目内容
            question_type: 题目类型
            user_answer: 用户答案
            correct_answer: 正确答案
            is_correct: 是否正确
            time_spent: 作答耗时（秒）

        Returns:
            是否记录成功
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO exercise_records
                    (user_id, subject, question, question_type, user_answer, correct_answer, is_correct, time_spent, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, subject, question, question_type, user_answer, correct_answer,
                    int(is_correct), time_spent, datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"记录练习结果失败: {e}")
            return False

    def get_user_qa_count(self, user_id: int, days: int = 30) -> Dict[str, int]:
        """
        获取用户问答次数统计

        Args:
            user_id: 用户ID
            days: 统计天数

        Returns:
            各学科问答次数
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        result = {s: 0 for s in SUBJECTS.keys()}

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT subject, COUNT(*) as count
                    FROM learning_records
                    WHERE user_id = ? AND timestamp >= ?
                    GROUP BY subject
                """, (user_id, start_date))

                for row in cursor.fetchall():
                    result[row['subject']] = row['count']
        except Exception as e:
            print(f"获取问答统计失败: {e}")

        return result

    def get_user_accuracy(self, user_id: int, days: int = 30) -> Dict[str, float]:
        """
        获取用户练习正确率

        Args:
            user_id: 用户ID
            days: 统计天数

        Returns:
            各学科正确率
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        result = {s: 0.0 for s in SUBJECTS.keys()}

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT subject,
                           SUM(is_correct) as correct_count,
                           COUNT(*) as total_count
                    FROM exercise_records
                    WHERE user_id = ? AND timestamp >= ?
                    GROUP BY subject
                """, (user_id, start_date))

                for row in cursor.fetchall():
                    if row['total_count'] > 0:
                        result[row['subject']] = round(row['correct_count'] / row['total_count'] * 100, 1)
        except Exception as e:
            print(f"获取正确率统计失败: {e}")

        return result

    def get_user_learning_days(self, user_id: int, days: int = 30) -> int:
        """
        获取用户累计学习天数

        Args:
            user_id: 用户ID
            days: 统计天数

        Returns:
            学习天数
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                # 从问答记录统计不同日期
                cursor.execute("""
                    SELECT DISTINCT DATE(timestamp) as date
                    FROM learning_records
                    WHERE user_id = ? AND timestamp >= ?
                """, (user_id, start_date))
                qa_dates = set(row['date'] for row in cursor.fetchall())

                # 从练习记录统计不同日期
                cursor.execute("""
                    SELECT DISTINCT DATE(timestamp) as date
                    FROM exercise_records
                    WHERE user_id = ? AND timestamp >= ?
                """, (user_id, start_date))
                exercise_dates = set(row['date'] for row in cursor.fetchall())

                # 合并两个来源的日期
                all_dates = qa_dates | exercise_dates
                return len(all_dates)
        except Exception as e:
            print(f"获取学习天数失败: {e}")
            return 0

    def get_accuracy_trend(self, user_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """
        获取正确率趋势数据

        Args:
            user_id: 用户ID
            days: 统计天数

        Returns:
            每日正确率列表
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        result = []

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DATE(timestamp) as date,
                           SUM(is_correct) as correct_count,
                           COUNT(*) as total_count
                    FROM exercise_records
                    WHERE user_id = ? AND timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (user_id, start_date))

                for row in cursor.fetchall():
                    accuracy = round(row['correct_count'] / row['total_count'] * 100, 1) if row['total_count'] > 0 else 0
                    result.append({
                        'date': row['date'],
                        'accuracy': accuracy,
                        'total': row['total_count'],
                        'correct': row['correct_count']
                    })
        except Exception as e:
            print(f"获取正确率趋势失败: {e}")

        return result

    def get_user_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        获取用户学习概况

        Args:
            user_id: 用户ID
            days: 统计天数

        Returns:
            学习概况数据
        """
        return {
            'learning_days': self.get_user_learning_days(user_id, days),
            'qa_count_by_subject': self.get_user_qa_count(user_id, days),
            'accuracy_by_subject': self.get_user_accuracy(user_id, days),
            'accuracy_trend': self.get_accuracy_trend(user_id, days)
        }

    def get_class_summary(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        获取班级学习概况（教师端）

        Args:
            days: 统计天数

        Returns:
            全班学生学习概况列表
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        result = []

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                # 获取所有学生
                cursor.execute("""
                    SELECT id, username FROM users WHERE role = 'student' AND is_active = 1
                """)
                students = cursor.fetchall()

                for student in students:
                    user_id = student['id']
                    username = student['username']

                    # 统计问答次数
                    cursor.execute("""
                        SELECT COUNT(*) as count FROM learning_records
                        WHERE user_id = ? AND timestamp >= ?
                    """, (user_id, start_date))
                    qa_count = cursor.fetchone()['count']

                    # 统计练习正确率
                    cursor.execute("""
                        SELECT SUM(is_correct) as correct, COUNT(*) as total
                        FROM exercise_records
                        WHERE user_id = ? AND timestamp >= ?
                    """, (user_id, start_date))
                    row = cursor.fetchone()
                    accuracy = round(row['correct'] / row['total'] * 100, 1) if row['total'] > 0 else 0

                    # 学习天数
                    learning_days = self.get_user_learning_days(user_id, days)

                    result.append({
                        'user_id': user_id,
                        'username': username,
                        'qa_count': qa_count,
                        'accuracy': accuracy,
                        'learning_days': learning_days
                    })

                # 按活跃度排序
                result.sort(key=lambda x: x['qa_count'], reverse=True)
        except Exception as e:
            print(f"获取班级概况失败: {e}")

        return result