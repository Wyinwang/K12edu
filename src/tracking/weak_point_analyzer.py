"""
错题本与薄弱知识点诊断模块
"""
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.models import get_db
from src.llm.qwen_llm import QwenLLM
from config.settings import SUBJECTS


class AnswerEvaluator:
    """答案评估器，用于提取知识点"""

    def __init__(self):
        self.llm = QwenLLM()

    def extract_knowledge_point(self, question: str, subject: str, question_type: str) -> str:
        """
        从题目中提取知识点

        Args:
            question: 题目内容
            subject: 学科
            question_type: 题目类型

        Returns:
            知识点标签
        """
        subject_name = SUBJECTS.get(subject, {}).get('name', subject)

        prompt = f"""分析以下{subject_name}题目涉及的知识点，用一个简短的标签概括（不超过10个字）。

题目类型：{question_type}
题目内容：{question}

只输出知识点标签，不要有其他内容。例如：
- 语文题目输出：古诗词、拼音、作文、阅读理解等
- 数学题目输出：加减法、分数、几何图形、应用题等
- 英语题目输出：单词拼写、语法时态、句型转换等"""

        try:
            result = self.llm.chat(prompt)
            # 清理输出
            knowledge_point = result.strip().replace('\n', '')
            if len(knowledge_point) > 15:
                knowledge_point = knowledge_point[:15]
            return knowledge_point
        except Exception as e:
            print(f"提取知识点失败: {e}")
            return "未分类"


class WrongQuestionManager:
    """错题管理器"""

    def __init__(self):
        self.db = get_db()
        self.evaluator = AnswerEvaluator()

    def add_wrong_question(
        self,
        user_id: int,
        subject: str,
        question: str,
        question_type: str,
        correct_answer: str,
        user_answer: str,
        explanation: str = ""
    ) -> bool:
        """
        添加错题到错题本

        Args:
            user_id: 用户ID
            subject: 学科
            question: 题目内容
            question_type: 题目类型
            correct_answer: 正确答案
            user_answer: 用户答案
            explanation: 解析

        Returns:
            是否添加成功
        """
        try:
            # 提取知识点
            knowledge_point = self.evaluator.extract_knowledge_point(question, subject, question_type)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # 检查是否已存在相同错题
                cursor.execute("""
                    SELECT id FROM wrong_questions
                    WHERE user_id = ? AND question = ? AND is_mastered = 0
                """, (user_id, question))
                existing = cursor.fetchone()

                if existing:
                    # 更新复习次数
                    cursor.execute("""
                        UPDATE wrong_questions SET review_count = review_count + 1
                        WHERE id = ?
                    """, (existing['id'],))
                else:
                    # 新增错题
                    cursor.execute("""
                        INSERT INTO wrong_questions
                        (user_id, subject, question, question_type, correct_answer, user_answer,
                         knowledge_point, explanation, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, subject, question, question_type, correct_answer, user_answer,
                        knowledge_point, explanation, datetime.now().isoformat()
                    ))

                # 更新知识点统计
                cursor.execute("""
                    INSERT INTO knowledge_point_stats (user_id, subject, knowledge_point, wrong_count, last_updated)
                    VALUES (?, ?, ?, 1, ?)
                    ON CONFLICT(user_id, subject, knowledge_point)
                    DO UPDATE SET wrong_count = wrong_count + 1, last_updated = ?
                """, (user_id, subject, knowledge_point, datetime.now().isoformat(), datetime.now().isoformat()))

                conn.commit()
                return True
        except Exception as e:
            print(f"添加错题失败: {e}")
            return False

    def get_wrong_questions(
        self,
        user_id: int,
        subject: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取用户错题列表

        Args:
            user_id: 用户ID
            subject: 学科（可选，不指定则返回全部学科）
            limit: 返回数量限制

        Returns:
            错题列表
        """
        result = []

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if subject:
                    cursor.execute("""
                        SELECT * FROM wrong_questions
                        WHERE user_id = ? AND subject = ? AND is_mastered = 0
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (user_id, subject, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM wrong_questions
                        WHERE user_id = ? AND is_mastered = 0
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (user_id, limit))

                for row in cursor.fetchall():
                    result.append(dict(row))
        except Exception as e:
            print(f"获取错题列表失败: {e}")

        return result

    def mark_mastered(self, wrong_question_id: int) -> bool:
        """
        标记错题已掌握

        Args:
            wrong_question_id: 错题ID

        Returns:
            是否标记成功
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE wrong_questions SET is_mastered = 1 WHERE id = ?
                """, (wrong_question_id,))

                # 同时更新知识点正确计数
                cursor.execute("""
                    SELECT user_id, subject, knowledge_point FROM wrong_questions WHERE id = ?
                """, (wrong_question_id,))
                row = cursor.fetchone()
                if row:
                    cursor.execute("""
                        INSERT INTO knowledge_point_stats (user_id, subject, knowledge_point, correct_count, last_updated)
                        VALUES (?, ?, ?, 1, ?)
                        ON CONFLICT(user_id, subject, knowledge_point)
                        DO UPDATE SET correct_count = correct_count + 1, last_updated = ?
                    """, (row['user_id'], row['subject'], row['knowledge_point'],
                          datetime.now().isoformat(), datetime.now().isoformat()))

                conn.commit()
                return True
        except Exception as e:
            print(f"标记掌握失败: {e}")
            return False

    def delete_wrong_question(self, wrong_question_id: int) -> bool:
        """
        删除错题

        Args:
            wrong_question_id: 错题ID

        Returns:
            是否删除成功
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM wrong_questions WHERE id = ?", (wrong_question_id,))
                conn.commit()
                return True
        except Exception as e:
            print(f"删除错题失败: {e}")
            return False


class WeakPointAnalyzer:
    """薄弱知识点分析器"""

    def __init__(self):
        self.db = get_db()

    def get_weak_points(self, user_id: int, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取用户薄弱知识点

        Args:
            user_id: 用户ID
            subject: 学科（可选）

        Returns:
            薄弱知识点列表
        """
        result = []

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if subject:
                    cursor.execute("""
                        SELECT knowledge_point, wrong_count, correct_count,
                               (wrong_count - correct_count) as weakness_score
                        FROM knowledge_point_stats
                        WHERE user_id = ? AND subject = ?
                        ORDER BY weakness_score DESC
                        LIMIT 10
                    """, (user_id, subject))
                else:
                    cursor.execute("""
                        SELECT subject, knowledge_point, wrong_count, correct_count,
                               (wrong_count - correct_count) as weakness_score
                        FROM knowledge_point_stats
                        WHERE user_id = ?
                        ORDER BY weakness_score DESC
                        LIMIT 20
                    """, (user_id,))

                for row in cursor.fetchall():
                    result.append(dict(row))
        except Exception as e:
            print(f"获取薄弱知识点失败: {e}")

        return result

    def get_weak_point_report(self, user_id: int) -> Dict[str, Any]:
        """
        生成薄弱知识点报告

        Args:
            user_id: 用户ID

        Returns:
            报告数据
        """
        weak_points = self.get_weak_points(user_id)

        # 按学科分组
        by_subject = defaultdict(list)
        for wp in weak_points:
            subject = wp.get('subject', 'unknown')
            by_subject[subject].append(wp)

        # 找出最薄弱的知识点（错误次数最多）
        top_weak = sorted(weak_points, key=lambda x: x.get('weakness_score', 0), reverse=True)[:5]

        return {
            'total_weak_points': len(weak_points),
            'by_subject': dict(by_subject),
            'top_weak_points': top_weak,
            'generated_at': datetime.now().isoformat()
        }