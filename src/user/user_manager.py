"""
用户管理模块
支持用户增删改查操作
"""
import os
import sys
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.models import get_db
from config.settings import ROLE_STUDENT, ROLE_TEACHER, ROLE_ADMIN


class UserManager:
    """用户管理器"""

    def __init__(self):
        self.db = get_db()

    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def create_user(
        self,
        username: str,
        password: str,
        role: str = ROLE_STUDENT
    ) -> Dict[str, Any]:
        """
        创建新用户

        Args:
            username: 用户名
            password: 密码
            role: 角色

        Returns:
            创建结果
        """
        if not username or not password:
            return {"success": False, "message": "用户名和密码不能为空"}

        if role not in [ROLE_STUDENT, ROLE_TEACHER, ROLE_ADMIN]:
            return {"success": False, "message": "无效的角色类型"}

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # 检查用户名是否已存在
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    return {"success": False, "message": "用户名已存在"}

                # 创建用户
                password_hash = self._hash_password(password)
                cursor.execute("""
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?)
                """, (username, password_hash, role, datetime.now().isoformat()))
                conn.commit()

                return {"success": True, "message": "用户创建成功", "user_id": cursor.lastrowid}
        except Exception as e:
            return {"success": False, "message": f"创建失败: {str(e)}"}

    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        获取所有用户列表

        Returns:
            用户列表
        """
        result = []

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, role, created_at, last_login_at, is_active
                    FROM users
                    ORDER BY created_at DESC
                """)

                for row in cursor.fetchall():
                    result.append(dict(row))
        except Exception as e:
            print(f"获取用户列表失败: {e}")

        return result

    def update_user(
        self,
        user_id: int,
        role: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        更新用户信息

        Args:
            user_id: 用户ID
            role: 新角色（可选）
            password: 新密码（可选）

        Returns:
            更新结果
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # 检查用户是否存在
                cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
                if not cursor.fetchone():
                    return {"success": False, "message": "用户不存在"}

                updates = []
                params = []

                if role and role in [ROLE_STUDENT, ROLE_TEACHER, ROLE_ADMIN]:
                    updates.append("role = ?")
                    params.append(role)

                if password:
                    updates.append("password_hash = ?")
                    params.append(self._hash_password(password))

                if not updates:
                    return {"success": False, "message": "没有需要更新的内容"}

                params.append(user_id)
                cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
                conn.commit()

                return {"success": True, "message": "用户信息更新成功"}
        except Exception as e:
            return {"success": False, "message": f"更新失败: {str(e)}"}

    def delete_user(self, user_id: int) -> Dict[str, Any]:
        """
        删除用户

        Args:
            user_id: 用户ID

        Returns:
            删除结果
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # 检查用户是否存在
                cursor.execute("SELECT id, role FROM users WHERE id = ?", (user_id,))
                user = cursor.fetchone()
                if not user:
                    return {"success": False, "message": "用户不存在"}

                # 删除用户相关数据
                cursor.execute("DELETE FROM learning_records WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM exercise_records WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM wrong_questions WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM knowledge_point_stats WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()

                return {"success": True, "message": "用户已删除"}
        except Exception as e:
            return {"success": False, "message": f"删除失败: {str(e)}"}

    def update_last_login(self, user_id: int) -> bool:
        """
        更新用户最后登录时间

        Args:
            user_id: 用户ID

        Returns:
            是否更新成功
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET last_login_at = ? WHERE id = ?
                """, (datetime.now().isoformat(), user_id))
                conn.commit()
                return True
        except Exception as e:
            print(f"更新登录时间失败: {e}")
            return False

    def verify_user_login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        验证用户登录（使用数据库）

        Args:
            username: 用户名
            password: 密码

        Returns:
            用户信息或None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, password_hash, role, is_active
                    FROM users WHERE username = ?
                """, (username,))
                user = cursor.fetchone()

                if not user:
                    return None

                if not user['is_active']:
                    return None

                password_hash = self._hash_password(password)
                if user['password_hash'] != password_hash:
                    return None

                # 更新登录时间
                self.update_last_login(user['id'])

                return {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role']
                }
        except Exception as e:
            print(f"验证登录失败: {e}")
            return None

    def init_default_users(self) -> None:
        """
        初始化默认用户
        如果数据库中没有用户，则创建默认账号
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM users")
                if cursor.fetchone()['count'] == 0:
                    # 创建默认账号
                    self.create_user("student", "student123", ROLE_STUDENT)
                    self.create_user("teacher", "teacher123", ROLE_TEACHER)
                    self.create_user("admin", "admin123", ROLE_ADMIN)
                    print("已创建默认用户账号")
        except Exception as e:
            print(f"初始化默认用户失败: {e}")