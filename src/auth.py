"""
简单会话登录校验（Streamlit 会话内）
支持从数据库验证用户登录
"""
import secrets
from typing import Optional

from config.settings import ROLE_ADMIN, ROLE_STUDENT, ROLE_TEACHER, load_auth_users
from src.database.models import get_db
from src.user.user_manager import UserManager


def is_staff(role: Optional[str]) -> bool:
    """教师或管理员可使用知识库管理等权限。"""
    return role in (ROLE_TEACHER, ROLE_ADMIN)


def is_admin(role: Optional[str]) -> bool:
    """是否为管理员"""
    return role == ROLE_ADMIN


def verify_login(username: str, password: str) -> Optional[str]:
    """
    验证用户名密码，成功返回角色，失败返回 None。
    支持两种验证方式：数据库验证和配置文件验证
    """
    if not username or not password:
        return None

    # 首先尝试数据库验证
    user_manager = UserManager()
    user_info = user_manager.verify_user_login(username, password)
    if user_info:
        # 保存用户ID到会话状态（需要在app.py中处理）
        return user_info['role']

    # 兜底：配置文件验证
    users = load_auth_users()
    info = users.get(username.strip())
    if not info:
        return None
    ok = secrets.compare_digest(
        password.encode("utf-8"),
        str(info["password"]).encode("utf-8"),
    )
    return str(info["role"]) if ok else None


def init_database_users():
    """初始化数据库用户表"""
    user_manager = UserManager()
    user_manager.init_default_users()