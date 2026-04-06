"""
简单会话登录校验（Streamlit 会话内）
"""
import secrets
from typing import Optional

from config.settings import ROLE_ADMIN, ROLE_STUDENT, ROLE_TEACHER, load_auth_users


def is_staff(role: Optional[str]) -> bool:
    """教师或管理员可使用知识库管理等权限。"""
    return role in (ROLE_TEACHER, ROLE_ADMIN)


def verify_login(username: str, password: str) -> Optional[str]:
    """验证用户名密码，成功返回角色，失败返回 None。"""
    if not username or not password:
        return None
    users = load_auth_users()
    info = users.get(username.strip())
    if not info:
        return None
    ok = secrets.compare_digest(
        password.encode("utf-8"),
        str(info["password"]).encode("utf-8"),
    )
    return str(info["role"]) if ok else None
