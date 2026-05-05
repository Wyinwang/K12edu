"""
数据库初始化脚本
运行此脚本初始化数据库和默认用户
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.models import DatabaseManager
from src.user.user_manager import UserManager
from config.settings import ROLE_STUDENT, ROLE_TEACHER, ROLE_ADMIN


def init_database():
    """初始化数据库"""
    print("正在初始化数据库...")
    db = DatabaseManager()
    print(f"数据库已创建: {db.db_path}")

    # 初始化默认用户
    user_manager = UserManager()
    user_manager.init_default_users()
    print("默认用户已创建:")
    print("  - student / student123 (学生)")
    print("  - teacher / teacher123 (教师)")
    print("  - admin / admin123 (管理员)")


if __name__ == "__main__":
    init_database()