"""
日志配置模块
配置系统的日志记录
"""
import os
import logging
from datetime import datetime
from config.settings import BASE_DIR

# 日志目录
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件路径
LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    设置并返回一个配置好的logger

    Args:
        name: logger名称
        level: 日志级别

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if not logger.handlers:
        # 文件handler
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)

    return logger


# 创建默认logger
default_logger = setup_logger("app")

# 创建各模块专用logger
document_logger = setup_logger("document")
vectorstore_logger = setup_logger("vectorstore")
rag_logger = setup_logger("rag")
