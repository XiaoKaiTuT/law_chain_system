# 该脚本用于创建并返回日志记录器

import logging
import os
from .config import config

LOG_FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - Line:%(lineno)d - %(message)s"
LOG_CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logger(file_name: str, log_file: str=config.LOG_FILE):
    """
    函数功能：创建并返回日志记录器，支持日志同时输出到控制台和文件，避免重复添加处理器
    :param log_file: 日志文件保存路径
    :return: 日志记录器
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    # 判断处理器是否存在，避免重复添加
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        console_formatter = logging.Formatter(LOG_CONSOLE_FORMAT)
        file_formatter = logging.Formatter(LOG_FILE_FORMAT)
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# 测试函数
if __name__ == '__main__':
    logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])
    logger.info("This is a test log.")
