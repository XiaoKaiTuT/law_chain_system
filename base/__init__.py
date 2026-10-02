# 该脚本用于初始化项目目录和日志记录器

import os
import sys

base_dir: str = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
# 判断项目目录是否在sys.path中，如果不在则添加
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from .config import config
from .logger import setup_logger