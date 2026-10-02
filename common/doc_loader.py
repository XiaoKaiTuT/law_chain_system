import os, sys
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader

# 将项目根目录添加到系统路径，目的是为了让Python能找到项目中的模块
current_dir: str = os.path.dirname(os.path.abspath(__file__))
mysql_qa_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(mysql_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
from common.utils.law_document_loaders import OCRDOCLoader, OCRPPTLoader, OCRIMGLoader, OCRPDFLoader
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# 文件类型映射表
FILE_LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader,
}

def _load_single_file(file_path: str) -> list:
    """
    函数功能：加载单个文件，返回 Document 列表
    :param file_path: 文件路径
    :return: 列表
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in FILE_LOADER_MAP:
        logger.warning(f"文件类型暂不支持: {ext}")
        return []
    loader = FILE_LOADER_MAP[ext]
    if ext == ".txt":
        return loader(file_path, encoding="utf-8").load()
    return loader(file_path).load()

def _detect_file_type(path: str) -> str | None:
    """
    函数功能：从父目录名推断 file_type
    :param path: 文件路径
    :return: 文件用途，例如 "law" 或 "case"，如果无法推断则返回 None
    """
    parent_name = os.path.basename(os.path.dirname(path))
    if parent_name in ['law', 'case']:
        return parent_name
    return None

def load_txt_from_directory(directory_path: str, file_type=None) -> list[tuple]:
    """
    函数功能：从目录中(含子目录)/文件中加载所有支持类型的文件
    :param directory_path: 目标文件夹/文件的绝对路径
    :param file_type: 文件用途，例如 "law" 或 "case"
    :return: (数据, 文件用途, 文件名) 的元组列表
    """
    logger.info(f"开始加载路径: {directory_path}")
    documents = []

    # 文件用途填写错误
    if file_type and file_type not in ['law', 'case']:
        logger.warning(f"文件用途填写错误: {file_type}")
        file_type = None

    # 1. 如果是文件
    if os.path.isfile(directory_path):
        ft = file_type or _detect_file_type(directory_path)
        if not ft:
            logger.warning("读取失败，文件用途未指定")
            return []
        docs = _load_single_file(directory_path)
        if docs:
            documents.append((docs, ft, os.path.basename(directory_path)))
            logger.info(f"成功加载文件: {os.path.basename(directory_path)}")

    # 2. 如果是目录
    elif os.path.isdir(directory_path):
        for dirpath, _, files in os.walk(directory_path):
            for file in files:
                ft = file_type or _detect_file_type(os.path.join(dirpath, file))
                if not ft:
                    logger.warning(f"跳过，文件用途未指定: {file}")
                    break
                docs = _load_single_file(os.path.join(dirpath, file))
                if docs:
                    documents.append((docs, ft, file))
                    logger.info(f"成功加载文件: {file}")
    # 3. 无效路径
    else:
        logger.warning(f"无效的文件路径: {directory_path}")

    return documents

# TODO 测试函数
if __name__ == '__main__':
    dir_path = r"D:\develop\PyCharm 2026.1.2\workspace\law_chain_system\common\data"
    documents = load_txt_from_directory(dir_path)
    print(f'加载的文件数量: {len(documents)}，读取后数据: {documents[0][0]}')
