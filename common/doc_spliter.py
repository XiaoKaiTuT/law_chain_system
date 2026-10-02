import os, sys

# 将项目根目录添加到系统路径，目的是为了让Python能找到项目中的模块
current_dir: str = os.path.dirname(os.path.abspath(__file__))
mysql_qa_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(mysql_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
from common.utils.law_text_spliter import split_cases, split_law
from common.doc_loader import load_txt_from_directory
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

FILE_TYPE_MAP = {
    "law": split_law,
    "case": split_cases
}

def split_dict_from_txt(document: list[tuple]) -> list[dict]:
    """
    函数功能：将文本进行分割
    :param document: 元组(需要切割的文本, 文件用途，例如 "law" 或 "case")
    :return: 分割后的字典列表
    """
    logger.info(f"开始分割文本，文本数量: {len(document)}")
    # 记录分割后的字典列表
    result = []
    for text, file_type, source in document:
        try:
            spliter = FILE_TYPE_MAP[file_type]
            preprocess_text = spliter("\n".join(doc.page_content for doc in text), source=source)
            result.extend(preprocess_text)
        except Exception as e:
            logger.error(f"文档分割失败: {e}")
    logger.info(f"文档切割完成")
    return result

# TODO 测试函数
if __name__ == '__main__':
    dir_path = r"D:\develop\PyCharm 2026.1.2\workspace\law_chain_system\common\data"
    documents = load_txt_from_directory(dir_path)
    split_result = split_dict_from_txt(documents)
    print(len(split_result))
    print(split_result)
