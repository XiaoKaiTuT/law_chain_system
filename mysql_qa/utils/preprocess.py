import os
import jieba
from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

def preprocess_text(text: str) -> list:
    """
    函数功能：将文本进行预处理，包括：转小写、清理多余空格/换行/制表符、分词
    :param text: 文本
    :return: 分词后的结果
    """
    try:
        return jieba.lcut("".join(text.strip().split()).lower())
    except AttributeError as e:
        logger.error(f'文本预处理失败: {e}')
        return []

# TODO 测试函数
if __name__ == '__main__':
    test_text = "欢迎使用百度搜索 BaiDuSearch"
    print(preprocess_text(test_text))