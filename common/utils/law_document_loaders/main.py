import os
import sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
common_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(common_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

def main():
    logger.info("文档加载运行测试开始")

    try:
        from common.utils.law_document_loaders import OCRDOCLoader
        docx_loader = OCRDOCLoader("../test_data/中华人民共和国民事诉讼法（《关于修改〈中华人民共和国民事诉讼法〉的决定》第五次修正））.docx")
        doc = docx_loader.load()
        logger.info(f"文档加载成功")
        print(f'doc -> {doc}')
    except Exception as e:
        logger.error(f"文档加载失败：{e}")

    try:
        from common.utils.law_document_loaders import OCRIMGLoader
        img_loader = OCRIMGLoader("../test_data/Day01_随堂图片.png")
        img = img_loader.load()
        logger.info(f"图片加载成功")
        print(f'img -> {img}')
    except Exception as e:
        logger.error(f"图片加载失败：{e}")

    try:
        from common.utils.law_document_loaders import OCRPDFLoader
        pdf_loader = OCRPDFLoader("../test_data/1 婚姻家庭继承.pdf")
        pdf_loader = pdf_loader.load()
        logger.info(f"PDF加载成功")
        print(f'pdf -> {pdf_loader}')
    except Exception as e:
        logger.error(f"PDF加载失败：{e}")

    try:
        from common.utils.law_document_loaders import OCRPPTLoader
        ppt_loader = OCRPPTLoader("../test_data/00-深度学习简介.pptx")
        ppt_loader = ppt_loader.load()
        logger.info(f"PPT加载成功")
        print(f'ppt -> {ppt_loader}')
    except Exception as e:
        logger.error(f"PPT加载失败：{e}")

    logger.info("文档加载测试完成")

if __name__ == '__main__':
    main()