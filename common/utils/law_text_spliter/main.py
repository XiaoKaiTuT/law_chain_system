import sys
import os

current_dir: str = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(common_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from common.utils.law_document_loaders import OCRDOCLoader, OCRPDFLoader
from common.utils.law_text_spliter import split_law, split_cases
from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

def main():
    try:
        docx_loader = OCRDOCLoader(
            r"/common/utils/test_data\中华人民共和国民事诉讼法（《关于修改〈中华人民共和国民事诉讼法〉的决定》第五次修正））.docx")
        logger.info("文档加载完成")
        law_text = "\n".join(doc.page_content for doc in docx_loader.load())
        print(law_text)
        logger.info("文档预处理完成")
        law_chunks = split_law(law_text, source="民事诉讼法.docx")
        logger.info("文档分块完成")
        print(len(law_chunks), law_chunks)
    except Exception as e:
        logger.error(f"文档处理失败: {e}")

    try:
        pdf_loader = OCRPDFLoader(r"/common/utils/test_data\1 婚姻家庭继承.pdf")
        logger.info("文档加载完成")
        case_text = "\n".join(doc.page_content for doc in pdf_loader.load())
        logger.info("文档预处理完成")
        case_chunks = split_cases(case_text, source="1 婚姻家庭继承.pdf")
        logger.info("文档分块完成")
        print(len(case_chunks), case_chunks[0])
    except Exception as e:
        logger.error(f"文档处理失败: {e}")

if __name__ == '__main__':
    main()