# 该脚本用于加载图片

from typing import Iterator
from .law_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

# OCR图片加载器，用于加载图片
class OCRIMGLoader(BaseLoader):
    def __init__(self, img_path: str) -> None:
        """
        函数功能：初始化OCR图片加载器
        :param img_path: 图片路径
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        """
        函数功能：惰性加载图片，返回一个生成器，每次返回一个图片的文本内容
        :return: 图片的文本内容
        """
        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        """
        函数功能：图片转文本
        :return: 图片的文本内容
        """
        resp = ""
        ocr = get_ocr()
        result, _ = ocr(self.img_path)
        if result:
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp

if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path=r'../test_data/Day01_随堂图片.png')
    doc = img_loader.load()
    # [Document(metadata={'source': '...'}, page_content='...'), ...]
    print(doc)
