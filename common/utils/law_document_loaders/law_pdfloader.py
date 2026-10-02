# 该文档用于加载PDF文件

import cv2                                  # 读取PDF文件的核心库，用于打开PDF读取每一页内容、提取图片
import fitz                                 # OpenCV图像处理库，可以对图片做预处理，让OCR识别更准确
import numpy as np
from PIL import Image                       # 图片处理库，用于打开、缩放、裁剪图片
from tqdm import tqdm
from typing import Iterator
from .law_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import CharacterTextSplitter

# PDF_OCR 控制：只对宽高超过页面一定比例(图片宽/页面宽, 图片高/页面高) 的图片进行OCR
# 作用：这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)

# OCR PDF加载器，用于加载PDF
class OCRPDFLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        """
        函数功能：初始化OCR PDF加载器
        :param file_path: PDF文件路径
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        函数功能：惰性加载PDF，返回一个生成器，每次返回一个PDF页面的文本内容
        :return: PDF页面的文本内容
        """
        line = self.pdf2text()
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self):
        """
        函数功能：将PDF文件转换为文本
        :return: PDF文件的文本内容
        """
        ocr = get_ocr()
        # 打开PDF文件
        doc = fitz.open(self.file_path)
        resp = ""
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")
        for i, page in enumerate(doc):
            b_unit.set_description("OCRPDFLoader context page index: {}".format(i))
            b_unit.refresh()
            # 提取文本：默认使用 “text” 模式提取文本
            text = page.get_text("text")
            resp += text + "\n"
            # 获取图片：获取所有显示的图像的原信息列表
            # 适用于所有文档类型，不仅限于PDF
            img_list = page.get_image_info(xrefs=True)
            for img in img_list:
                # xref一种编号，指向该图像对象在PDF文件中的位置，程序可以通过这个编号快速定位和提取图像数据
                if xref := img.get("xref"):
                    # 图像在页面上的位置和尺寸
                    bbox = img["bbox"]
                    # 检查图片尺寸是否超过设定的阈值
                    if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0] or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                        continue
                    # 提取图片原始数据
                    pix = fitz.Pixmap(doc, xref)
                    # 如果Page有旋转角度，则旋转图片
                    if int(page.rotation) != 0:
                        # 将原始数据变为图片矩阵
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        # 转换为PIL图片对象
                        tmp_img = Image.fromarray(img_array)
                        # 颜色格式转换 RGB -> BGR
                        ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                        # 旋转图片
                        rot_img = self.rotate_img(img=ori_img, angle=360 - page.rotation)
                        # 再转回RGB
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        # 若页面没有旋转，直接变成图像矩阵
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)
            # 更新进度
            b_unit.update(1)
        return resp

    def rotate_img(self, img, angle):
        """
        函数功能：旋转图片
        :param img: 图片
        :param angle: 旋转角度
        :return: 旋转后的图片
        """
        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        # 获取旋转矩阵
        # 参数1为旋转中心点;
        # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算图像新边界
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img

if __name__ == '__main__':
    pdf_loader = OCRPDFLoader(file_path=r"../test_data/1 婚姻家庭继承.pdf")
    doc = pdf_loader.load()
    print(type(doc))
    print(doc)

