# 该脚本用于获取OCR识别器

def get_ocr(use_cuda = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        # 参:1：启用检测模型的GPU加速      参2：启用分类模型的GPU加速     参3：启用识别模型的GPU加速
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr

# 测试函数
if __name__ == '__main__':
    ocr = get_ocr()
    # <rapidocr_paddle.main.RapidOCR object at 0x0000015D0C1E0130>
    print(ocr)