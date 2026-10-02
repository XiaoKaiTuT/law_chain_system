# 条件导入各个文档加载器，允许在缺少依赖时继续运行
try:
    from .law_docloader import OCRDOCLoader
except ImportError:
    pass

try:
    from .law_pptloader import OCRPPTLoader
except ImportError:
    pass

try:
    from .law_imgloader import OCRIMGLoader
except ImportError:
    pass

try:
    from .law_pdfloader import OCRPDFLoader
except ImportError:
    pass