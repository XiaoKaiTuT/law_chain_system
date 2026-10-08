import os, sys
import torch
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# 模型参数
MODEL_DIR = Path(rag_qa_dir) / "models"
MODEL_NAME = "legal_classifier"
MAX_LENGTH = 128
LABELS = ["日常", "法律"]

# 全局变量
_classifier_instance = None

class ClassifyTools:
    def __init__(self):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(Path(MODEL_DIR) / MODEL_NAME))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(Path(MODEL_DIR) / MODEL_NAME)).to(self.device)
            if self.device == "cuda":
                self.model.half()
        except OSError as e:
            self.logger.error(f"模型加载失败，请确认模型存在: {Path(MODEL_DIR) / MODEL_NAME}")
            self.logger.error(f"原始错误: {e}")
            raise

    @staticmethod
    def _normalize(text: str) -> str:
        """
        函数功能：文本预处理
        :param text: 文本
        :return: 处理后的文本
        """
        return text.strip().rstrip("?？").strip()

    def classify(self, text: str) -> str:
        """
        函数功能：对文本进行法律标签分类
        :param text: 需要分类的文本
        :return: 文本的法律标签，以及标签的概率
        """
        text = self._normalize(text)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encoded).logits

        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs))

        label = LABELS[idx]
        self.logger.info(f"文本：{text}, 标签：{label}")

        # 对概率进行判断，输出不同级别的日志
        prob = float(probs[0][idx])
        if prob < 0.7:
            logger.warning(f"低置信度判断: {label} ({prob:.3f} | {text})")
        else:
            logger.info(f"高置信度判断: {label} ({prob:.3f} | {text})")

        return label

def get_legal_classify() -> ClassifyTools:
    """
    函数功能：获取全局单例 (模型只加载一次)
    :return: ClassifyTools 单例
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ClassifyTools()
    return _classifier_instance

# TODO 测试函数
if __name__ == '__main__':
    legal_classifier = ClassifyTools()
    text = [
        "今天天气怎么样",
        "押金不退",
        "你好，我想问下离婚财产怎么分"
    ]
    for question in text:
        classify = legal_classifier.classify(question)



