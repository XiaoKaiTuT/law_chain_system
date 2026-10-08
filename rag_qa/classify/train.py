import os
import json
import sys
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
logger = setup_logger("train")

# 参数设置
MODEL_NAME = "chinese-roberta-wwm-ext"                  # 模型名称
DATA_DIR = Path(current_dir) / "data"                   # 数据路径
MODEL_DIR = Path(project_dir) / "rag_qa" / "models"     # 模型路径
OUTER_DIR = Path(current_dir) / "models"
MAX_LENGTH = 128                                        # 最长序列长度
LABELS = ["日常", "法律"]                                 # 标签label：0=日常
LEARNING_RATE = 2e-5                                    # 学习率
EPOCHS = 4                                              # 轮数
BATCH_SIZE = 32                                         # 批处理大小
WEIGHT_DECAY = 0.01                                     # 权重衰减
WARMUP_RATIO = 0.1                                      # 预热比例
SEED = 22                                               # 随机种子

class TrainClassifier:
    def __init__(self):
        self.logger = logger

    def load_jsonl(self, path: Path) -> list[dict]:
        """
        函数功能：加载 JSONL 文件
        :param path: 文件路径
        :return: 数据列表
        """
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")
        records = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"{path.name} 第 {lineno} 行解析失败，跳过：{e}")
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("text"), str) and obj["text"].strip():
                    records.append({"text": obj["text"].strip(), "label": int(obj["label"])})
        return records

    def build_datasets(self, tokenizer):
        """
        函数功能：加载 train / dev / test 并转为 tokenized Dataset
        :param tokenizer: 分词器
        :return: 数据集
        """
        splits = {}
        for name in {"train", "dev", "test"}:
            raw = self.load_jsonl(DATA_DIR / f"{name}.jsonl")
            ds = Dataset.from_list(raw)
            ds = ds.map(
                lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
                batched=True,
            )
            # HuggingFace Trainer要求标签列名叫labels，否则训练时报错
            ds = ds.rename_column("label", "labels")
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            splits[name] = ds
            logger.info(f"{name}: {len(ds)} 条")
        return splits["train"], splits["dev"], splits["test"]

    @staticmethod
    def compute_metrics(eval_pred) -> dict:
        """
        函数功能：评估指标
        :param eval_pred: 原始分数、标签
        :return: 评估指标
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=[0, 1], zero_division=0
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1.mean(),
            "f1_daily": f1[0],
            "f1_legal": f1[1],
            "recall_legal": recall[1],
            "precision_legal": precision[1],
        }

    def train_model(self):
        """
        函数功能：训练模型
        :return:
        """
        self.logger.info(f"使用模型: {MODEL_NAME}")
        self.logger.info(f"设备：{'cuda' if torch.cuda.is_available() else 'cpu'}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # 加载模型分词器
        tokenizer = AutoTokenizer.from_pretrained(Path(MODEL_DIR) / MODEL_NAME)
        train_ds, dev_ds, test_ds = self.build_datasets(tokenizer)

        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(Path(MODEL_DIR) / MODEL_NAME, num_labels=2)
        model.config.id2label = {0: LABELS[0], 1: LABELS[1]}
        model.config.label2id = {LABELS[0]: 0, LABELS[1]: 1}

        args = TrainingArguments(
            output_dir=str(OUTER_DIR),                              # 模型保存路径
            learning_rate=LEARNING_RATE,                            # 学习率
            per_device_train_batch_size=BATCH_SIZE,                 # 训练批次处理大小
            per_device_eval_batch_size=BATCH_SIZE * 2,              # 测试批次处理大小
            num_train_epochs=EPOCHS,                                # 轮数
            weight_decay=WEIGHT_DECAY,                              # 权重衰减
            warmup_ratio=WARMUP_RATIO,                              # 预热比例
            seed=SEED,                                              # 随机种子
            eval_strategy="epoch",                                  # 每轮结束时评估
            save_strategy="epoch",                                  # 每轮结束时保存
            metric_for_best_model="f1",                             # 最佳模型评估指标
            greater_is_better=True,                                 # True表示指标越高越好
            load_best_model_at_end=True,                            # 训练结束后，加载指标最佳模型
            save_total_limit=2,                                     # 最多保留的检查点数量
            logging_steps=20,                                       # 打印日志间隔
            report_to="none",                                       # 不报告给任何地方
            fp16=torch.cuda.is_available(),                         # 是否使用半精度浮点数
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),               # 自动 padding 工具
            compute_metrics=self.compute_metrics,                           # 评估指标函数
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],   # 早停回调
        )

        self.logger.info("开始训练")
        trainer.train()

        # 保存模型
        OUTER_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(OUTER_DIR))
        tokenizer.save_pretrained(str(OUTER_DIR))
        self.logger.info(f"最佳模型已保存到：{OUTER_DIR}")

        # dev 详细报告
        self.logger.info("验证集 (dev) 详细报告")
        dev_pred = trainer.predict(dev_ds)
        dev_preds = np.argmax(dev_pred.predictions, axis=-1)
        self._print_report(dev_pred.label_ids, dev_preds)

        # test 最终报告
        self.logger.info("测试集 (test) 最终报告")
        test_pred = trainer.predict(test_ds)
        test_preds = np.argmax(test_pred.predictions, axis=-1)
        self._print_report(test_pred.label_ids, test_preds)

    def _print_report(self, y_true, y_pred) -> None:
        """
        函数功能：分类报告 + 混淆矩阵
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :return: None
        """
        # 打印分类报告
        report = classification_report(
            y_true,
            y_pred,
            target_names=LABELS,        # 标签映射名称
            digits=4,                   # 小数位数
            zero_division=0,            # 除以0，返回0，不报错
        )
        self.logger.info("\n" + report)

        # 打印混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        self.logger.info(f"混淆矩阵\n{cm}")

        # 关键解读：法律被潘成日常 = 最严重错误
        legal_to_daily = cm[1][0]
        daily_to_legal = cm[0][1]
        self.logger.info(f"\n  ⚠️ 法律→日常（漏判，最严重）: {legal_to_daily} 条")
        self.logger.info(f"     日常→法律（多走一次检索，代价低）: {daily_to_legal} 条")

def main():
    trainer = TrainClassifier()
    trainer.train_model()

if __name__ == '__main__':
    main()




