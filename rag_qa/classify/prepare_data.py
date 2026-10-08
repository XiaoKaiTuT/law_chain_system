import json
import os, sys
import random
from collections import defaultdict
from pathlib import Path

current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
logger = setup_logger("prepare_data")

DATA_DIR = Path(current_dir) / "data"       # 数据目录
RAW = DATA_DIR / "raw.jsonl"                # 原始数据文件
SEED = 22                                   # 随机种子
RATIOS = (0.8, 0.1, 0.1)                    # train / dev / test
LABEL_NAMES = {0: "日常", 1: "法律"}          # 标签映射
GREETINGS = ("你好", "您好", "请问", "麻烦", "咨询", "想问", "请教", "打扰", "谢谢", "多谢", "辛苦", "嗨", "哈喽", "在吗")

class PrepareData:
    def __init__(self):
        self.logger = logger

    def load_raw(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path}")
        records, bad = [], 0
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                self.logger.warning(f"第 {lineno} 行解析失败, 跳过：{e}")
                bad += 1
                continue
            text, label = obj.get("text"), obj.get("label")
            if not isinstance(text, str) or not text.strip():
                bad += 1
                continue
            if label not in (0, 1):
                bad += 1
                continue
            records.append({"text": text.strip(), "label": int(label)})
        logger.info(f"读入 {len(records)} 条 (跳过 {bad} 条异常)")
        return records

    @staticmethod
    def normalize(text: str) -> str:
        """
        函数功能：删除末尾问号和多余空白
        :param text: 文本
        :return: 预处理后的文本
        """
        return text.strip().rstrip("?？").strip()

    @staticmethod
    def dedup(records: list[dict]) -> list[dict]:
        """
        函数功能：防止数据泄露，重复
        :param records: 数据列表
        :return: 去重后的数据列表
        """
        seen, out, removed = set(), [], 0
        for r in records:
            key = r["text"]
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            out.append(r)
        logger.info(f"去重：移除 {removed} 条，剩余 {len(out)} 条")
        return out

    @staticmethod
    def feature_key(r:dict) -> str:
        """
        函数功能：组合特征键，用于分层抽样，四大特征：label/长度/是否带问候/是否第一人称
        :param r: 数据记录
        :return: 组合特征键
        """
        t, lab = r["text"], r["label"]
        if len(t) <= 6:
            ln = "xs"  # 极短
        elif len(t) <= 15:
            ln = "s"  # 短
        elif len(t) > 40:
            ln = "l"  # 长
        else:
            ln = "m"  # 中
        g = "G" if any(w in t for w in GREETINGS) else "g"
        p = "P" if "我" in t[:10] else "p"
        return f"{lab}_{ln}_{g}_{p}"

    def stratified_split(self, records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """
        函数功能：分桶策略，对数据进行分层抽样
        :param records: 数据列表
        :return: 训练集/验证集/测试集
        """
        # 进行分桶
        rng = random.Random(SEED)
        bucket = defaultdict(list)
        for r in records:
            bucket[self.feature_key(r)].append(r)

        # 初始化 训练集/验证集/测试集/合并集
        train, dev, test = [], [], []
        merged = []

        # 对每个桶进行打乱并分层装入 训练集/验证集/测试集/合并集
        for key, items in sorted(bucket.items()):
            rng.shuffle(items)
            n = len(items)
            if n < 10:
                merged.extend(items)
                continue
            n_dev = max(1, round(n * RATIOS[1]))
            n_test = max(1, round(n * RATIOS[2]))
            n_train = n - n_dev - n_test
            train.extend(items[:n_train])
            dev.extend(items[n_train:n_train + n_dev])
            test.extend(items[n_train + n_dev:])

        # 将合并集按照 label 进行处理
        if merged:
            logger.warning(f"有 {len(merged)} 条样本所在的特征桶过小，已合并后按 label 切分")
            by_label = defaultdict(list)
            for r in merged:
                by_label[r["label"]].append(r)
            for lab, items in sorted(by_label.items()):
                rng.shuffle(items)
                n = len(items)
                n_dev = max(1, round(n * RATIOS[1]))
                n_test = max(1, round(n * RATIOS[2]))
                n_train = n - n_dev - n_test
                train.extend(items[:n_train])
                dev.extend(items[n_train: n_train + n_dev])
                test.extend(items[n_train + n_dev:])

        for part in (train, dev, test):
            rng.shuffle(part)
        return train, dev, test

    @staticmethod
    def save_jsonl(path: Path, records: list[dict]) -> None:
        """
        函数功能：保存 JSONL 文件
        :param path: 保存路径
        :param records: 写入的数据记录
        :return: None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def stat(records: list[dict]) -> dict:
        """
        函数功能：对数据进行统计
        :param records: 数据列表
        :return: 数据统计信息
        """
        n = len(records)
        if n == 0:
            return {}
        s = {
            "总数": n,
            "日常(label=0)": sum(1 for r in records if r["label"] == 0),
            "法律(label=1)": sum(1 for r in records if r["label"] == 1),
            "带问候词": sum(1 for r in records if any(w in r["text"] for w in GREETINGS)),
            "长句(>40字)": sum(1 for r in records if len(r["text"]) > 40),
            "极短(<=6字)": sum(1 for r in records if len(r["text"]) <= 6),
            "第一人称": sum(1 for r in records if "我" in r["text"][:10]),
        }
        return s

    def print_report(self, splits: dict):
        """
        函数功能：对数据进行统计并打印报告
        :param splits: 数据切分
        :return:None
        """
        names = list(splits.keys())
        stats = {k: self.stat(v) for k, v in splits.items()}
        keys = [k for k in stats["train"]]

        print("切分结果统计 (比例应基本一致，否则说明分层有问题)")
        head = f"{'指标':<16}" + "".join(f"{n:>12}" for n in names) + f"{'合计':>12}"
        print(head)
        print("  " + "-" * (16 + 12 * (len(names) + 1)))
        for k in keys:
            row = f"  {k:<16}"
            total = 0
            for n in names:
                v = stats[n].get(k, 0)
                total += v
                row += f"{v:>12}"
            row += f"{total:>12}"
            print(row)
        print()
        # 比例视图，更容易看出偏差
        print("  占比视图（%）:")
        for k in keys:
            if k == "总数":
                continue
            row = f"  {k:<16}"
            for n in names:
                tot = stats[n]["总数"]
                row += f"{stats[n].get(k, 0) / tot * 100:>11.1f}%"
            print(row)

def main():
    prepare_data = PrepareData()
    raw = prepare_data.load_raw(RAW)
    for r in raw:
        r["text"] = prepare_data.normalize(r["text"])
    raw = [r for r in raw if r["text"]]
    raw = prepare_data.dedup(raw)

    train, dev, test = prepare_data.stratified_split(raw)
    splits = {"train": train, "dev": dev, "test": test}

    # 完整性检查：不能有遗漏或重复
    total = len(train) + len(dev) + len(test)
    assert total == len(raw), f"切分后总数 {total} != 原始 {len(raw)}"
    all_texts = [r["text"] for r in train + dev + test]
    assert len(all_texts) == len(set(all_texts)), "切分后出现重复样本！"
    # 三份之间不能有交集（数据泄漏检查）
    s_train, s_dev, s_test = (set(r["text"] for r in p) for p in (train, dev, test))
    assert not (s_train & s_dev), "train 与 dev 有重叠！"
    assert not (s_train & s_test), "train 与 test 有重叠！"
    assert not (s_dev & s_test), "dev 与 test 有重叠！"

    for name, part in splits.items():
        prepare_data.save_jsonl(DATA_DIR / f"{name}.jsonl", part)
        logger.info(f"已写出 {name}.jsonl: {len(part)} 条")

    prepare_data.print_report(splits)

    logger.info("完成。三份数据无重叠、无重复。")

# TODO 测试函数
if __name__ == '__main__':
    main()
