"""数据准备与批采样。

分工：
    * 下载与切分语料 -> scripts/download_data.ps1（网络操作放脚本里，代码保持纯粹）
    * 读取 + 批采样  -> 本文件

目录约定：
    data/raw/<name>.txt       原始语料（UTF-8 纯文本）
    data/processed/<name>/    切分并编码后的结果，训练时直接读这里，跳过重复编码
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# 允许以脚本方式直接运行本文件（python common/data.py）时也能找到工程根目录
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.tokenizer import CharTokenizer  # noqa: E402

# ----------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# id 的存储类型：词表大小决定需要多少位。
# 字符级 TinyShakespeare 只有 65 个字符 -> uint8 足够；中文词表几千 -> uint16。
_UINT8_LIMIT = 256
_UINT16_LIMIT = 65536


def _id_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= _UINT8_LIMIT:
        return np.dtype(np.uint8)
    if vocab_size <= _UINT16_LIMIT:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


def raw_text_path(name: str) -> Path:
    return RAW_DIR / f"{name}.txt"


def processed_dir(name: str) -> Path:
    return PROCESSED_DIR / name


# ----------------------------------------------------------------------
# 步骤一：用语料构建词表并编码
# ----------------------------------------------------------------------
def prepare(name: str, val_ratio: float = 0.1, force: bool = False) -> Path:
    """把 data/raw/<name>.txt 处理成 train.bin / val.bin / tokenizer.json / meta.json。

    这一步只做一次，之后训练直接读 .bin，省掉每次重新编码的开销。
    """
    src = raw_text_path(name)
    if not src.exists():
        raise FileNotFoundError(
            f"找不到原始语料 {src}\n"
            "请先运行：powershell -File scripts\\download_data.ps1"
        )

    out_dir = processed_dir(name)
    meta_file = out_dir / "meta.json"
    if meta_file.exists() and not force:
        return out_dir

    text = src.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"语料文件为空：{src}")

    tokenizer = CharTokenizer.from_text(text)
    ids = np.array(tokenizer.encode(text), dtype=_id_dtype(tokenizer.vocab_size))

    split = int(len(ids) * (1.0 - val_ratio))
    if split <= 0 or split >= len(ids):
        raise ValueError(f"语料太短（{len(ids)} 个字符），无法切分训练/验证集")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = ids[:split], ids[split:]
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    tokenizer.save(out_dir / "tokenizer.json")

    meta = {
        "name": name,
        "vocab_size": tokenizer.vocab_size,
        "dtype": np.dtype(ids.dtype).name,
        "total_tokens": int(len(ids)),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "val_ratio": val_ratio,
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


# ----------------------------------------------------------------------
# 步骤二：读取已处理好的数据
# ----------------------------------------------------------------------
class TokenDataset:
    """一份已经编码好的语料：训练集 + 验证集 + 词表。

    用法：
        ds = TokenDataset.load("tinyshakespeare")
        print(ds.summary())
        x, y = ds.get_batch("train", batch_size=32, block_size=64)
    """

    def __init__(self, name: str, train: np.ndarray, val: np.ndarray,
                 tokenizer: CharTokenizer, meta: Dict[str, Any]):
        self.name = name
        self.train = train
        self.val = val
        self.tokenizer = tokenizer
        self.meta = meta

    # ---- 构造 ----
    @classmethod
    def load(cls, name: str = "tinyshakespeare", auto_prepare: bool = True) -> "TokenDataset":
        out_dir = processed_dir(name)
        meta_file = out_dir / "meta.json"
        if not meta_file.exists():
            if not auto_prepare:
                raise FileNotFoundError(f"数据未处理：{out_dir}")
            prepare(name)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        dtype = np.dtype(meta["dtype"])
        train = np.fromfile(out_dir / "train.bin", dtype=dtype)
        val = np.fromfile(out_dir / "val.bin", dtype=dtype)
        tokenizer = CharTokenizer.load(out_dir / "tokenizer.json")
        return cls(name, train, val, tokenizer, meta)

    # ---- 基本信息 ----
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def block_size(self) -> int:
        """语料本身能支持的最大上下文长度（实际用的由 GPTConfig 决定）。"""
        return min(len(self.train), len(self.val))

    def summary(self) -> str:
        return (
            f"数据集: {self.name}\n"
            f"  词表大小  : {self.vocab_size}\n"
            f"  训练 token: {len(self.train):,}\n"
            f"  验证 token: {len(self.val):,}\n"
            f"  存储 dtype: {self.train.dtype}"
        )

    # ---- 采样 ----
    def split(self, which: str = "train") -> np.ndarray:
        if which in ("train", "training"):
            return self.train
        if which in ("val", "valid", "validation"):
            return self.val
        raise ValueError(f"未知的数据划分 {which!r}，可选 'train' / 'val'")

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 32,
        block_size: int = 64,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机抽一批 (x, y)。

        x 是长度 block_size 的连续片段，y 是它在文本里右移一位的结果 —— 也就是
        预测下一个 token 这个任务的全部秘密：标签不用人工标注，文本自身就是标签。
        """
        data = self.split(split)
        if len(data) <= block_size + 1:
            raise ValueError(
                f"{split} 集只有 {len(data)} 个 token，放不下 block_size={block_size} 的样本"
            )
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
        ])
        if device is not None:
            x, y = x.to(device), y.to(device)
        return x, y


def load_dataset(name: str = "tinyshakespeare", auto_prepare: bool = True) -> TokenDataset:
    """便捷入口，等价于 TokenDataset.load(name)。"""
    return TokenDataset.load(name, auto_prepare=auto_prepare)


def available_datasets() -> list:
    """列出 data/raw 下已经下载好的语料名。"""
    if not RAW_DIR.exists():
        return []
    return sorted(p.stem for p in RAW_DIR.glob("*.txt"))


if __name__ == "__main__":
    # 直接运行本文件时做个自检：python common/data.py
    from common.utils import setup_console

    setup_console()
    names = available_datasets()
    print(f"已下载的语料: {names or '（无，请先运行 scripts/download_data.ps1）'}")
    if names:
        ds = load_dataset(names[0])
        print(ds.summary())
        xb, yb = ds.get_batch("train", batch_size=2, block_size=16)
        print("x shape:", tuple(xb.shape), "y shape:", tuple(yb.shape))
        print("x[0]:", ds.tokenizer.decode(xb[0].tolist()).replace("\n", "\\n"))
        print("y[0]:", ds.tokenizer.decode(yb[0].tolist()).replace("\n", "\\n"))
