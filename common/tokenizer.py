"""字符级 tokenizer：把文本和整数 id 相互转换。

这是 01 讲的主角。核心只有三步：
    1. 统计文本里出现过哪些字符（去重 + 排序 = 确定性的词表）
    2. 字符 -> id 用字典查
    3. 文本 -> id 列表 = 每个字符查一次

为什么先用字符级？
    * 不需要外部依赖，不会出现 [UNK]
    * 中文里一个汉字约等于一个 token，词表小、语义单位直观
    * 词表规模可控（几百到几千），Embedding 表很小，CPU 上跑得快

真实的大模型用 BPE / SentencePiece 做子词切分（词表 3 万到 20 万），
动机在 01 讲里会解释：字符级序列太长，子词能在词表大小与序列长度之间取平衡。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


class CharTokenizer:
    """最朴素的字符级 tokenizer。

    属性
    ----
    itos : list[str]      id -> 字符
    stoi : dict[str, int] 字符 -> id
    """

    def __init__(self, itos: Sequence[str]):
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("词表里有重复字符，请检查输入")

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """扫描文本，用出现过的所有字符组成词表（按码点排序保证可复现）。"""
        chars = sorted(set(text))
        if not chars:
            raise ValueError("输入文本为空，无法构建词表")
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ------------------------------------------------------------------
    # 编解码
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """文本 -> id 列表。遇到词表外的字符会明确报错，而不是静默变成 [UNK]。"""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise KeyError(
                f"字符 {e.args[0]!r} 不在词表中。字符级 tokenizer 无法处理词表外字符，"
                "请确认 encode 的文本与构建词表时的语料是同一批。"
            ) from None

    def decode(self, ids: Iterable[int]) -> str:
        """id 列表 -> 文本。"""
        return "".join(self.itos[int(i)] for i in ids)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        head = "".join(repr(c) for c in self.itos[:8])
        return f"CharTokenizer(vocab_size={self.vocab_size}, first8={head})"

    # ------------------------------------------------------------------
    # 存取（训练出的模型要配着自己的词表用）
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"type": "char", "itos": self.itos}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "char":
            raise ValueError(f"不是字符级词表文件：{path}")
        return cls(payload["itos"])

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def describe(self, text: str | None = None) -> str:
        """打印词表摘要，可选用一段文本展示字符与 id 的对应关系。"""
        lines = [f"词表大小: {self.vocab_size}"]
        preview = "".join(self.itos[:40]).replace("\n", "\\n")
        lines.append(f"前 40 个字符: {preview!r}")
        if text:
            ids = self.encode(text[:32])
            lines.append(f"原文: {text[:32]!r}")
            lines.append(f"编码: {ids}")
            lines.append(f"解码: {self.decode(ids)!r}")
        return "\n".join(lines)
