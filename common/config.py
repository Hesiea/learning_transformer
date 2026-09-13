"""GPT 超参数配置。

这个文件是整个工程的合同第一页：所有模型、训练、采样脚本都从这里取超参数，
所以改一处就能影响全部实验。

命名与 GPT-2 论文 / nanoGPT 保持一致：
    n_layer    层数       —— 堆叠多少个 Transformer Block
    n_head     头数       —— 多头注意力切成几份
    n_embd     嵌入维度   —— 残差流的宽度（hidden size）
    block_size 上下文长度 —— 一次最多能看多少个 token
    vocab_size 词表大小   —— 决定 Embedding 的行数
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass
class GPTConfig:
    # ---- 词表与上下文 ----
    vocab_size: int = 65          # 字符级 TinyShakespeare 的词表大小；中文语料会重新统计
    block_size: int = 128         # 上下文长度（序列最长 token 数）

    # ---- 模型规模 ----
    n_layer: int = 4              # Transformer Block 堆叠层数
    n_head: int = 4               # 注意力头数，要求 n_embd % n_head == 0
    n_embd: int = 128             # 嵌入维度 / 残差流宽度

    # ---- 正则化 ----
    dropout: float = 0.0          # 教学实验默认关闭，便于复现
    bias: bool = True             # Linear 与 LayerNorm 是否带偏置

    def __post_init__(self) -> None:
        """最基本的合法性检查，把错误提前到构造时暴露。"""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd({self.n_embd}) 必须能被 n_head({self.n_head}) 整除，"
                "否则无法把通道平均分给每个头"
            )
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是正数，当前为 {getattr(self, name)}")

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度 d_k = n_embd / n_head。"""
        return self.n_embd // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        """转成普通 dict，方便存进 checkpoint。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        """从 dict 还原配置；多余字段（例如旧版本留下的）会被忽略。"""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, **overrides: Any) -> "GPTConfig":
        """根据 tokenizer 的词表大小自动填好 vocab_size。"""
        return cls(vocab_size=tokenizer.vocab_size, **overrides)


def get_config(preset: str = "tiny", **overrides: Any) -> Dict[str, Any]:
    """取一个预设规模，并按需覆盖若干字段。返回可直接传给 GPT(**cfg) 的字典。

    预设
    ----
    micro : 玩具规模，用于测试与冒烟验证，CPU 上几秒就能看到 loss 下降
    tiny  : 默认教学规模
    mini  : 想在 CPU 上认真训练一段时间的规模
    small : 接近早期 GPT-1 量级，CPU 上很慢，仅作对照
    """
    presets: Dict[str, Dict[str, Any]] = {
        "micro": dict(n_layer=2, n_head=2, n_embd=64, block_size=64, dropout=0.0),
        "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0),
        "mini": dict(n_layer=6, n_head=6, n_embd=192, block_size=192, dropout=0.1),
        "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.1),
    }
    if preset not in presets:
        raise KeyError(f"未知预设 {preset!r}，可选：{sorted(presets)}")

    cfg: Dict[str, Any] = dict(presets[preset])
    cfg.update(overrides)

    known = {f.name for f in fields(GPTConfig)}
    unknown = set(cfg) - known
    if unknown:
        raise KeyError(f"配置里出现未知字段：{sorted(unknown)}")
    return cfg


def make_config(preset: str = "tiny", vocab_size: int | None = None,
                **overrides: Any) -> GPTConfig:
    """get_config 的 dataclass 版本，顺带把 vocab_size 填进去。"""
    cfg = get_config(preset, **overrides)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return GPTConfig(**cfg)


def count_parameters(module: Any, trainable_only: bool = True) -> int:
    """统计参数量（utils.num_params 的别名，保留以兼容教学代码）。"""
    params = getattr(module, "parameters", None)
    if params is None:
        return 0
    it = params() if callable(params) else params
    return sum(p.numel() for p in it if (p.requires_grad or not trainable_only))
