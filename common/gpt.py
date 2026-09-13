"""05 讲的积木：LayerNorm、MLP、Transformer Block 与完整 GPT。

为什么单独放一个文件？
    GPT 的定义有十几个细节（先 norm 还是后 norm、残差加在哪、要不要 bias、
    初始化怎么缩放），散在课程脚本里很容易看漏。集中在这里，代码本身就是规格说明书。

结构总览（数据在各层之间的形状变化）：

    idx (B, T)                     整数 id
     |  wte + wpe
     v
    x   (B, T, C)                  残差流：整条主干都保持这个形状
     |  +------- x n_layer -------+
     |  |  x = x + attn(norm(x))  |
     |  |  x = x + mlp(norm(x))   |
     |  +-------------------------+
     |  norm_f
     v
    logits (B, T, vocab)           每个位置上对下一个 token 的打分
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 允许以脚本方式直接运行本文件（python common/gpt.py）时也能找到工程根目录
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.attention import CausalSelfAttention, MultiHeadAttention  # noqa: E402


# ======================================================================
# LayerNorm
# ======================================================================
class LayerNorm(nn.Module):
    """带可选 bias 的 LayerNorm。

    做什么：对每个 token 自己的特征向量做标准化（减均值、除标准差），
    再用两个可学习参数 weight/bias 把它缩放平移回去。

    与 BatchNorm 的区别（面试高频）：
        LayerNorm 在 (C,) 上归一化 —— 与同批其他样本无关，
                  所以 batch_size=1、序列长度变化时都照样工作。
        BatchNorm 在 (B, T) 上归一化 —— 依赖 batch 统计量，
                  在变长序列与自回归生成场景下会出问题。
    这也是 Transformer 能用小 batch、能在线推理的原因之一。
    """

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


# ======================================================================
# MLP（前馈网络）
# ======================================================================
class MLP(nn.Module):
    """位置独立的两层前馈网络：C -> 4C -> C。

    三个要点：
        1. 隐层通常是输入的 4 倍宽 —— 注意力负责在位置之间搬运信息，
           MLP 负责在单个位置上做非线性加工，需要足够的通道容量。
        2. 逐位置独立计算（每个位置用同一套权重），不涉及位置间信息流动。
        3. 参数量占整个 GPT 的大头：2 × 4C² = 8C²，而注意力是 4C²。
           也就是说大约 2/3 的参数其实在 MLP 里。
    """

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = True,
                 expansion: int = 4, activation: str = "gelu"):
        super().__init__()
        hidden = expansion * n_embd
        self.fc_in = nn.Linear(n_embd, hidden, bias=bias)
        self.fc_out = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x, approximate="tanh")
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":       # LLaMA 等现代模型用 SwiGLU，基础就是 SiLU
            return F.silu(x)
        raise ValueError(f"未知激活函数 {self.activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self._act(self.fc_in(x))))


# ======================================================================
# Transformer Block
# ======================================================================
class Block(nn.Module):
    """一个 Transformer 层：注意力 + MLP，各自套一层残差。

        x = x + attn(ln_1(x))
        x = x + mlp (ln_2(x))

    这是 GPT-2 的 Pre-LN 写法（先归一化再进子层）。原始 Transformer 用 Post-LN：
        x = ln(x + attn(x))
    Pre-LN 的好处是不需要 warmup 也能稳定训练深层网络，所以现代模型基本都用它。

    残差连接为什么关键？
        反向传播时加法把梯度原样传回上一层（dy/dx = 1），
        这让几十层甚至上百层的网络依然能把梯度送到最底部。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 bias: bool = True, use_fused_attention: bool = True):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        if use_fused_attention:
            self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias=False)
        else:
            # 教学版：n_head 个独立的小头，好读但慢
            self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout, bias=False)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias=bias)
        self.use_fused_attention = use_fused_attention

    def forward(self, x: torch.Tensor, return_weights: bool = False,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.use_fused_attention:
            a, att, new_cache = self.attn(self.ln_1(x), return_weights=return_weights, cache=cache)
        else:
            if return_weights:
                a, att = self.attn(self.ln_1(x), return_weights=True)
            else:
                a, att = self.attn(self.ln_1(x)), None
            new_cache = None
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, att, new_cache


# ======================================================================
# 完整 GPT
# ======================================================================
class GPT(nn.Module):
    """Decoder-only GPT：Embedding -> n_layer 个 Block -> LayerNorm -> 语言模型头。"""

    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int,
                 n_embd: int, dropout: float = 0.0, bias: bool = True,
                 use_fused_attention: bool = True, tie_weights: bool = True,
                 activation: str = "gelu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.tie_weights = tie_weights

        self.wte = nn.Embedding(vocab_size, n_embd)     # token embedding
        self.wpe = nn.Embedding(block_size, n_embd)     # position embedding
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout, bias, use_fused_attention)
            for _ in range(n_layer)
        ])
        self.ln_f = LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)   # 输出层通常不带 bias

        # 权重共享：输入嵌入与输出投影共用同一个矩阵。
        # 两者都在描述 token 与向量空间的对应关系，共享能省一大块参数
        # （GPT-2 就是这么做的），在中小模型上通常还能略微提升效果。
        if tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 的技巧：残差分支上的投影层按 1/sqrt(2*n_layer) 缩小初始化。
        # 每个 Block 有两条残差路径，深度越大累积方差越大。
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "GPT":
        """从 GPTConfig（dataclass）或普通 dict 构造模型。"""
        cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        cfg.update(kwargs)
        allowed = {
            "vocab_size", "block_size", "n_layer", "n_head", "n_embd",
            "dropout", "bias", "use_fused_attention", "tie_weights", "activation",
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise KeyError(f"GPT 不认识这些参数：{sorted(unknown)}")
        return cls(**cfg)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        cache: Optional[list] = None,
        pos_offset: int = 0,
    ):
        """前向传播。

        idx     : (B, T) 整数 token id
        targets : (B, T) 每个位置的目标（就是 idx 右移一位），传入则返回 loss
        return_weights : 是否返回每一层的注意力权重，便于可视化
        cache   : 长度 n_layer 的列表，每层存一个 (k, v)，用于 09 讲的增量推理
        pos_offset : idx[0] 在整条序列里的绝对位置。只有增量推理（配合 cache）才需要传：
            此时 idx 只含最后一个 token，位置嵌入必须取 wpe[T_past] 而不是 wpe[0]，
            否则这一步喂进去的向量与全量前向在同一个位置上算出的向量不一致，
            缓存路径就会悄悄偏离正确结果（误差可达 0.1 量级，且不会报错）。
            注意掩码不需要这个偏移：它由 cache 的长度推出每个 query 的真实位置。

        返回 (logits, loss, weights, new_cache)
            logits    : (B, T, vocab)
            loss      : 标量或 None
            weights   : (n_layer, B, n_head, T, T) 或 None
            new_cache : 新的 KV Cache
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                f"输入长度 {T} 超过 block_size={self.block_size}。"
                "请截断输入，或使用 09 讲滑动窗口的生成方式。"
            )

        # 位置嵌入按「序列内的绝对位置」取，所以增量推理时要加上已缓存的长度
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)   # (T,)
        if int(pos.max().item()) >= self.block_size:
            raise ValueError(
                f"位置 {int(pos.max().item())} 超出 block_size={self.block_size}，"
                "wpe 没有对应的行。增量推理时应保持总长度不超过 block_size，"
                "或像 generate 那样在超出窗口后重建 cache。"
            )
        x = self.drop(self.wte(idx) + self.wpe(pos))           # (B, T, C)

        new_cache: list = []
        all_weights = [] if return_weights else None
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, att, layer_new_cache = block(x, return_weights=return_weights, cache=layer_cache)
            new_cache.append(layer_new_cache)
            if return_weights:
                all_weights.append(att)

        x = self.ln_f(x)                                       # (B, T, C)
        logits = self.lm_head(x)                               # (B, T, vocab)

        loss = None
        if targets is not None:
            # 把所有位置展平成一个大 batch 再算交叉熵：等价于逐位置求平均
            loss = F.cross_entropy(
                logits.view(B * T, self.vocab_size), targets.view(B * T), ignore_index=-1
            )

        weights = torch.stack(all_weights, dim=0) if return_weights else None
        return logits, loss, weights, new_cache

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = False,
        seed: Optional[int] = None,
        stop_string: Optional[str] = None,
        tokenizer: Any = None,
    ) -> torch.Tensor:
        """自回归生成。

        参数
        ----
        temperature : >1 更随机、<1 更保守、=1 保持模型原始分布
        top_k       : 只在概率最高的 k 个 token 里采样（None 或 0 表示不限制）
        top_p       : 核采样，取累计概率达到 p 的最小集合
        use_cache   : 是否启用 KV Cache（09 讲）。开启后每步只前向 1 个 token。
        seed        : 固定随机数，便于复现某次采样
        stop_string : 生成到该子串就提前停止（需要同时传 tokenizer）

        实现要点（09 讲会逐条验证）：
            * cache_len 记录「cache 里已经有多少个位置」，它同时决定下一步的位置嵌入
              偏移 pos_offset —— 增量推理时位置嵌入必须按绝对位置取，不能每步都从 0 开始。
            * 没有 cache 时每步重喂 idx_cond = generated[:, -block_size:]，此时
              cache_len 就是 idx_cond 的长度，pos_offset 恒为 0。
            * 序列一旦超过 block_size，generated 会被截断成滑动窗口；此时缓存里的
              历史位置与窗口内的位置对不上了，所以整份 cache 作废并基于窗口重建一次
              （多一次全量前向，代价很小，换来的是位置编码始终正确）。
            * 为什么重建时只喂 block_size-1 个 token？因为位置必须落在
              [0, block_size-1] 区间里，而每一步还要为「即将生成的那个 token」留一个位置，
              所以可复用的历史最多是 block_size-1 个。这与全量路径的窗口长度一致，
              两条路径在超长生成时才会给出完全相同的 token。
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        if idx.size(1) > self.block_size:
            raise ValueError(
                f"prompt 长度 {idx.size(1)} 超过 block_size={self.block_size}，"
                "请先截断到 block_size 以内再生成。"
            )
        generated = idx
        cache = None
        cache_len = 0
        cache_ready = False        # cache 是否已经包含了 generated 的全部历史

        for _ in range(max_new_tokens):
            if use_cache:
                # ---- 情况一：cache 里还没有历史（第一轮，或刚被废弃）----
                # 必须先用完整的 prompt 前向一次把 cache 建起来。
                # 如果这一步偷懒只喂最后一个 token，模型就会在「只看到 1 个 token」
                # 的情况下续写，输出与全量路径完全不同 —— 而且不会报错，极难发现。
                if not cache_ready:
                    idx_cond = generated
                    pos_offset = 0
                    logits, _, _, cache = self(idx_cond, cache=None, pos_offset=0)
                    cache_len = idx_cond.size(1)
                    cache_ready = True
                    logits = logits[:, -1, :] / max(temperature, 1e-6)
                else:
                    # ---- 情况二：窗口即将滑动，缓存的位置与窗口对不上了 ----
                    # 作废并按窗口内最近的历史重建一次（多一次全量前向，代价很小）。
                    if cache_len + 1 >= self.block_size:
                        seed_ids = generated[:, -(self.block_size - 1):]
                        cache = None
                        _, _, _, cache = self(seed_ids, cache=None, pos_offset=0)
                        cache_len = seed_ids.size(1)
                    idx_cond = generated[:, -1:]                 # 有 cache 只需喂最后一个
                    pos_offset = cache_len                       # 绝对位置 = 已缓存长度
                    logits, _, _, cache = self(idx_cond, cache=cache, pos_offset=pos_offset)
                    cache_len = pos_offset + idx_cond.size(1)
                    logits = logits[:, -1, :] / max(temperature, 1e-6)
            else:
                idx_cond = generated[:, -self.block_size:]       # 重喂整段历史
                logits, _, _, _ = self(idx_cond, cache=None, pos_offset=0)
                logits = logits[:, -1, :] / max(temperature, 1e-6)   # (B, vocab)

            logits = _apply_top_k_top_p(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)    # (B, 1)
            generated = torch.cat([generated, next_id], dim=1)

            if generated.size(1) > self.block_size:
                generated = generated[:, -self.block_size:]      # 滑动窗口
                cache_ready = False                              # 窗口变了，cache 必须重建
            if stop_string and tokenizer is not None:
                if stop_string in tokenizer.decode(generated[0].tolist()):
                    break
        return generated


def _apply_top_k_top_p(logits: torch.Tensor, top_k: Optional[int],
                       top_p: Optional[float]) -> torch.Tensor:
    """把不合格的 token 分数设为 -inf，使它们的采样概率为 0。"""
    if top_k:
        k = min(int(top_k), logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # 累计概率超过 p 之后的部分全部剔除（保留第一个越界的，保证至少有一个候选）
        remove = (cumulative - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


# ======================================================================
# 参数量估算（不用真的建模型就能算出来，方便选规模）
# ======================================================================
def estimate_params(vocab_size: int, block_size: int, n_layer: int, n_embd: int,
                    tie_weights: bool = True) -> Dict[str, int]:
    """按公式拆解参数量：嵌入 / 注意力 / MLP / 输出头。"""
    emb = vocab_size * n_embd
    pos = block_size * n_embd
    attn_per_layer = 4 * n_embd * n_embd                    # c_attn(3C²) + c_proj(C²)，注意力不带 bias
    mlp_per_layer = 8 * n_embd * n_embd + 5 * n_embd        # 2 × C × 4C，外加 4C + C 个 bias
    ln_per_layer = 4 * n_embd                               # 两个 LayerNorm，各 weight+bias，共 4C
    per_layer = attn_per_layer + mlp_per_layer + ln_per_layer
    head = 0 if tie_weights else vocab_size * n_embd
    total = emb + pos + per_layer * n_layer + head + 2 * n_embd
    return {
        "token_embedding": emb,
        "position_embedding": pos,
        "attn_per_layer": attn_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "ln_per_layer": ln_per_layer,
        "per_layer": per_layer,
        "all_layers": per_layer * n_layer,
        "lm_head": head,
        "final_norm": 2 * n_embd,
        "total": total,
    }


if __name__ == "__main__":
    # 自检：估算值与实际建出来的模型参数量应当吻合
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common.utils import human_params, num_params, setup_console

    setup_console()
    kwargs = dict(vocab_size=65, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(**kwargs)
    est = estimate_params(vocab_size=65, block_size=64, n_layer=4, n_embd=128)
    actual = num_params(model)
    print(f"估算: {est['total']:,}   实际: {actual:,}   差值: {actual - est['total']:,}")
    print(f"约 {human_params(actual)}")
    x = torch.randint(0, 65, (2, 16))
    logits, loss, _, _ = model(x, x)
    print("前向自检:", tuple(logits.shape), "loss =", round(loss.item(), 4))
