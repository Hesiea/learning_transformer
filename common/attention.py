"""注意力机制的标准实现（03、04 讲会反复引用本文件并与手写版对照）。

课程里先手推一遍公式，再回到这里看工程实现长什么样：
    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + mask) V

本文件三个类，复杂度依次上升：
    Head                 单个注意力头（03 讲）
    MultiHeadAttention   多头注意力，n_head 个独立的小头（04 讲）
    CausalSelfAttention  GPT 实际使用的实现，一次大矩阵算出所有头，并支持 KV Cache
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 03 讲：单个注意力头
# ======================================================================
class Head(nn.Module):
    """一个自注意力头：没有多头、没有掩码，纯粹的按相关性加权平均。

    四个字母的含义：
        B  batch      批大小
        T  time       序列长度
        C  channels   n_embd，输入特征维度
        hs head_size  每个头的维度 = C / n_head
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        # tril 是下三角矩阵。注册成 buffer 表示它不是参数，但要跟着模型一起搬设备。
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, T, C = x.shape
        q = self.query(x)                                  # (B, T, hs)
        k = self.key(x)                                    # (B, T, hs)
        v = self.value(x)                                  # (B, T, hs)

        # 相关性打分：每个位置与每个位置的点积 -> (B, T, T)
        # 除以 sqrt(head_size) 让方差保持为 1，否则 softmax 会过度尖锐、梯度消失
        att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)

        # 因果掩码：位置 i 只能看 j <= i，未来位置置为 -inf（softmax 后变成 0）
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v                                      # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        if return_weights:
            return out, att
        return out


# ======================================================================
# 04 讲：多头注意力（好读版）
# ======================================================================
class MultiHeadAttention(nn.Module):
    """把 C 个通道平均分给 n_head 个头，各自独立做注意力，最后拼回来再投影。

    为什么要多头？
        一个 softmax 权重向量只能表达一种关注模式。多头相当于让模型同时用
        几套不同的检索条件看同一段历史：有的头盯语法、有的头盯指代、有的头盯位置。
        这和 CNN 用多个卷积核是同一个思想：给模型多种互补的视角。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd({n_embd}) 必须能被 n_head({n_head}) 整除")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, self.head_size, block_size, dropout, bias) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)     # 输出投影，把多头结果混合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        if return_weights:
            outs, weights = [], []
            for h in self.heads:
                o, w = h(x, return_weights=True)
                outs.append(o)
                weights.append(w)
            out = torch.cat(outs, dim=-1)                    # (B, T, C)
            return self.dropout(self.proj(out)), torch.stack(weights, dim=1)  # (B, nh, T, T)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        return self.dropout(self.proj(out))


# ======================================================================
# GPT 实际使用的因果自注意力
# ======================================================================
class CausalSelfAttention(nn.Module):
    """一次大矩阵乘法算出所有头（比循环快得多），并支持 KV Cache。

    与 MultiHeadAttention 数学上等价，区别只是实现效率：
        MultiHeadAttention : n_head 个独立的 Linear，循环调用
        CausalSelfAttention: 一个 (C -> 3C) 的 Linear，然后 reshape 成多头
    前者好读，后者是工业写法。04 讲会验证两者的数值一致性。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd({n_embd}) 必须能被 n_head({n_head}) 整除")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.n_embd = n_embd
        self.block_size = block_size

        # 一个线性层同时产出 q、k、v，省两次矩阵乘法开销
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)   # 输出投影
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # 预生成一个足够大的下三角矩阵。推理时缓存可能达到 2×block_size，留两倍余量。
        self.register_buffer("tril", torch.tril(torch.ones(2 * block_size, 2 * block_size)))

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """前向传播。

        参数
        ----
        x : (B, T, C)
        return_weights : 是否额外返回注意力权重 (B, n_head, T, T)，用于可视化
        cache : KV Cache，形如 (k_prev, v_prev)，各为 (B, n_head, T_past, hs)。
                推理时把历史 K/V 缓存下来，每步只算新 token 的 q，
                能把自回归生成从每步重算全序列降到每步只算一个 token。09 讲详述。

        返回 (out, attn_weights, new_cache)，其中 weights 在不请求时为 None。
        """
        B, T, C = x.shape

        qkv = self.c_attn(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        # 拆出头这一维后，注意力就是在最后两维上做矩阵乘法
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # ---- KV Cache：把历史拼到当前前面 ----
        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat([k_prev, k], dim=2)                 # 沿 T 维拼接
            v = torch.cat([v_prev, v], dim=2)
        new_cache = (k, v)

        T_total = k.size(2)                                   # 当前 token 能看到的总长度
        is_full_forward = (cache is None) and (T == T_total)

        # 掩码的对齐规则：第 i 个 query 对应序列中的第 (T_total - T + i) 个位置，
        # 它只能看到 0..(T_total - T + i) 这些 key。
        # 注意：KV Cache 场景下不能直接用 is_causal=True —— 当 query 只有 1 个、
        # key 有 T_total 个时，SDPA 会按左上角对齐处理，把全部历史都当成未来而屏蔽掉。
        if is_full_forward:
            causal_mask = None                               # 交给 SDPA 内部生成，最快
        else:
            q_pos = torch.arange(T_total - T, T_total, device=x.device).unsqueeze(1)  # (T,1)
            k_pos = torch.arange(T_total, device=x.device).unsqueeze(0)               # (1,T_total)
            causal_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)                  # (1,1,T,Tt)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_full_forward,
        )                                                     # (B, nh, T, hs)

        # 拼回 (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        if return_weights:
            # 需要权重时用显式写法重算一遍（便于可视化；推理路径不受影响）
            with torch.no_grad():
                att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
                causal = torch.ones(T, T_total, dtype=torch.bool, device=x.device)
                causal = torch.tril(causal, diagonal=T_total - T)
                att = att.masked_fill(~causal, float("-inf"))
                att = F.softmax(att, dim=-1)
            return out, att, new_cache
        return out, None, new_cache


# ======================================================================
# 课程用的手写参考实现
# ======================================================================
def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """完全按公式手写，用于和 nn 实现对照。

    q, k, v : (B, T, hs)
    返回     : out (B, T, hs), 权重 (B, T, T)
    """
    B, T, hs = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(hs)           # (B, T, T)
    if causal:
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


def build_causal_mask(T: int, device: torch.device | None = None) -> torch.Tensor:
    """返回 (T, T) 的布尔掩码，True 表示允许被看到。"""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
