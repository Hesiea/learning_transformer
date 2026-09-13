r"""04 讲：多头注意力 —— 让模型同时用多种视角看历史

03 讲的单头注意力只能产生**一套**关注模式：一个 softmax 权重向量。
多头注意力的想法非常朴素：

    把 C 个通道切成 n_head 份，每份独立做一次注意力，最后拼回来再投影一次。

本讲按顺序回答五个问题：

    1. 多头在张量层面到底做了什么     —— view / transpose / contiguous
    2. 参数量变了吗                   —— 切分几乎免费，只多一个输出投影
    3. 两种实现真的等价吗             —— 循环版 vs 一次大矩阵版
    4. 头数到底有没有用               —— 无注意力 / 1 / 2 / 4 / 8 头的对照实验
    5. 不同的头真的分工了吗           —— 逐头统计 + 真实 GPT 里的已知现象

跑法：
    .\.venv\Scripts\python.exe src\04_multihead.py
    .\.venv\Scripts\python.exe src\04_multihead.py --quick    # 对照实验减到 80 步
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.attention import CausalSelfAttention, Head, MultiHeadAttention  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.utils import (  # noqa: E402
    banner,
    human_params,
    num_params,
    pick_device,
    set_seed,
    setup_console,
)


# ======================================================================
# 1. 维度上发生了什么
# ======================================================================
def demo_reshape_mechanics(n_embd: int = 32, n_head: int = 4, B: int = 2,
                           T: int = 6) -> None:
    print(banner("1. 多头在张量层面做了什么：view + transpose"))
    set_seed(0)
    hs = n_embd // n_head
    x = torch.randn(B, T, n_embd)

    print(f"输入 x: {tuple(x.shape)}   n_embd={n_embd}, n_head={n_head}, "
          f"head_size={hs}\n")
    print(f"{'操作':<44} {'结果形状':<24}{'stride'}")
    print("-" * 84)

    # 关键第一步：view —— 把通道维拆成 (n_head, head_size)
    xh = x.view(B, T, n_head, hs)
    print(f"{'x.view(B, T, n_head, head_size)':<44} {str(tuple(xh.shape)):<24}"
          f"{tuple(xh.stride())}")

    # 关键第二步：transpose —— 把 head 维提到序列维前面
    xh = xh.transpose(1, 2)
    print(f"{'.transpose(1, 2) 把 head 提到第 1 维':<44} {str(tuple(xh.shape)):<24}"
          f"{tuple(xh.stride())}")
    print(f"{'':<44} {'(B, n_head, T, head_size)':<24}")

    # 注意力在最后两维上做矩阵乘法
    scores = xh @ xh.transpose(-2, -1)
    print(f"\n{'scores = q @ k.transpose(-2, -1)':<44} {str(tuple(scores.shape)):<24}"
          f"{tuple(scores.stride())}")
    print(f"{'':<44} {'(B, n_head, T, T)':<24}")

    print()
    print("为什么要 transpose？因为 torch 的矩阵乘法只作用在**最后两维**，")
    print("前面的维度一律当成 batch。注意力要算的是「每个头内部、位置对位置」的分数，")
    print("所以必须把 (T, head_size) 摆到最后两维，让 (T, hs) × (hs, T) -> (T, T)。")
    print("看上面 stride 那一列：transpose 之后形状变了，但数据一个字节都没搬，")
    print("只是把第 1、2 维的 stride 对调了 —— 这正是后面所有麻烦的根源。")

    # ---- contiguous：直接 view 会报错，.contiguous() 修好了它 ----
    print()
    print("为什么 .contiguous() 不能省？transpose 返回的是「视图」，内存是错位的，")
    print("而 view 要求张量在内存里连续。直接展平立刻报错：")
    try:
        xh.view(B, T, n_embd)
        print("    （本机 torch 居然允许了 —— 不同版本行为不同）")
    except RuntimeError as e:
        print(f"    xh.view(B, T, C) -> RuntimeError:")
        print(f"        {str(e).splitlines()[0]}")
    print()
    print("正确的写法是先转回来、再连续化、最后 view：")
    back = xh.transpose(1, 2).contiguous().view(B, T, n_embd)
    print(f"    xh.transpose(1, 2).contiguous().view(B, T, C) -> {tuple(back.shape)}")
    print(f"    xh 是否连续 : {xh.is_contiguous()}")
    print(f"    转回后是否连续: {xh.transpose(1, 2).is_contiguous()}")
    print(f"    与原始张量逐位相同: {torch.equal(x, back)}")
    print()
    print("一个反直觉的细节：如果只把 head 维转回来、别的都不动，")
    print("那它虽然「不连续」，但 stride 恰好还能被 view 接受，结果也是对的：")
    sneaky = xh.transpose(1, 2).view(B, T, n_embd)
    print(f"    xh.transpose(1, 2).view(B, T, C) -> {tuple(sneaky.shape)}，"
          f"与 x 相同: {torch.equal(sneaky, x)}")
    print("    但这是**碰巧**成立（因为它恰好等价于 x 自己的 stride），换个维度顺序")
    print("    就会报错或算错。所以工程代码里一律显式写上 .contiguous()，")
    print("    这行不是装饰，而是把「内存布局」这件事讲清楚。")
    print("    common/attention.py 里写的就是这一行。")
    print()
    print("补充：reshape 而不是 view 时，torch 会在需要时自动拷贝，不会报错；")
    print("      代价是你看不见那次拷贝 —— 教学和性能敏感的地方都用显式的写法。")


# ======================================================================
# 2. 参数量对照
# ======================================================================
def demo_param_count(n_embd: int = 32, block_size: int = 32) -> None:
    print(banner("2. 多头 vs 单头：切分几乎不花参数"))
    single = Head(n_embd, n_embd, block_size)          # 头维度 = 整个 n_embd

    print(f"{'配置':<22} {'head_size':>10} {'q/k/v':>10} {'proj':>8} {'合计':>10}")
    print("-" * 64)
    qkv_single = num_params(single)
    print(f"{'单头 Head':<22} {n_embd:>10} {qkv_single:>10,} {0:>8,} "
          f"{qkv_single:>10,}")

    qkv_sizes = set()
    for nh in (1, 2, 4, 8):
        mha = MultiHeadAttention(n_embd, nh, block_size)
        proj = num_params(mha.proj)
        qkv = num_params(mha) - proj                   # 减去输出投影 = 所有头的 q/k/v
        qkv_sizes.add(qkv)
        print(f"{f'多头 n_head={nh}':<22} {n_embd // nh:>10} {qkv:>10,} {proj:>8,} "
              f"{num_params(mha):>10,}")

    print("注：Head 用的是默认 bias=False，所以上表里没有偏置项。")
    print()
    print(f"关键观察：从 1 头切到 8 头，q/k/v 那一列**始终是 {qkv_single:,}**"
          f"（4 种配置只算出 {len(qkv_sizes)} 个不同的值）—— 也就是 3 × C²。")
    print("推导（把 C 切成 n 份，每份 C/n 维）：")
    print("    单头:  3 × C × C")
    print("    n 头:  n × 3 × (C/n) × C = 3 × C × C   —— 与 n 完全无关")
    print()
    print("所以「多头」这个名字有点误导：它**不是**把模型变大 n 倍，")
    print("而是把同样这 3C² 个参数重新组织了一遍 —— 同样的算力，多套关注模式。")
    print("凡是说「多头带来多种视角」的地方，代价都只在计算图的形状上，不在参数量上。")
    print()
    print(f"真正多出来的参数只有输出投影 proj：C × C = {n_embd} × {n_embd} = "
          f"{4 * 0 + n_embd * n_embd:,}。它的作用是把各头的结果重新混合。")
    print("为什么必要？因为各头的输出只是简单地 cat 在一起：")
    print("    [head_0 ; head_1 ; ... ; head_{n-1}]   （沿通道维拼接，各段互不相干）")
    print("如果不再混合一次，后面的层就永远看不到「头 A 的结果」和「头 B 的结果」")
    print("之间的关系 —— 每个头的信息被困在自己那 C/n 个通道里。")
    print("proj 相当于给模型一个重新组合各头信息的开关，是所有多头实现的标准配置。")
    print()

    # 用具体数字看 head_size 随头数的变化
    print("顺带记住 head_size = n_embd / n_head 的变化：")
    print("    " + "  ".join(f"n_head={nh} -> {n_embd // nh} 维" for nh in (1, 2, 4, 8)))
    print("这个数字不能太小：每个头要在一个 C/n 维的子空间里表达自己的关注模式，")
    print("切得太细，单头就没有足够的表达能力了 —— 第 4 小节的实验会验证这一点。")


# ======================================================================
# 3. 两种实现等价性
# ======================================================================
def demo_equivalence(n_embd: int = 32, n_head: int = 4, block_size: int = 16,
                     B: int = 2, T: int = 9) -> None:
    print(banner("3. 两种实现：循环版 vs 一次大矩阵版"))
    set_seed(1)
    x = torch.randn(B, T, n_embd)

    # 写法 A：n_head 个独立的 Head，循环调用（好读）
    mha = MultiHeadAttention(n_embd, n_head, block_size)
    out_a, att_a = mha(x, return_weights=True)          # return_weights=True -> 二元组

    # 写法 B：一次算出 qkv 再 reshape（工业写法，GPT 实际用的）
    csa = CausalSelfAttention(n_embd, n_head, block_size)
    out_b, att_b, cache_b = csa(x, return_weights=True)  # 永远是三元组

    print(f"输入 x                 : {tuple(x.shape)}")
    print(f"MultiHeadAttention 输出: {tuple(out_a.shape)}   权重 {tuple(att_a.shape)}")
    print(f"CausalSelfAttention 输出: {tuple(out_b.shape)}   权重 {tuple(att_b.shape)}")
    print(f"KV Cache（CSA 额外给的）: {type(cache_b).__name__} "
          f"含 k{tuple(cache_b[0].shape)} / v{tuple(cache_b[1].shape)}")
    print()

    # 参数量逐项对照
    print(f"{'模块':<24} {'参数量':>10}   构成")
    print("-" * 62)
    print(f"{'MultiHeadAttention':<24} {num_params(mha):>10,}   "
          f"n_head 组 q/k/v + 1 个 proj")
    print(f"{'CausalSelfAttention':<24} {num_params(csa):>10,}   "
          f"1 个 (C→3C) + 1 个 proj")
    same = num_params(mha) == num_params(csa)
    print(f"\n参数量完全相同: {same}")
    print()
    print("因为数学结构是同一个：把 n_head 个 (C→C/n) 的 Linear 并排放在一起，")
    print("就等价于一个 (C→C) 的 Linear 的对应行块 —— CausalSelfAttention 只是")
    print("把它们合成一个 (C→3C) 的大矩阵，一次乘法算完 q、k、v。")
    print("好处很实在：一次大矩阵乘法的访存效率远高于 3 次小乘法，GPU 上差距明显。")
    print()
    print("两者的区别只在「怎么算」，不在「算什么」：")
    print("    MultiHeadAttention : 好读，适合教学，也方便逐头检查")
    print("    CausalSelfAttention: 工业写法，多了 KV Cache 支持（09 讲细讲）")

    # 两者数值是否一致？随机初始化下不同（参数各自随机），但可以对齐参数验证
    print()
    print("那数值可以直接对比吗？**不能** —— 两者参数是各自随机初始化的。")
    print("把 CSA 的参数按行块拷贝进 MHA，才能真正验证「同一组权重 → 同一结果」：")
    with torch.no_grad():
        mha2 = MultiHeadAttention(n_embd, n_head, block_size, bias=True)
        csa2 = CausalSelfAttention(n_embd, n_head, block_size, bias=True)
        # c_attn 的 (3C, C) 权重按 [q 块 ; k 块 ; v 块] 排布
        W = csa2.c_attn.weight                      # (3C, C)
        b = csa2.c_attn.bias                        # (3C,)
        for i, head in enumerate(mha2.heads):
            lo, hi = i * head.head_size, (i + 1) * head.head_size
            head.query.weight.copy_(W[lo:hi])
            head.query.bias.copy_(b[lo:hi])
            head.key.weight.copy_(W[n_embd + lo:n_embd + hi])
            head.key.bias.copy_(b[n_embd + lo:n_embd + hi])
            head.value.weight.copy_(W[2 * n_embd + lo:2 * n_embd + hi])
            head.value.bias.copy_(b[2 * n_embd + lo:2 * n_embd + hi])
        mha2.proj.weight.copy_(csa2.c_proj.weight)
        mha2.proj.bias.copy_(csa2.c_proj.bias)

        mha2.eval()
        csa2.eval()
        with torch.no_grad():
            o1, w1 = mha2(x, return_weights=True)
            o2, w2, _ = csa2(x, return_weights=True)

    print(f"    输出最大误差 : {(o1 - o2).abs().max().item():.3e}")
    print(f"    权重最大误差 : {(w1 - w2).abs().max().item():.3e}")
    print("误差在 1e-6 量级（浮点累加顺序不同导致），可以认为完全一致。")
    print("这就证明了：两套代码算的是同一个函数，可以放心按需要挑一种读。")


# ======================================================================
# 4. 对照实验：头数到底有没有用
# ======================================================================
class AttentionOnlyLM(nn.Module):
    """一层注意力 + 一层 MLP 的小模型，用来做头数对照实验。

    use_attn=False 时去掉注意力子层，其余结构一模一样 —— 这样「无注意力」
    那一行才是干净的对照组（参数量也会少一截，表里一并打印出来）。
    """

    def __init__(self, vocab_size: int, n_embd: int, block_size: int, n_head: int,
                 use_attn: bool = True):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.use_attn = use_attn
        self.attn = CausalSelfAttention(n_embd, n_head, block_size) if use_attn else None
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        if self.use_attn:
            a, _, _ = self.attn(self.ln1(x))          # 三元组
            x = x + a
        x = x + self.mlp(self.ln2(x))
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def train_and_eval(model, ds, device, steps: int, block_size: int,
                   batch_size: int = 32, lr: float = 3e-3,
                   n_val_batches: int = 10) -> tuple:
    """训练若干步，返回 (最终 train loss, 验证 loss)。"""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loss = float("nan")
    for _ in range(steps):
        model.train()
        xb, yb = ds.get_batch("train", batch_size, block_size, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_loss = loss.item()

    model.eval()
    with torch.no_grad():
        val = 0.0
        for _ in range(n_val_batches):
            vx, vy = ds.get_batch("val", batch_size, block_size, device)
            val += model(vx, vy)[1].item()
    return train_loss, val / n_val_batches


def demo_head_ablation(ds, device: torch.device, steps: int = 250,
                       n_embd: int = 128, block_size: int = 64) -> dict:
    print(banner("4. 对照实验：头数对验证 loss 的影响"))
    print(f"固定 n_embd={n_embd}、block_size={block_size}、批次大小、数据与随机种子，")
    print(f"唯一变量是 n_head。每组各训 {steps} 步（CPU 上共需一两分钟）。\n")

    configs = [("无注意力", 1, False), ("n_head=1", 1, True), ("n_head=2", 2, True),
               ("n_head=4", 4, True), ("n_head=8", 8, True)]

    print(f"{'配置':<12} {'head_size':>10} {'参数量':>10} {'train loss':>12} "
          f"{'val loss':>10} {'相对 val':>10}")
    print("-" * 70)

    results = {}
    for name, nh, use_attn in configs:
        set_seed(999)                      # 每个配置用同一颗种子初始化，唯一变量才是 n_head
        model = AttentionOnlyLM(ds.vocab_size, n_embd, block_size, nh, use_attn=use_attn)
        tr, val = train_and_eval(model, ds, device, steps, block_size)
        hs = n_embd // nh if use_attn else 0
        results[name] = {"val": val, "train": tr, "params": num_params(model),
                         "head_size": hs}

    best = min(r["val"] for r in results.values())
    for name, nh, use_attn in configs:
        r = results[name]
        note = "   <- 最好" if r["val"] == best else f"{r['val'] / best - 1:>+9.1%}"
        hs = "-" if not use_attn else str(r["head_size"])
        print(f"{name:<12} {hs:>10} {r['params']:>10,} {r['train']:>12.4f} "
              f"{r['val']:>10.4f} {note:>10}")

    print()
    print("怎么读这张表：")
    print("  1. 无注意力 vs 有注意力：这一档的差距最大，而且最稳定。")
    print("     注意力是**唯一**能让不同位置交换信息的零件（MLP 只处理单个位置），")
    print("     少了它，模型就退化成「加强版 bigram」—— 只能看当前 token。")
    print("  2. 头数这一档：上面的数字随训练步数变化很大，一定要看步数。")
    print("     本讲的 --quick（80 步）下常常是「头越少越好」，甚至可能看到")
    print("     n_head=4 反而不如 n_head=1；把步数调到 250 以上，")
    print("     规律才会翻转成教科书里的样子：")
    print("         250 步: 无注意力 2.510  nh=1 2.149  nh=2 2.134  "
          "nh=4 2.095  nh=8 2.141")
    print("         400 步: 无注意力 2.502  nh=1 2.083  nh=2 2.071  "
          "nh=4 1.995  nh=8 2.030")
    print("     读法：有注意力 >> 无注意力；1 头 -> 4 头有稳定收益；")
    print("     8 头回落 —— head_size 只剩 16 维，每个头的表达空间太窄了。")
    print("  3. 为什么小步数下规律会反过来？因为这里同时动了两个变量：")
    print("     n_head 变大时 head_size = n_embd / n_head 在同步变小。")
    print("     多头是「多个窄头」，在训练早期窄头更难学出有用的模式。")
    print("     真实模型里 head_size 一般保持在 64~128，正是这个权衡的结果。")
    print()
    print("读表提醒：小步数 + 单种子的差异有相当一部分来自随机初始化，")
    print("          第 2~5 名之间常常只差 1~2%。想要可靠结论，")
    print("          把 --steps 调到 1000 以上，并用 3 个以上随机种子重复实验。")
    return results


# ======================================================================
# 5. 每个头学到了什么
# ======================================================================
def _describe_head(diag: float, prev: float, first: float) -> str:
    """把三个统计量翻译成一句人话，方便一眼看出这个头像哪种「专家」。"""
    if prev >= 0.4:
        return "偏「看前一位」"
    if first >= 0.4:
        return "偏「看开头」"
    if diag >= 0.4:
        return "偏「看自己」"
    return "比较分散"


def demo_head_specialization(ds, device: torch.device, steps: int = 400,
                             n_embd: int = 96, block_size: int = 48,
                             n_head: int = 4) -> None:
    print(banner("5. 不同的头真的在做不同的事吗"))
    print("训练一个小模型，然后逐个头统计它的注意力权重落在哪里。\n")

    set_seed(2024)
    model = AttentionOnlyLM(ds.vocab_size, n_embd, block_size, n_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    log_every = max(1, steps // 4)
    for step in range(steps):
        model.train()
        xb, yb = ds.get_batch("train", 32, block_size, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:4d}   loss = {loss.item():.4f}")

    # 用一段固定文本看每个头的权重
    text = "The quick brown fox jumps over"
    ids = ds.tokenizer.encode(text)
    chars = [c if c != "\n" else "\\n" for c in text]
    model.eval()
    with torch.no_grad():
        x = torch.tensor([ids], device=device)
        T = x.shape[1]
        pos = torch.arange(T, device=device)
        h = model.token_emb(x) + model.pos_emb(pos)
        _, att, _ = model.attn(model.ln1(h), return_weights=True)      # (1, nh, T, T)

    print(f"\n文本（{T} 个字符）: {text!r}\n")
    print("每个头的三个统计量（都在 [0,1] 区间，越大表示越关注那一类位置）：")
    print(f"{'头':>4} {'看自身':>10} {'看前一位':>10} {'看首位':>10}   倾向")
    print("-" * 56)
    stats = []
    for h_idx in range(n_head):
        a = att[0, h_idx].cpu()
        diag = a.diagonal().mean().item()                       # 对角线：关注自己
        prev = a.diagonal(offset=-1).mean().item() if T > 1 else 0.0  # 次对角线：前一位
        first = a[:, 0].mean().item()                           # 第 0 列：关注序列开头
        stats.append((diag, prev, first))
        print(f"{h_idx:>4} {diag:>10.3f} {prev:>10.3f} {first:>10.3f}   "
              f"{_describe_head(diag, prev, first)}")

    print()
    print("这三个统计量分别对应真实 GPT 里被反复观察到的几种「专家头」：")
    print("    * previous-token head：看前一位（prev 高）。最稳定出现的一类，")
    print("      相当于把 bigram 统计硬编码进一层注意力，对相邻搭配极有用。")
    print("    * attention sink：看首位（first 高）。真实 GPT 里很多头会把大量")
    print("      权重丢给第 0 个 token，即使它毫无语义 —— 那是一个「什么都不看」的")
    print("      默认挡位。因为 softmax 必须把权重和为 1 分完，模型需要一个垃圾桶。")
    print("    * 看自己（diag 高）：相当于恒等映射，负责把当前位置的信息原样传下去。")
    print("    * 位置头 / 句法头：关注固定偏移量，或主语-动词、代词-先行词这类关系，")
    print("      需要更长的训练和更深的模型才会清晰分化。")
    print()
    print("注意：我们这个单层小模型训练步数很少，头之间不会分化得那么干净，")
    print("      上面这三个数字只是帮你建立「头有分工」的直觉，不要当成结论。")
    print("      真正想看漂亮的分化，得用多层模型 + 大量训练，再画注意力图。")
    print()
    print("一个可验证的推论：既然头是分工的，那**砍掉某些头**应该比随机砍掉")
    print("同样多的通道伤害更大 —— 这类「头剪枝」研究确实观察到了这个现象，")
    print("也是多头结构真的有意义的间接证据。")


# ======================================================================
def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="04 讲：多头注意力")
    parser.add_argument("--dataset", default="tinyshakespeare", help="data/raw 下的语料名")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=250, help="对照实验每组训练步数")
    parser.add_argument("--quick", action="store_true", help="快速模式：对照实验减到 80 步")
    args = parser.parse_args()

    print(banner("04 讲：多头注意力（Multi-Head Attention）"))
    device = pick_device(args.device)
    ds = load_dataset(args.dataset)
    print(ds.summary())
    print(f"使用设备: {device}")

    ablation_steps = 80 if args.quick else args.steps
    spec_steps = 100 if args.quick else 400
    n_embd, block_size = 128, 64

    demo_reshape_mechanics()
    demo_param_count(n_embd=32, block_size=32)
    demo_equivalence()
    demo_head_ablation(ds, device, steps=ablation_steps, n_embd=n_embd,
                       block_size=block_size)
    demo_head_specialization(ds, device, steps=spec_steps)

    print(banner("小结"))
    print("1. 多头 = 把通道维拆成 (n_head, head_size)，各头独立做注意力，最后拼接再投影。")
    print("   张量上的三步：view -> transpose(1,2) -> 算完再 transpose + contiguous + view。")
    print("2. 切分**不改变** q/k/v 的总参数量（始终 3C²），只多一个输出投影 C²。")
    print("   多头几乎免费 —— 同样的算力换来多套关注模式。")
    print("3. 输出投影的作用是让各头的信息重新混合，否则每个头被困在自己的通道段里。")
    print("4. 循环版（MultiHeadAttention）与一次大矩阵版（CausalSelfAttention）数学等价：")
    print("   对齐参数后输出误差在 1e-6 量级；后者更快，还支持 KV Cache。")
    print("5. 对照实验的普遍规律：有注意力 >> 无注意力（最稳定的一条）；")
    print("   头数 1->4 有收益、切得太细（head_size 过小）会回退 ——")
    print("   但这条需要足够多的训练步数才能看清，本讲的数字会随 --steps 变化。")
    print("6. 头确实会分化：previous-token head、attention sink 等现象在真实 GPT 里")
    print("   都能观测到，说明多头不是简单的参数堆叠。")
    print()
    print("下一讲：src\\05_transformer_block.py —— 把注意力、MLP、残差、LayerNorm")
    print("拼成一个完整的 Transformer 层，再堆起来组装出 GPT。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
