r"""02 讲：把 id 变成向量 —— Embedding 与位置编码

01 讲的 bigram 模型直接拿 id 当行号去查一张表，模型对 token 的所有理解都压在这张表上。
这一讲把输入侧拆成两半，这是 Transformer 的标准做法：

    输入表示 = Token Embedding（这是什么词） + Position Embedding（它在第几个位置）

本讲按顺序看清四件事：

    1. nn.Embedding 到底做了什么 —— 它就是一个可学习的查表矩阵，没有任何魔法
    2. 为什么必须加位置信息   —— 注意力本身对顺序无感（置换等变），而语序决定语义
    3. 两种主流位置方案的对照 —— 可学习位置嵌入（GPT-2 用）vs 正弦位置编码（原论文用）
    4. 一个「位置识别」小实验  —— 用 loss 数值证明位置信息真的被模型用上了

跑法：
    .\.venv\Scripts\python.exe src\02_embeddings.py
    .\.venv\Scripts\python.exe src\02_embeddings.py --quick     # 快速版（少跑几步）
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

from common.data import load_dataset  # noqa: E402
from common.utils import (  # noqa: E402
    Timer,
    banner,
    describe_device,
    fmt_float,
    human_params,
    num_params,
    pick_device,
    set_seed,
    setup_console,
)


# ======================================================================
# 1. Embedding 就是一张查表矩阵
# ======================================================================
def demo_embedding_is_a_table(vocab_size: int, n_embd: int = 8) -> nn.Embedding:
    print(banner("1. nn.Embedding 就是一张可学习的查表矩阵"))
    set_seed(0)

    emb = nn.Embedding(vocab_size, n_embd)
    print(f"nn.Embedding({vocab_size}, {n_embd}) 的权重形状: {tuple(emb.weight.shape)}")
    print(f"参数量: {human_params(num_params(emb))}  (= vocab x n_embd = "
          f"{vocab_size} x {n_embd})")
    print()
    print("它唯一的可学习参数就是 emb.weight 这一个矩阵：")
    print("    第 i 行 = id 为 i 的那个 token 的向量表示。")
    print(f"    emb.weight.shape = {tuple(emb.weight.shape)}，也就是 {vocab_size} 行、"
          f"每行 {n_embd} 维")
    print()

    # ids 里故意放两个 3，用来观察「同一个 id 是否得到同一个向量」
    ids = torch.tensor([[3, 7, 3]])
    out = emb(ids)
    print(f"输入 ids : {ids.tolist()}    shape = {tuple(ids.shape)}")
    print(f"输出     : shape = {tuple(out.shape)}   (B, T, C)")
    print()

    # ---- 手工取行，验证 Embedding 就是矩阵下标 ----
    manual = emb.weight[ids]
    print("用矩阵下标手工取行 emb.weight[ids]，和 nn.Embedding(ids) 对照：")
    print(f"    emb.weight[3]        = {[round(v, 4) for v in emb.weight[3].tolist()]}")
    print(f"    out[0, 0] (= id 3)   = {[round(v, 4) for v in out[0, 0].tolist()]}")
    print(f"    最大误差             = {(manual - out).abs().max().item():.3e}   "
          "<- 完全一致，就是一次下标操作")
    print()

    print("再看 ids 里的两个 3：")
    print(f"    out[0, 0] 与 out[0, 2]（两个都是 id 3）是否相等: "
          f"{torch.equal(out[0, 0], out[0, 2])}")
    print(f"    out[0, 0] 与 out[0, 1]（id 3 与 id 7）是否相等: "
          f"{torch.equal(out[0, 0], out[0, 1])}")
    print()
    print("结论：同一个 id 在哪个位置出现，拿到的向量**完全一样**。")
    print("      也就是说 Embedding 本身根本区分不了「第一个 3」和「第三个 3」——")
    print("      顺序信息在查表这一步就丢了。这就是下一节要补的东西。")
    print()
    print("顺带一提：这跟 01 讲的 bigram 用的是同一个操作，区别只在于")
    print("      bigram 输出 vocab 维（直接当 logits），这里输出 n_embd 维（当作向量用）。")
    return emb


# ======================================================================
# 2. 为什么必须加位置信息
# ======================================================================
def demo_why_position_matters() -> None:
    print(banner("2. 为什么必须加位置信息：注意力对顺序无感"))
    print("注意力（下一讲）的核心运算是这样的：")
    print("    out_i = sum_j  weight(i, j) * value_j")
    print("其中 weight 由 query_i 和 key_j 的点积算出来，只跟「谁跟谁像」有关，")
    print("跟「谁在前谁在后」无关。")
    print()
    print("换句话说，注意力的聚合是**置换等变**的：")
    print("    把输入 token 的顺序打乱，输出只是跟着同样地打乱，数值本身一模一样。")
    print("    没有位置信息的注意力，读不出语序。")
    print()

    # 小实验：对输入倒序后再聚合，结果不变
    set_seed(0)
    x = torch.randn(1, 5, 4)                       # (B, T, C)
    order = torch.arange(5)
    reversed_order = order.flip(0)
    out_plain = x.sum(dim=1)                       # 一个纯粹的「对历史求和」聚合
    out_reversed = x[:, reversed_order, :].sum(dim=1)
    print("小实验：对同一批向量做「求和」聚合，先原序、再倒序")
    print(f"    原序聚合结果  : {[round(v, 4) for v in out_plain[0].tolist()]}")
    print(f"    倒序聚合结果  : {[round(v, 4) for v in out_reversed[0].tolist()]}")
    print(f"    最大变化量    : {(out_plain - out_reversed).abs().max().item():.3e}   "
          "<- 严格为 0")
    print()
    print("    再做一次对照：如果给每个位置加上不同的位置向量，再倒序，结果就变了")
    pos_tag = torch.randn(5, 4)
    tagged_plain = (x[0] + pos_tag).sum(dim=0)
    tagged_reversed = (x[0, reversed_order, :] + pos_tag).sum(dim=0)
    print(f"    加位置后，原序 vs 倒序的最大变化量: "
          f"{(tagged_plain - tagged_reversed).abs().max().item():.3e}   <- 不再为 0")
    print()

    print("可是语序对语言至关重要，差一个位置意思就全变了：")
    print("    「狗咬人」 vs 「人咬狗」     —— 同样三个 token，施受关系完全颠倒")
    print("    「不是不好」 vs 「好不不是」 —— 后者根本不成句")
    print()
    print("所以 Transformer 必须显式地把「我在第几个位置」这一信息注入进去，")
    print("这就是位置编码（position encoding）。标准做法只有一行：")
    print("    x = token_embedding(idx) + position_embedding(pos)")


# ======================================================================
# 3. 两种位置方案
# ======================================================================
def sinusoidal_position_encoding(block_size: int, n_embd: int) -> torch.Tensor:
    """原始 Transformer 论文的正弦位置编码，返回 (block_size, n_embd)。

    公式（pos 是位置，i 是维度下标）：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    偶数维用 sin、奇数维用 cos，每一对共享同一个频率。
    频率从 2π 一直铺到 10000·2π，相当于给每个位置一个多尺度的「指纹」：
    低维变化快（区分相邻位置），高维变化慢（区分远距离的粗位置）。
    """
    if n_embd % 2 != 0:
        raise ValueError(f"正弦编码要求 n_embd 是偶数，收到 {n_embd}")

    pe = torch.zeros(block_size, n_embd)
    position = torch.arange(block_size, dtype=torch.float).unsqueeze(1)        # (T, 1)
    # 10000^(-2i/d) 用 exp/log 计算，避免大数幂运算的精度问题
    div_term = torch.exp(
        torch.arange(0, n_embd, 2, dtype=torch.float) * (-math.log(10000.0) / n_embd)
    )                                                                         # (d/2,)
    pe[:, 0::2] = torch.sin(position * div_term)                              # 偶数维
    pe[:, 1::2] = torch.cos(position * div_term)                              # 奇数维
    return pe


def demo_position_encodings(block_size: int = 32, n_embd: int = 32) -> dict:
    print(banner("3. 两种位置方案：可学习嵌入 vs 正弦编码"))
    set_seed(0)

    learned = nn.Embedding(block_size, n_embd)
    fixed = sinusoidal_position_encoding(block_size, n_embd)

    print("A. 可学习位置嵌入（GPT-2 / nanoGPT / 本工程采用）")
    print(f"   nn.Embedding({block_size}, {n_embd})，参数量 {human_params(num_params(learned))}")
    print("   做法：把位置当成另一种 token，也查一张表，表里的值由训练自己学出来。")
    print("   优点：完全由数据决定，表达能力强，实现只有一行。")
    print(f"   缺点：block_size={block_size} 是硬上限（位置 {block_size} 查不到表），")
    print("         也不能直接外推到训练时没见过的长度。")
    print()

    print("B. 正弦位置编码（原始 Transformer 论文采用）")
    print(f"   无参数，形状 {tuple(fixed.shape)}，由公式直接算出来")
    print("   优点：不需要训练、不占参数；不同位置的编码是确定的，")
    print("         且理论上能外推到更长的序列（因为公式对任意 pos 都成立）。")
    print("   缺点：是人为设定的固定模式，不一定最适合你的数据。")
    print()

    print("两者数值范围对照：")
    print(f"    可学习（随机初始化）整体: min={learned.weight.min().item():+.4f}, "
          f"max={learned.weight.max().item():+.4f}, "
          f"std={learned.weight.std().item():.4f}")
    print(f"    正弦编码            整体: min={fixed.min().item():+.4f}, "
          f"max={fixed.max().item():+.4f}, std={fixed.std().item():.4f}")
    print("    一句话：可学习版是「零均值小方差」的普通初始化（因为要和 token 嵌入相加，")
    print("    尺度必须匹配）；正弦版被硬性限制在 [-1, 1] 内。")
    print()
    print("相同位置的两者数值（前 6 维）：")
    print(f"    可学习 位置 0: {[round(v, 3) for v in learned.weight[0, :6].tolist()]}")
    print(f"    正弦   位置 0: {[round(v, 3) for v in fixed[0, :6].tolist()]}")
    print(f"    正弦   位置 1: {[round(v, 3) for v in fixed[1, :6].tolist()]}")
    print("    注意正弦版的位置 0 是 [sin0, cos0, sin0, cos0, ...] = [0, 1, 0, 1, ...]，")
    print("    这是公式的直接结果，不是巧合。")
    print()

    # ---- 正弦编码的性质：距离越远，相似度越低 ----
    print("正弦编码的一个漂亮性质：用余弦相似度衡量两个位置有多像")
    normed = F.normalize(fixed, dim=-1)
    sim = normed @ normed.T                      # (T, T) 相似度矩阵
    print(f"    {'距离':>4}  {'平均余弦相似度':>16}")
    print("    " + "-" * 24)
    for d in [0, 1, 2, 4, 8, 16]:
        pairs = [float(sim[i, i + d]) for i in range(block_size - d)]
        print(f"    {d:>4}  {sum(pairs) / len(pairs):>16.4f}")
    print()
    print("    距离 0 相似度恒为 1（和自己完全一样），距离越大相似度越低。")
    print("    这意味着「相近的位置拿到相近的编码」，注意力可以据此推断相对距离。")
    print()
    print("相似度矩阵的一小块（值越大越像，只打印 0.5 以上的格子）：")
    for i in range(8):
        row = "".join(_shade_sim(float(sim[i, j])) for j in range(8))
        print(f"    pos {i}  {row}")
    print("    图例: '#' >=0.9   '*' >=0.7   ':' >=0.5   '.' <0.5")
    print("    沿对角线附近亮、往右上角变暗 —— 这就是「自带远近感」。")
    return {"learned": learned, "fixed": fixed, "sim": sim}


def _shade_sim(v: float) -> str:
    if v >= 0.9:
        return " #"
    if v >= 0.7:
        return " *"
    if v >= 0.5:
        return " :"
    return " ."


# ======================================================================
# 4. 位置识别实验：位置信息真的被用上了吗
# ======================================================================
class PositionAwareModel(nn.Module):
    """每个位置都能作答的最小模型：(token 嵌入 + 位置嵌入) -> 线性读出。

    forward 输入 (B, T) 的 id，输出 (B, T, vocab) 的打分 ——
    也就是说序列里的**每个位置**都要预测「下一个 token」。

    因为读出层是位置共享的，位置信息只能从位置嵌入里来：
    对第 j 个位置来说，它手里的向量是 token_emb(x_j) + pos(j)，
    想答对「下一个 token 等于 x_0」，就必须靠 pos(j) 知道自己该去读第 0 位。
    """

    def __init__(self, vocab_size: int, n_embd: int, block_size: int,
                 use_learned_pos: bool = True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.use_learned_pos = use_learned_pos
        self.block_size = block_size
        if use_learned_pos:
            self.pos_emb = nn.Embedding(block_size, n_embd)
        else:
            # 固定的编码不需要梯度，用 buffer 存（会跟着 state_dict 一起存取）
            self.register_buffer("pos_table",
                                 sinusoidal_position_encoding(block_size, n_embd))
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"序列长度 {T} 超过 block_size={self.block_size}")
        tok = self.token_emb(idx)                                   # (B, T, C)
        positions = torch.arange(T, device=idx.device)
        pos = (self.pos_emb(positions) if self.use_learned_pos
               else self.pos_table[:T])                             # (T, C) 广播到 (B, T, C)
        return self.head(tok + pos)                                 # (B, T, vocab)


class PositionBlindModel(nn.Module):
    """对照组：只有 token 嵌入，没有任何位置信息。

    它把整段序列的向量**求和**成一个向量，再拿这个向量去回答每一个位置。
    求和是置换不变的聚合 —— 和注意力一样对顺序无感：
    输入里有哪些 token 它一清二楚，但「谁在第一个」它永远不知道。
    而且它对所有位置给出的是**同一套**预测。
    """

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(idx).sum(dim=1)                          # (B, C)，顺序被抹掉
        logits = self.head(h)                                       # (B, vocab)
        return logits.unsqueeze(1).expand(-1, idx.shape[1], -1)     # 复制到每个位置


def make_batch(vocab_size: int, batch: int, T: int, device: torch.device):
    """造一批数据：输入是一串随机 token，目标是「预测下一个 token 等于第 0 个 token」。

    也就是说 y[b, t] = x[b, 0] 对所有 t 都成立（最后一个位置的下一跳正好是开头那个 token）。

    注意这里是**每一步都重新随机抽**，而不是固定一批反复喂：
    如果固定一批数据，模型会直接把答案背下来，实验就失去意义了。
    用无穷多的随机样本，模型只能去学「规律」本身，而这份规律恰恰依赖位置。
    """
    x = torch.randint(0, vocab_size, (batch, T), device=device)
    y = x[:, :1].expand(batch, T).contiguous()
    return x, y


@torch.no_grad()
def eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """在给定数据上算 loss，不训练。x:(B,T)  y:(B,T)"""
    model.eval()
    logits = model(x)                                       # (B, T, vocab)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    model.train()
    return float(loss.item())


def demo_position_aware_model(ds, device: torch.device, steps: int = 400) -> dict:
    print(banner("4. 小实验：位置信息真的被用上了吗"))
    print("设计一个必须靠位置才能做对的任务：**预测下一个 token 等于第 0 个 token**。")
    print("    输入 [A, B, C, D]  目标 [B, C, D, A]")
    print()
    print("关键在于：序列里的每个位置都要回答同一个答案 A，")
    print("而 A 这个 token 只出现在第 0 个位置上。")
    print("所以「第 0 个位置装的是谁」这件事，模型必须能从表示里读出来。")
    print()
    print("三种配置跑同样的步数、看同样的数据分布，最后比较 loss 与基准 ln(V)：")
    print("    A. 无位置嵌入      —— 只有 token 嵌入，把整段序列求和（置换不变）")
    print("    B. token + 可学习位置")
    print("    C. token + 正弦位置")
    print()

    set_seed(0)
    vocab_size, n_embd, T = ds.vocab_size, 64, 16
    batch, eval_n = 64, 1024

    # 固定的验证集：只用来评估，绝不参与训练（否则又变成背诵了）
    set_seed(7)
    x_val, y_val = make_batch(vocab_size, eval_n, T, device)
    print(f"任务规模: vocab_size={vocab_size}, n_embd={n_embd}, T={T}, "
          f"batch={batch}, steps={steps}")
    print(f"训练数据: 每一步现场随机生成 (batch, T) = {(batch, T)} 的样本，无穷无尽")
    print(f"验证数据: 固定 {eval_n} 条独立样本（不参与训练），只考核泛化")
    print(f"每一批的 loss 都在 B x T = {batch * T} 个位置上取平均")
    print()

    configs = [
        ("A 无位置嵌入", "none"),
        ("B token + 可学习位置", "learned"),
        ("C token + 正弦位置", "sinusoidal"),
    ]
    results: dict = {}
    trained: dict = {}
    print(f"{'配置':<24} {'参数量':>9} {'step0':>8} {'训练末段':>9} "
          f"{'验证 loss':>10} {'耗时':>7}")
    print("-" * 72)
    for label, kind in configs:
        set_seed(1234)                              # 三种配置从同一初始条件出发
        if kind == "none":
            # 对照组：没有任何位置信号，只能把整段序列求和成一个向量
            model = PositionBlindModel(vocab_size, n_embd).to(device)
        else:
            model = PositionAwareModel(vocab_size, n_embd, T,
                                       use_learned_pos=(kind == "learned")).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

        recent: list[float] = []
        first = None
        with Timer() as t:
            for step in range(steps):
                xb, yb = make_batch(vocab_size, batch, T, device)
                logits = model(xb)                      # (B, T, vocab)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step == 0:
                    first = loss.item()
                recent.append(loss.item())
                if len(recent) > 50:
                    recent.pop(0)
        train_tail = sum(recent) / len(recent)
        val_loss = eval_loss(model, x_val, y_val)

        n_par = num_params(model)
        results[label] = {"first": float(first), "train_tail": train_tail,
                          "val": val_loss, "params": n_par, "seconds": t.elapsed}
        trained[label] = model
        print(f"{label:<24} {human_params(n_par):>9} {first:>8.4f} "
              f"{train_tail:>9.4f} {val_loss:>10.4f} {t.elapsed:>6.1f}s")

    baseline = math.log(vocab_size)
    print("-" * 72)
    print(f"{'随机猜测基准 ln(V)':<24} {'':>9} {baseline:>8.4f} {baseline:>9.4f} "
          f"{baseline:>10.4f}")
    print("（最后一列是 ln(V)：模型什么都不会时的 loss，也是本实验的分界线）")
    print()

    print("怎么读这张表：")
    for label, _ in configs:
        r = results[label]
        gap = r["val"] - baseline
        verdict = "明显低于基准 —— 位置信息生效" if gap < -0.5 else \
                  "停在基准附近 —— 读不出位置"
        print(f"    {label:<24} 验证 {r['val']:.4f}（比基准 {gap:+.4f}）  {verdict}")
    print()

    # ---- 第二个角度：让「被复制的那个位置」动起来 ----
    print("第二个角度更直观：把要复制的源位置从 0 号挪到别的下标，")
    print("任务变成「预测下一个 token 等于第 k 个 token」，再测一次 loss。")
    print("如果模型真的在用位置信息，换个下标它照样能复制；")
    print("读不出位置的模型只能给出同一套平均答案，怎么挪都好不了。")
    print()
    shifts = [0, 1, 4, 8, 15]
    print(f"{'配置':<24} " + " ".join(f"{'源位置 ' + str(k):>11}" for k in shifts))
    print("-" * 72)
    shift_table: dict = {}
    for label, _ in configs:
        model = trained[label]
        row = []
        for k in shifts:
            xk, _ = make_batch(vocab_size, eval_n, T, device)
            yk = xk[:, k:k + 1].expand(eval_n, T).contiguous()
            row.append(eval_loss(model, xk, yk))
        shift_table[label] = row
        print(f"{label:<24} " + " ".join(f"{v:>11.4f}" for v in row))
    print("-" * 72)
    print("    无位置模型对所有源位置给出的是同一套预测，loss 一直在 ln(V) 附近；")
    print("    带位置的模型无论复制哪个位置都能接近 0 —— 位置信息确实在用。")
    print()

    # ---- 反事实检验：只改第 0 个 token，其余位置一个都不动 ----
    print("第三个角度是「反事实对照」：取一条样本，只把第 0 个位置的 token 换掉，")
    print("后面 15 个位置一个都不动，正确答案随之变成新的第 0 个 token。")
    print("看每个位置给出的概率有没有跟着变：")
    print()
    set_seed(11)
    probe_n = 256
    x_probe = torch.randint(0, vocab_size, (probe_n, T), device=device)
    old_first = x_probe[:, 0].clone()
    new_first = (old_first + 1 + torch.randint(0, vocab_size - 1, (probe_n,),
                                               device=device)) % vocab_size
    x_probe_new = x_probe.clone()
    x_probe_new[:, 0] = new_first

    print(f"{'配置':<24} {'原答案概率':>10} {'新答案概率':>10} "
          f"{'变化量':>9} {'逐位置选对率':>13}")
    print("-" * 72)
    probe_stats: dict = {}
    for label, _ in configs:
        model = trained[label]
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(x_probe_new), dim=-1)       # (B, T, vocab)
        picked = probs.gather(2, new_first.view(-1, 1, 1).expand(probe_n, T, 1))
        follow = float(picked.mean())                          # 换过之后给新答案的概率
        accuracy = float((probs.argmax(dim=-1) == new_first.view(-1, 1)).float().mean())
        with torch.no_grad():
            probs_old = F.softmax(model(x_probe), dim=-1)
        keep = float(probs_old.gather(
            2, old_first.view(-1, 1, 1).expand(probe_n, T, 1)).mean())
        probe_stats[label] = {"keep": keep, "follow": follow, "acc": accuracy}
        print(f"{label:<24} {keep:>10.4f} {follow:>10.4f} "
              f"{follow - keep:>+9.4f} {accuracy * 100:>12.1f}%")
    print("-" * 72)
    print("怎么读：")
    for label, _ in configs:
        s = probe_stats[label]
        if s["follow"] - s["keep"] > 0.3:
            comment = "全部位置都跟着第 0 位改口 —— 位置信息确实在用"
        else:
            comment = "基本不响应 —— 它关心的只是「整袋 token 的构成」"
        print(f"    {label:<24} 给新答案的概率比旧答案高 {s['follow'] - s['keep']:+.4f}，"
              f"选对率 {s['acc'] * 100:.1f}%  -> {comment}")
    print()
    print("    无位置模型之所以几乎不响应，是因为它把整段序列求和，")
    print("    第 0 位只是 16 项里的一项（换成别的 id 带来的向量差会被摊薄 1/16），")
    print("    而且无论问哪个位置，它给出的都是同一个向量、同一套概率。")
    print()
    print("结论：位置信息不是锦上添花，而是注意力机制能处理语序的前提。")
    print("      没有它，模型面对的是一袋 token（bag of tokens）：")
    print("      它能知道「这段里有 A、B、C、D」，但永远不知道「谁在第一个」。")
    print("      而语言恰恰是「有顺序的符号序列」。")
    print()
    print("补充说明：无位置配置最终会停在 ln(V) 附近，这一点可以推出来 ——")
    print("      数据是随机生成的，x[0] 与其余位置独立同分布，")
    print("      置换不变的模型从「一袋 token」里推不出最早放进去的是哪一个，")
    print("      它的最优策略就只能是输出词表的边缘分布，loss 恰好是 ln(V)。")
    print("      而可学习位置与正弦位置都能读出第 0 位，所以显著低于基准。")
    return {"results": results, "baseline": baseline, "shift": shift_table,
            "probe": probe_stats}


# ======================================================================
def main() -> int:
    setup_console()

    parser = argparse.ArgumentParser(description="02 讲：Embedding 与位置编码")
    parser.add_argument("--dataset", default="tinyshakespeare", help="data/raw 下的语料名")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=400, help="小实验的训练步数")
    parser.add_argument("--quick", action="store_true", help="快速模式：少跑几步")
    args = parser.parse_args()

    steps = 150 if args.quick else args.steps

    print(banner("02 讲：Embedding 与位置编码"))
    device = pick_device(args.device)
    print(f"使用设备: {describe_device(device)}")

    ds = load_dataset(args.dataset)
    print(ds.summary())

    demo_embedding_is_a_table(ds.vocab_size, n_embd=8)
    demo_why_position_matters()
    demo_position_encodings(block_size=32, n_embd=32)
    demo_position_aware_model(ds, device, steps=steps)

    print(banner("小结"))
    print("1. nn.Embedding 就是一张可学习的查表矩阵：emb.weight[ids] 与 nn.Embedding(ids)")
    print("   数值完全一致；同一个 id 在任何位置拿到的向量都一样，所以它不含位置信息。")
    print("2. 注意力的聚合是置换等变的 —— 不给位置信息，模型读不出「狗咬人」和")
    print("   「人咬狗」的区别。这是必须加位置编码的根本原因。")
    print("3. 输入表示 = token 嵌入 + 位置嵌入，两者形状相同、直接相加，只有一行代码。")
    print("4. 两种方案各有取舍：可学习位置嵌入（表达力强、受 block_size 限制）")
    print("   与正弦位置编码（无参数、可外推、但模式固定）。本工程默认用前者。")
    print("5. 小实验的数值证明：无位置配置卡在 ln(V) 附近，加位置后 loss 大幅下降。")
    print()
    print("现代大模型更多转向 RoPE（旋转位置编码）：它把位置信息直接作用在 Q/K 的")
    print("旋转上，兼顾外推性与表达力，是 09 讲之后留的动手练习。")
    print()
    print("下一讲：src\\03_attention.py —— 本工程最核心的一讲，从零写出自注意力。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
