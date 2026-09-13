r"""03 讲：自注意力 —— 从「加权平均」到 Attention(Q, K, V)

这是全工程最核心的一讲。注意力听起来玄，本质只有一句话：

    每个位置的输出 = 对所有历史位置的 value 做加权平均，权重由「相关性」决定。

所以整讲就两件事：先看清楚「加权平均」长什么样，再搞清楚「权重从哪来」。

路线图（每一步只加一个想法，并且立刻用数值验证）：

    1. 热身：加权平均            —— 注意力最后一步就是它
    2. 相关性从哪来              —— 点积衡量相似度
    3. Q / K / V 的分工          —— 图书馆检索的比喻，三者同源
    4. 为什么必须除以 √d_k       —— 不缩放的 softmax 会「几乎 one-hot」
    5. 因果掩码                  —— GPT 与 BERT 的分水岭
    6. 手写版 vs torch 版        —— 数值误差应当接近 0，并验证因果性
    7. 让注意力自己学            —— 单层注意力小模型 vs 01 讲的 bigram
    8. 正式实现                  —— common/attention.py 里的 Head
    9. 注意力可视化              —— 字符热力图 + 每行和为 1 的自检

跑法：
    .\.venv\Scripts\python.exe src\03_attention.py
    .\.venv\Scripts\python.exe src\03_attention.py --quick     # 训练减到 150 步
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

from common.attention import (  # noqa: E402
    CausalSelfAttention,
    Head,
    build_causal_mask,
    manual_attention,
)
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
# 1. 热身：加权平均
# ======================================================================
def demo_weighted_average() -> None:
    print(banner("1. 热身：注意力最后一步就是加权平均"))
    values = torch.tensor([1.0, 2.0, 10.0])
    weights = torch.tensor([0.5, 0.3, 0.2])

    out = (weights * values).sum()
    v_list = [round(v, 2) for v in values.tolist()]
    w_list = [round(w, 2) for w in weights.tolist()]
    print(f"values  = {v_list}   ← 每个位置携带的内容")
    print(f"weights = {w_list}   （和 = {weights.sum():.2f}）← 每个位置的重要性")
    print(f"\n加权平均 = 0.5×1 + 0.3×2 + 0.2×10 = {out:.3f}   ← 这就是「注意力输出」")
    print()
    print("换个角度看这行公式，它就是注意力的完整前向：")
    print("    out = Σ_j  weight(i, j) · value(j)      （j 取遍第 i 个位置能看到的所有位置）")
    print()
    print("写成矩阵乘法就是 out = weights @ values；手写版和矩阵版必须给出同一个数：")
    manual = sum(w * v for w, v in zip(weights.tolist(), values.tolist()))
    print(f"    手写求和 = {manual:.6f}   矩阵乘法 = {(weights @ values).item():.6f}")
    print()
    print("权重的含义：我从每个位置「取」了多少信息。")
    print("    权重集中 -> 只盯着一个位置看（例如只看前一个词，能拼出词对）")
    print("    权重分散 -> 把整段历史揉在一起（例如判断整体语气）")
    print()
    print("模型要学的**不是**这个加权平均（它是固定的），而是**怎么算出这些权重**。")
    print("这就是 query 和 key 存在的唯一理由。")


# ======================================================================
# 2. 相关性来自点积
# ======================================================================
def demo_dot_product_similarity() -> None:
    print(banner("2. 相关性从哪来：点积就是相似度"))
    set_seed(0)
    a = torch.tensor([1.0, 0.0])       # 基准方向
    b = torch.tensor([0.9, 0.1])       # 与 a 几乎同向
    c = torch.tensor([0.0, 1.0])       # 与 a 垂直
    d = torch.tensor([-1.0, 0.0])      # 与 a 完全反向

    print(f"基准向量 a = {a.tolist()}\n")
    print(f"{'向量':<10} {'取值':<16} {'a·v':>8} {'夹角':>8} {'cos 相似度':>12}")
    print("-" * 60)
    for name, vec in [("同向 b", b), ("垂直 c", c), ("反向 d", d)]:
        dot = float(a @ vec)
        cos = float(F.cosine_similarity(a, vec, dim=0))
        # acos 的定义域是 [-1, 1]，浮点误差可能让 cos 略微越界，先夹一下
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        print(f"{name:<10} {str(vec.tolist()):<16} {dot:>+8.3f} {angle:>7.0f}° {cos:>12.3f}")

    print()
    print("规律：点积越大 = 两个向量越「同向」= 越相关；垂直时接近 0，反向时变负。")
    print("这正是我们要的「打分器」，而且它有一个巨大的好处：")
    print("    一次矩阵乘法就能算完所有位置两两之间的分数 —— 天然适合 GPU 并行。")
    print()
    print("于是把当前 token 变成 query、历史 token 变成 key，")
    print("query·key 就直接给出了「我该从谁那里取信息」的打分。")
    print("注意它是**无界的**（这里能到 +1.0，在 768 维空间里能到几十），")
    print("而无界分数直接进 softmax 会出问题 —— 第 4 小节专门处理这件事。")


# ======================================================================
# 3. Q / K / V 的分工
# ======================================================================
def demo_qkv_roles(n_embd: int = 8) -> None:
    print(banner("3. Q / K / V 的分工：图书馆检索的比喻"))
    print("把自注意力想成一次图书馆检索：")
    print("    query (Q) : 我要找什么     —— 由**当前位置**生成（我的需求）")
    print("    key   (K) : 我这本书讲什么 —— 由每个位置生成（书脊上的标签）")
    print("    value (V) : 我这本书的内容 —— 由每个位置生成（真正被取走的东西）")
    print()
    print("检索流程和注意力一一对应：")
    print("    1. 拿我的 query 去和每一本书的 key 比对   -> Q Kᵀ / √d_k  （打分）")
    print("    2. 用 softmax 把分数变成「注意力配额」     -> softmax          （归一化）")
    print("    3. 按配额把各本书的 value 混合起来          -> weights @ V      （加权平均）")
    print()
    print("最关键的一点：**三个角色都由同一个输入 x 线性变换而来**，只是权重矩阵不同：")
    print("    Q = x @ W_q        K = x @ W_k        V = x @ W_v")
    print("这就是所谓「自」注意力：query 和 key/value 来自同一段序列，自己看自己。")
    print()

    # 用数值证明「同源」：同一个 x，三个投影，三种角色
    set_seed(0)
    x = torch.randn(1, 1, n_embd)
    w_q, w_k, w_v = (torch.randn(n_embd, n_embd) / math.sqrt(n_embd) for _ in range(3))
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    print(f"同一个 x（形状 {tuple(x.shape)}）经过三个不同矩阵：")
    print(f"    q = x @ W_q  -> {tuple(q.shape)}  前 4 维 "
          f"{[round(t, 3) for t in q[0, 0, :4].tolist()]}")
    print(f"    k = x @ W_k  -> {tuple(k.shape)}  前 4 维 "
          f"{[round(t, 3) for t in k[0, 0, :4].tolist()]}")
    print(f"    v = x @ W_v  -> {tuple(v.shape)}  前 4 维 "
          f"{[round(t, 3) for t in v[0, 0, :4].tolist()]}")
    print("三个向量完全不同 —— 同一个位置在扮演三种不同角色。")
    print()
    print("为什么 K 和 V 不合并成一个？如果共用，那么「好匹配」和「内容有用」就被绑死了：")
    print("    模型就没法学出「这个位置很重要，但它的内容我不想要」这类行为。")
    print("分开之后，匹配用一套参数、取内容用另一套，模型自由得多。")


# ======================================================================
# 4. 为什么必须除以 √d_k
# ======================================================================
def demo_scaling(d_k_list=(4, 16, 64, 256, 1024), T: int = 64) -> None:
    print(banner("4. 为什么必须除以 √d_k：不缩放的 softmax 会「几乎 one-hot」"))
    set_seed(0)
    print(f"造一个随机的 query，去和 {T} 个随机 key 做点积，观察 softmax 出来的权重分布。\n")
    print(f"{'d_k':>6} {'点积标准差':>12} {'理论 √d_k':>10} "
          f"{'未缩放 max 权重':>16} {'缩放后 max 权重':>16}")
    print("-" * 66)

    for d_k in d_k_list:
        q = torch.randn(1, 1, d_k)
        k = torch.randn(1, T, d_k)
        # 每个分量都是两个独立 N(0,1) 的乘积之和，所以标准差 ≈ √d_k
        scores = q @ k.transpose(-2, -1)                       # (1, 1, T)
        scaled = scores / math.sqrt(d_k)

        p_raw = F.softmax(scores, dim=-1)
        p_scaled = F.softmax(scaled, dim=-1)
        print(f"{d_k:>6} {scores.std().item():>12.2f} {math.sqrt(d_k):>10.2f} "
              f"{p_raw.max().item():>16.4f} {p_scaled.max().item():>16.4f}")

    print()
    print("读法：")
    print("  第二列随 d_k 增大而增大（≈ √d_k），第三列是它的理论值，两者吻合。")
    print("  第四列（未缩放的最大权重）迅速逼近 1 —— 这时的注意力「几乎 one-hot」：")
    print("      最大值 ≈ 1，其余 ≈ 0，模型实际上只从**一个**位置取信息。")
    print("  第五列（除以 √d_k 之后）始终停留在比较温和的区间。")
    print()
    print("为什么 one-hot 是灾难？看 softmax 的导数：")
    print("    权重饱和到 0 或 1 时，softmax 的雅可比矩阵趋近于 0，")
    print("    梯度传不回 q 和 k，这些参数就「学不动」了。")
    print()
    print("为什么要除以**恰好** √d_k？因为点积的方差是 d_k：")
    print("    q·k = Σ q_i k_i，每项方差为 1，d_k 项相加 -> 方差 d_k")
    print("  除以 √d_k 之后方差回到 1，softmax 的输入尺度与 d_k 无关 —— 换模型深度、")
    print("  换头维度都不需要重新调这个超参数。这是 Transformer 里最容易被忽略、")
    print("  却最不能删的一个细节。")


# ======================================================================
# 5. 因果掩码
# ======================================================================
def demo_causal_mask(T: int = 6) -> None:
    print(banner("5. 因果掩码：不许偷看未来（GPT 与 BERT 的分水岭）"))
    mask = build_causal_mask(T)
    print(f"build_causal_mask({T}) 返回形状 {tuple(mask.shape)} 的布尔矩阵，"
          "True = 允许被看到：\n")
    for i in range(T):
        row = " ".join("1" if bool(mask[i, j]) else "·" for j in range(T))
        print(f"    位置 {i}:  {row}")
    print("              " + " ".join(str(j) for j in range(T)) + "   ← 被看的位置 j")
    print()
    print("结构说明：")
    print("    第 0 行只有 1 个 1  —— 序列开头没有任何历史，只能看自己。")
    print("    第 i 行有 i+1 个 1 —— 位置 i 能看到 0..i 共 i+1 个位置。")
    print("    严格上三角全为 False —— 未来位置永远看不见。")
    print()
    print("在实现里，这个布尔矩阵是这样用的（common/attention.py 里的两行）：")
    print("    att = att.masked_fill(tril[:T, :T] == 0, float('-inf'))")
    print("    att = F.softmax(att, dim=-1)")
    print("被填成 -inf 的位置，softmax 之后恰好是 exp(-inf) = 0 —— 权重真真正正为 0，")
    print("而不是「很小的数」。这一点在验证因果性时很重要。")
    print()
    print("为什么必须掩码？因为训练时我们把整段文本一次性喂进去，同时在每个位置")
    print("预测下一个 token。如果位置 3 能看见位置 5 的内容，它在预测位置 4 时就已经")
    print("「知道答案」了，模型会学会抄近路：训练 loss 一路下降，推理时没有未来可看，")
    print("表现立刻崩塌。掩码强制模型只用过去推断未来 —— 和推理时的处境完全一致。")
    print()
    print("这也是 GPT 与 BERT 的根本差别：")
    print("    GPT  （Decoder-only）下三角掩码，只能看左边 -> 天生适合「生成」")
    print("    BERT （Encoder）      双向可见，能看两边   -> 适合「理解」，")
    print("                          但没法直接用来续写，因为训练目标里没有下一词预测")


# ======================================================================
# 6. 手写 vs torch
# ======================================================================
def demo_manual_vs_torch(n_embd: int = 32, T: int = 12, B: int = 2) -> None:
    print(banner("6. 手写注意力 vs torch 实现：数值必须一致"))
    set_seed(42)

    # ---- 手写版：从 x 到输出，公式全程展开 ----
    x = torch.randn(B, T, n_embd)
    W_q = torch.randn(n_embd, n_embd) / math.sqrt(n_embd)   # 假装是训练好的权重
    W_k = torch.randn(n_embd, n_embd) / math.sqrt(n_embd)
    W_v = torch.randn(n_embd, n_embd) / math.sqrt(n_embd)
    q, k, v = x @ W_q, x @ W_k, x @ W_v
    out_manual, att_manual = manual_attention(q, k, v, causal=True)

    # ---- torch 版：一行搞定 ----
    out_torch = F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
    ).squeeze(1)

    err = (out_manual - out_torch).abs().max().item()
    print(f"输入 x           : {tuple(x.shape)}")
    print(f"手写输出         : {tuple(out_manual.shape)}")
    print(f"torch 输出       : {tuple(out_torch.shape)}")
    print(f"注意力权重       : {tuple(att_manual.shape)}   （每行和为 "
          f"{att_manual.sum(-1).mean().item():.6f}）")
    print(f"\n两者最大数值误差 : {err:.3e}   ← 应当接近 0")
    print()
    print("结论：F.scaled_dot_product_attention 做的事和我们手推的公式完全一样，")
    print("      softmax(QKᵀ/√d_k + mask) V，只是底层做了算子融合与访存优化。")
    print("      看懂手写版，就等于看懂了它 —— 这句话在 04 讲还会再用一次。")

    # ---- 因果性：只改最后一个 token，前面的输出必须一点不变 ----
    print()
    print("因果性验证：只修改**最后一个** token 的输入，看前面位置的输出变不变。")
    x2 = x.clone()
    x2[:, -1, :] += 5.0                                  # 只动最后一个位置
    q2, k2, v2 = x2 @ W_q, x2 @ W_k, x2 @ W_v
    out2, _ = manual_attention(q2, k2, v2, causal=True)

    diff_past = (out_manual[:, :-1] - out2[:, :-1]).abs().max().item()
    diff_last = (out_manual[:, -1] - out2[:, -1]).abs().max().item()
    print(f"    前 T-1 个位置输出的最大变化量 : {diff_past:.3e}   ← 必须严格为 0")
    print(f"    最后一个位置输出的最大变化量  : {diff_last:.3e}   ← 应该明显大于 0")
    print()
    print("前者严格为 0，说明信息只能从左流向右：位置 i 的输出只依赖 0..i。")
    print("后者不为 0，说明修改确实被「看见」了 —— 掩码没有把一切都挡掉。")
    print("（如果 diff_last 也是 0，那说明掩码把整行都屏蔽了，权重会退化，")
    print("  这正好也是第 9 小节可视化要检查的东西。）")


# ======================================================================
# 7. 让注意力自己学：单层注意力小模型
# ======================================================================
class TinyAttentionLM(nn.Module):
    """只有一层注意力的小语言模型：token 嵌入 + 位置嵌入 + 多头因果自注意力 + 输出头。

    和 01 讲 bigram 的唯一区别是中间那一层注意力：
        bigram : 位置 i 只用 x[i] 预测下一个 token（上下文窗口 = 1）
        本模型 : 位置 i 用 x[0..i] 的加权平均预测下一个 token（上下文窗口 = T）

    刻意**不加 MLP**，这样参数量能压到和 bigram 同一量级，
    让「同样大小的模型，能不能用上上下文」成为唯一的变量。
    """

    def __init__(self, vocab_size: int, n_embd: int = 32, block_size: int = 64,
                 n_head: int = 4):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln = nn.LayerNorm(n_embd)          # 残差后的归一化，05 讲细讲
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_weights: bool = False):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"序列长度 {T} 超过 block_size={self.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)      # (B, T, C)

        attn_out, attn_w, _ = self.attn(x, return_weights=return_weights)  # 三元组
        x = self.ln(x + attn_out)                        # 残差连接

        logits = self.head(x)                            # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss, attn_w


@torch.no_grad()
def evaluate_loss(model, ds, device, block_size: int, batch_size: int = 32,
                  n_batches: int = 10) -> float:
    """在验证集上多抽几批求平均，减少单批采样带来的噪声。"""
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        xb, yb = ds.get_batch("val", batch_size, block_size, device)
        total += model(xb, yb)[1].item()
    return total / n_batches


def train_bigram_baseline(ds, device, steps: int, block_size: int = 64,
                          batch_size: int = 32) -> dict:
    """01 讲的 bigram：只有一张 (vocab, vocab) 的查表，完全没有上下文。"""
    set_seed(1337)
    model = nn.Embedding(ds.vocab_size, ds.vocab_size).to(device)   # 参数量 = V²
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(steps):
        xb, yb = ds.get_batch("train", batch_size, block_size, device)
        loss = F.cross_entropy(model(xb).view(-1, ds.vocab_size), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val = 0.0
        for _ in range(10):
            vx, vy = ds.get_batch("val", batch_size, block_size, device)
            val += F.cross_entropy(model(vx).view(-1, ds.vocab_size), vy.view(-1)).item()
    return {"params": num_params(model), "val": val / 10}


def demo_learning_attention(ds, device: torch.device, steps: int = 600,
                            n_embd: int = 32, n_head: int = 4,
                            block_size: int = 64) -> TinyAttentionLM:
    print(banner("7. 让注意力自己学：单层注意力模型 vs 01 讲的 bigram"))
    print("语料是英文戏剧（TinyShakespeare）。我们训练一个**只有一层注意力**的小模型，")
    print("它没有 MLP、没有多层堆叠，全部本事就是「对历史做加权平均」。")
    print("如果它的 loss 明显低于 bigram，就说明上下文信息是真有用的。\n")

    set_seed(1337)
    model = TinyAttentionLM(ds.vocab_size, n_embd, block_size, n_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    n_params = num_params(model)
    print(f"模型配置: n_embd={n_embd}, n_head={n_head}, block_size={block_size}")
    print(f"参数量  : {n_params:,}  ({human_params(n_params)})")
    print(f"\n{'step':>6} {'train loss':>12} {'val loss':>12}")
    print("-" * 32)
    log_every = max(1, steps // 6)
    for step in range(steps + 1):
        model.train()
        xb, yb = ds.get_batch("train", 32, block_size, device)
        _, loss, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            val = evaluate_loss(model, ds, device, block_size)
            print(f"{step:>6} {loss.item():>12.4f} {val:>12.4f}")

    val_final = evaluate_loss(model, ds, device, block_size)
    train_final = loss.item()
    baseline = math.log(ds.vocab_size)

    print()
    print(f"训练 loss      : {train_final:.4f}")
    print(f"验证 loss      : {val_final:.4f}")
    print(f"随机猜测 ln(V) : {baseline:.4f}   (V={ds.vocab_size})")
    print(f"比随机猜测低   : {baseline - val_final:+.4f}")
    print()
    print("再和 01 讲的 bigram 基线在同一条件下正面对比一次：")
    big = train_bigram_baseline(ds, device, steps, block_size)
    print(f"    {'模型':<22} {'参数量':>10} {'验证 loss':>12} {'比 bigram 低':>14}")
    print("    " + "-" * 62)
    print(f"    {'bigram（无上下文）':<22} {big['params']:>10,} {big['val']:>12.4f} {'——':>14}")
    gain = big["val"] - val_final
    print(f"    {'单层注意力（本文）':<22} {n_params:>10,} {val_final:>12.4f} "
          f"{gain:>+13.4f}")
    print()
    print("怎么读这个结果（注意不要过度解读）：")
    print(f"  1. 本模型参数量约是 bigram 的 {n_params / big['params']:.1f} 倍，"
          f"但验证 loss 低了 {gain:.3f}。")
    print("     单看这个差距不算惊人，可它只花了**一层**注意力、还没有 MLP 帮忙。")
    print("  2. bigram 的天花板是可以算出来的：它每个位置只看到 1 个 token，")
    print("     所以永远学不会「the 后面该接什么」这种需要更长上下文的东西。")
    print("  3. 真正重要的是趋势：一旦模型能对整段历史做加权平均，")
    print("     05 讲再把它堆成多层并加上 MLP，loss 会继续往下掉。")
    print()
    print("第 9 小节会把训练好的注意力权重画出来，看看它到底在看哪里。")
    return model


# ======================================================================
# 8. 正式实现：Head
# ======================================================================
def demo_head_module(n_embd: int = 32, head_size: int = 32, block_size: int = 16,
                     B: int = 2, T: int = 10) -> None:
    print(banner("8. 正式实现：common/attention.py 里的 Head"))
    set_seed(0)
    head = Head(n_embd, head_size, block_size)
    x = torch.randn(B, T, n_embd)

    out, att = head(x, return_weights=True)
    print(f"输入 x            : {tuple(x.shape)}        (B, T, n_embd)")
    print(f"输出 out          : {tuple(out.shape)}        (B, T, head_size)")
    print(f"注意力权重 att    : {tuple(att.shape)}        (B, T, T)")
    print()
    print("内部四步（对应源码里的四行）：")
    print(f"    q = self.query(x)                       -> {tuple(head.query(x).shape)}")
    print(f"    att = q @ k.transpose(-2, -1) / √hs     -> {tuple(att.shape)}")
    print("    att = att.masked_fill(tril == 0, -inf)  -> 上三角变 0")
    print(f"    out = att @ v                           -> {tuple(out.shape)}")
    print()
    print("参数量审计：")
    print(f"    query : n_embd × head_size = {n_embd} × {head_size} = {num_params(head.query):,}")
    print(f"    key   : n_embd × head_size = {n_embd} × {head_size} = {num_params(head.key):,}")
    print(f"    value : n_embd × head_size = {n_embd} × {head_size} = {num_params(head.value):,}")
    qkv = num_params(head.query) + num_params(head.key) + num_params(head.value)
    print(f"    合计  : {qkv:,} = 3 × n_embd × head_size = 3 × {n_embd} × {head_size}")
    print("（Head 本身没有输出投影，那是 04 讲 MultiHeadAttention 才加的东西。）")
    print()
    print("自检：Head 的输出应当与手写公式逐位一致：")
    q, k, v = head.query(x), head.key(x), head.value(x)
    out_manual, att_manual = manual_attention(q, k, v, causal=True)
    e1 = (out - out_manual).abs().max().item()
    e2 = (att - att_manual).abs().max().item()
    print(f"    输出最大误差     : {e1:.3e}")
    print(f"    权重最大误差     : {e2:.3e}")
    print(f"    权重上三角最大值 : {att.triu(diagonal=1).abs().max().item():.3e}   (应为 0)")


# ======================================================================
# 9. 注意力可视化
# ======================================================================
_SHADES = ((0.02, " "), (0.08, "."), (0.20, ":"), (0.50, "*"))


def _shade(v: float) -> str:
    """把一个权重值映射成一个字符，用来在终端里画热力图。"""
    for thresh, ch in _SHADES:
        if v < thresh:
            return ch
    return "#"


def demo_visualize_attention(ds, device: torch.device, model=None, n_show: int = 16) -> None:
    print(banner("9. 注意力可视化：用字符热力图看「下三角」"))
    text = "First Citizen:\nBefore we proceed"
    ids = ds.tokenizer.encode(text)
    chars = [c if c != "\n" else "\\n" for c in text]
    n_show = min(n_show, len(chars))

    if model is not None:
        model.eval()
        with torch.no_grad():
            x = torch.tensor([ids], device=device)
            _, _, weights = model(x, return_weights=True)     # (B, n_head, T, T)
        att = weights[0].mean(dim=0).cpu()                    # 多头取平均 -> (T, T)
        title = "训练后模型的注意力（4 头平均）"
    else:
        set_seed(0)
        head = Head(32, 32, 64).to(device)
        with torch.no_grad():
            att = head(torch.randn(1, len(ids), 32, device=device),
                       return_weights=True)[1][0].cpu()
        title = "随机初始化单头的注意力"

    print(f"示例文本（{len(chars)} 个字符）: {text!r}")
    print(f"{title}，只显示前 {n_show} 个位置。\n")
    print("        " + "".join(f"{c:>2}" for c in chars[:n_show]) + "   ← 被看的位置 (key)")
    for i in range(n_show):
        row = "".join(f"{_shade(att[i, j].item()):>2}" for j in range(n_show))
        print(f"  {chars[i]:>4}  {row}   | {att[i, :n_show].sum().item():.2f}")
    print("\n（行末的数字是「该行落在已显示列上的权重之和」。")
    print("  前面的行会小于 1，是因为它们的部分权重落在更早的位置上，只是没显示；")
    print("  但**整行**（所有列）的和必须严格等于 1。）")
    print()
    print("字符图例:  ' ' <0.02   '.' <0.08   ':' <0.20   '*' <0.50   '#' >=0.50")
    print()
    print("形状才是重点：右上角一片空白（看不见未来），左下角有内容（能看见过去）。")

    # ---- 两项硬性自检 ----
    upper = att.triu(diagonal=1).abs().sum().item()
    row_sums = att.sum(-1)
    print()
    print("自检 1：上三角（未来位置）权重绝对值之和 = "
          f"{upper:.3e}   {'（严格为 0）' if upper == 0.0 else '（不是 0，有问题！）'}")
    print(f"自检 2：每一行权重之和 min={row_sums.min().item():.6f}, "
          f"max={row_sums.max().item():.6f}   （应全部为 1）")
    print(f"        行和为 1 的最大偏差 = {(row_sums - 1).abs().max().item():.3e}")

    # ---- 前几个位置「在看谁」----
    print("\n前 6 个位置各自的注意力分布（理解为「它在向谁取信息」）：")
    for i in range(min(6, n_show)):
        pairs = sorted(((att[i, j].item(), j) for j in range(i + 1)), reverse=True)[:3]
        desc = "  ".join(f"{chars[j]!r}: {p:.3f}" for p, j in pairs)
        print(f"    位置 {i:>2} {chars[i]!r:>6} -> {desc}")
    print()
    print("训练初期这些权重大体是均匀的（≈ 1/(i+1)）；训练之后会逐渐分化，")
    print("有的行集中到前一个字符，有的行摊平到整段历史 —— 这正是注意力在「学」。")

    _maybe_plot(att, chars, n_show)


def _setup_cjk_font() -> bool:
    """让 matplotlib 能画出中文标题。

    找不到中文字体时返回 False —— 调用方会改用纯 ASCII 标签，
    因为默认的 DejaVu Sans 缺中文字形，会在 stderr 上刷一大堆
    Glyph missing 警告（PowerShell 会因此判定脚本失败）。
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                 "Source Han Sans SC", "Malgun Gothic", "MS Gothic"):
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _maybe_plot(att: torch.Tensor, chars: list, n_show: int) -> None:
    """装了 matplotlib 就顺手存一张热力图，没装就跳过（不影响课程主流程）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n（未安装 matplotlib，跳过图片输出；终端字符热力图已经够用）")
        return

    cjk = _setup_cjk_font()
    if cjk:
        labels = {"xlabel": "被关注的位置 key", "ylabel": "当前位置 query",
                  "title": "因果自注意力权重（上三角恒为 0）"}
    else:
        # 没有中文字体就用英文标签，避免缺字形警告
        labels = {"xlabel": "key position attended to", "ylabel": "query position",
                  "title": "Causal self-attention weights (upper triangle = 0)"}

    out_dir = Path(__file__).resolve().parent.parent / "out" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(att[:n_show, :n_show].cpu().numpy(), cmap="viridis", vmin=0.0)
    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(chars[:n_show], fontsize=9)
    ax.set_yticklabels(chars[:n_show], fontsize=9)
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.set_title(labels["title"])
    fig.colorbar(im, ax=ax, shrink=0.85)
    path = out_dir / "03_attention_weights.png"

    # 即便字形全都命中，某些版本仍会为别的缺失字符报警告；这里统一压掉，
    # 保证脚本在 PowerShell 里的退出码不受画图影响。
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\n热力图已保存: {path}" + ("" if cjk else "（未找到中文字体，标题用英文）"))


# ======================================================================
def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="03 讲：自注意力 Attention(Q, K, V)")
    parser.add_argument("--dataset", default="tinyshakespeare", help="data/raw 下的语料名")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=600, help="小模型训练步数")
    parser.add_argument("--quick", action="store_true", help="快速模式：训练减到 150 步")
    args = parser.parse_args()

    print(banner("03 讲：自注意力 Attention(Q, K, V)"))
    device = pick_device(args.device)
    ds = load_dataset(args.dataset)
    print(ds.summary())
    print(f"使用设备: {device}")

    steps = 150 if args.quick else args.steps

    demo_weighted_average()
    demo_dot_product_similarity()
    demo_qkv_roles()
    demo_scaling()
    demo_causal_mask()
    demo_manual_vs_torch()
    model = demo_learning_attention(ds, device, steps=steps)
    demo_head_module()
    demo_visualize_attention(ds, device, model=model)

    print(banner("小结"))
    print("1. 注意力 = 用相关性当权重的加权平均，主体公式只有一行：")
    print("       Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + mask) V")
    print("2. Q 提问、K 应答、V 供内容；三者由同一个 x 线性变换而来（这就是「自」）。")
    print("3. 除以 √d_k 把点积方差拉回 1，防止 softmax 饱和成 one-hot、梯度消失。")
    print("4. 因果掩码让位置 i 只能看 0..i，这是 GPT 能「续写」而不是「抄答案」的前提。")
    print("5. 手写版与 torch 版数值一致，因果性可以严格验证为 0。")
    print("6. 单层注意力已经明显低于 bigram 的 loss —— 上下文信息确实有用。")
    print()
    print("但单头注意力只有「一套」关注模式：一个 softmax 权重向量。")
    print("下一讲：src\\04_multihead.py —— 把通道分给多个头，让模型同时用多种视角看历史。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
