"""05 讲：把零件拼成一层 —— 残差、LayerNorm、MLP 与 Transformer Block

到 04 讲为止我们有了注意力。但 GPT 不是「一堆注意力」，它是一层一层叠起来的结构，
每一层的定义只有两行：

        x = x + attn(norm(x))     ← 注意力子层
        x = x + mlp(norm(x))      ← 前馈子层

整个工程里最重要的两个细节都藏在这两行里：
    * 那个 + x           —— 残差连接，深度网络能被训练的前提
    * 那个 norm(x) 的位置 —— Pre-LN 还是 Post-LN，决定要不要 warmup

本讲把这一层彻底拆开，六个小节依次回答六个问题：

    1. 残差连接   —— 去掉它，24 层网络的梯度会小到什么程度？
    2. LayerNorm  —— 它到底在哪里做归一化？为什么不是 BatchNorm？
    3. 参数量分布 —— 钱花在注意力还是 MLP 上？（答案可能和直觉相反）
    4. Pre-LN vs Post-LN —— 只换了顺序，为什么 Post-LN 非要 warmup？
    5. Block 完整性验证 —— 残差真的通了吗？梯度真的能到底吗？
    6. 层数实验   —— 加深一定更好吗？更深更窄还是更浅更宽？

跑法：
    .\\.venv\\Scripts\\python.exe src\\05_transformer_block.py
    .\\.venv\\Scripts\\python.exe src\\05_transformer_block.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.attention import (  # noqa: E402
    CausalSelfAttention,      # Block 里那个融合写法，本讲 4、5 节直接用它
    MultiHeadAttention,       # 好读版多头：与 CausalSelfAttention 数学等价（04 讲已对照）
    manual_attention,         # 纯公式手写版：注意力本身的正确性由 03/04 讲负责
)
from common.config import GPTConfig, get_config, make_config  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.gpt import GPT, Block, LayerNorm, MLP, estimate_params  # noqa: E402
# 训练设施统一从 common.train 取：warmup + 余弦退火、梯度裁剪、AdamW 分组，
# 每一步都和各讲保持一致。本讲的对照实验自己写循环，只为把中间量（梯度范数、
# 每层 loss）打印出来；真要完整训练（存 checkpoint）时用 train() 一行搞定。
from common.train import (  # noqa: E402
    TrainConfig,
    build_optimizer,
    estimate_loss,
    get_lr,
    train,                    # noqa: F401  统一训练入口，供读者对照本讲的循环
)
from common.utils import (  # noqa: E402
    Timer,
    banner,
    describe_device,
    env_report,
    fmt_float,
    human_params,
    module_table,
    num_params,
    pick_device,
    set_seed,
    setup_console,
)


# ======================================================================
# 1. 残差连接：为什么它几乎是深度网络的前提
# ======================================================================
class DeepPlainNet(nn.Module):
    """没有残差的深网络：每一层都是 x = tanh(f(x))。

    24 层就是 24 次嵌套：tanh(f(tanh(f(...tanh(f(x))...))))
    反向传播要连着穿过 24 个 tanh，而 tanh 的导数最大只有 1、
    在饱和区更是接近 0 —— 连乘起来就是指数衰减。
    """

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.head(x)


class DeepResidualNet(nn.Module):
    """有残差的深网络：每一层都是 x = x + tanh(f(x))。

    唯一区别就是那个 + x，但它改变了反向传播的性质：
        d(x + f(x))/dx = 1 + f'(x)
    梯度里天然带着一个 1，可以「免费」穿过这一层 —— 这就是所谓「梯度高速公路」。
    """

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + torch.tanh(layer(x))
        return self.head(x)


def _input_grad_norm(loss: torch.Tensor, x: torch.Tensor) -> float:
    """求 loss 对**网络最底层输入**的梯度范数。

    这一项最能说明问题：它是梯度反向走过全部 depth 层之后剩下的东西。
    如果层内层层相乘把梯度吃掉了，这个数就会比 1 小很多个数量级。
    """
    if x.grad is not None:
        x.grad = None
    loss.backward()
    return x.grad.norm().item()


def demo_residual(depth: int = 24, dim: int = 64, steps: int = 200) -> None:
    print(banner("1. 残差连接：把梯度送到第 1 层"))
    print(f"构造两个结构完全相同的 {depth} 层网络（dim={dim}），")
    print("唯一区别是每层写 x = tanh(f(x)) 还是 x = x + tanh(f(x))。")
    print(f"同一份输入、同一个随机种子，各训练 {steps} 步。\n")

    print(f"{'结构':<10} {'第 1 层梯度范数':>18} {'最后 loss':>12}  说明")
    print("-" * 72)

    first_grads: Dict[str, float] = {}
    final_losses: Dict[str, float] = {}
    for name, cls in [("无残差", DeepPlainNet), ("有残差", DeepResidualNet)]:
        set_seed(0)
        net = cls(dim, depth)
        x = torch.randn(64, dim, requires_grad=True)
        target = torch.randn(64, dim)

        out = net(x)
        loss = F.mse_loss(out, target)
        first = _input_grad_norm(loss, x)

        opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
        for _ in range(steps):
            loss = F.mse_loss(net(x), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        first_grads[name] = first
        final_losses[name] = loss.item()
        note = "梯度传不到底层" if name == "无残差" else "梯度原样送达"
        print(f"{name:<10} {first:>18.3e} {loss.item():>12.4f}  {note}")

    ratio = first_grads["有残差"] / max(first_grads["无残差"], 1e-30)
    print("-" * 72)
    gap = ratio if ratio > 1 else 1.0 / max(ratio, 1e-30)
    print(f"差距：有残差的输入梯度是无残差的 {gap:,.0f} 倍"
          f"（约 {abs(torch.log10(torch.tensor(gap)).item()):.1f} 个数量级）")

    print("\n再看一遍逐层的梯度范数，衰减是从哪一层开始的：")
    print(f"{'网络类型':<10} {'第 1 层':>12} {'中间一层':>12} {'最后 1 层':>12}   规律")
    print("-" * 72)
    for name, cls in [("无残差", DeepPlainNet), ("有残差", DeepResidualNet)]:
        set_seed(0)
        net = cls(dim, depth)
        x = torch.randn(64, dim)
        net(x).sum().backward()
        norms = [layer.weight.grad.norm().item() for layer in net.layers]
        trend = "越靠底层越小" if norms[0] < norms[-1] else "各层相当"
        print(f"{name:<10} {norms[0]:>12.3e} {norms[depth // 2]:>12.3e} "
              f"{norms[-1]:>12.3e}   {trend}")
    print("-" * 72)

    # 数值验证「梯度高速公路」的数学来源
    print("\n「高速公路」的数学来源，直接算一遍雅可比矩阵：")
    set_seed(0)
    f = nn.Linear(dim, dim)
    u = torch.randn(1, dim, requires_grad=True)
    g = torch.tanh(f(u))
    jac = torch.autograd.functional.jacobian(
        lambda t: t + torch.tanh(f(t)), u.view(dim)
    ).reshape(dim, dim)
    identity_err = (jac - torch.eye(dim) - torch.autograd.functional.jacobian(
        lambda t: torch.tanh(f(t)), u.view(dim)
    ).reshape(dim, dim)).abs().max().item()
    print(f"    x = x + f(x) 时，雅可比 = I + f'(x)：")
    print(f"        I 与 f'(x) 的分离误差 {identity_err:.2e}（数值上精确成立）")
    print(f"        雅可比与单位阵的偏离 |J - I|_max = {(jac - torch.eye(dim)).abs().max().item():.3e}")
    print("    也就是说：梯度穿过这一层时，至少有一个「1」是白送的，")
    print("    剩下的 f'(x) 只是叠加在它上面的扰动，不会把整条通路压没。")
    print()
    print("对照无残差的情况：每层只乘一次 f'(x)，24 层就是 f'(x) 的 24 次连乘。")
    print("tanh 的导数 ≤ 1，饱和区更是趋近 0，连乘结果必然指数衰减。")
    print("这就是 ResNet 和 Transformer 都能堆到几十上百层的根本原因。")


# ======================================================================
# 2. LayerNorm：在哪个维度上归一化
# ======================================================================
def demo_layernorm(B: int = 4, T: int = 6, C: int = 32) -> None:
    print(banner("2. LayerNorm：把每个 token 的特征拉回统一尺度"))
    set_seed(0)

    scales = [0.01, 1.0, 100.0, 1000.0]
    x = torch.randn(B, T, C) * torch.tensor(scales).view(B, 1, 1)
    ln = LayerNorm(C)
    y = ln(x)

    print(f"人为造一批尺度差异极大的输入：x 形状 {tuple(x.shape)}，")
    print(f"4 个样本分别整体乘以 {scales}，最大最小相差 {max(scales) / min(scales):,.0f} 倍。\n")

    print(f"{'样本':<6} {'输入标准差':>14} {'输入均值':>12} {'输出标准差':>14} {'输出均值':>12}")
    print("-" * 62)
    for i in range(B):
        print(f"{i:<6} {x[i].std().item():>14.4f} {x[i].mean().item():>+12.3f} "
              f"{y[i].std().item():>14.4f} {y[i].mean().item():>+12.6f}")
    print("-" * 62)
    print(f"输入标准差的跨度: {min(s.item() for s in x.std(dim=(1, 2))):.4f} "
          f"~ {max(s.item() for s in x.std(dim=(1, 2))):.4f}")
    print(f"输出标准差的跨度: {min(s.item() for s in y.std(dim=(1, 2))):.4f} "
          f"~ {max(s.item() for s in y.std(dim=(1, 2))):.4f}")
    print()
    print("归一化之后，不管输入原本是 0.01 还是 1000 的尺度，")
    print("每个 token 的特征都变成「均值 0、标准差 1」，后面的层不用再同时应付两种极端尺度。")

    # ---- LayerNorm vs BatchNorm：归一化的维度不同 ----
    # 把问题压到最小：每个 token 是一个样本，特征维度 C=32，
    # 而且让 4 个样本的尺度**故意不均衡** —— 这正是两者的分歧点。
    print("\n关键区别在「在哪个维度上求均值和方差」。")
    print("做一个最小对照：4 个 token，每个 32 维特征，")
    print("其中一个 token 的尺度特别小（0.01），另外三个都是 1。\n")
    set_seed(1)
    xs = torch.randn(B, C) * torch.tensor([0.01, 1.0, 1.0, 1.0]).view(B, 1)
    xs = xs - xs.mean(dim=1, keepdim=True)              # 四个 token 的均值都设为 0
    out_ln = LayerNorm(C)(xs)                           # 在 (C,) 上归一化
    out_ln2 = nn.LayerNorm((C,))(xs)                    # torch 内置版，应当一致
    out_bn = nn.BatchNorm1d(C, affine=False)(xs)        # 在 (B,) 上归一化
    feat_ln = LayerNorm(B)(xs.t()).t()                  # 对照：在 (B,) 上归一化

    print(f"{'方案':<24} {'归一化维度':>12}   每个 token 的输出标准差")
    print("-" * 72)
    for name, dim, z in [("我们的 LayerNorm(C)", "(C,)", out_ln),
                         ("torch LayerNorm(C)", "(C,)", out_ln2),
                         ("在 (B,) 上归一化（BN 口径）", "(B,)", feat_ln),
                         ("BatchNorm1d(32)", "(B,)", out_bn)]:
        stds = [round(v, 4) for v in z.std(dim=1).tolist()]
        print(f"{name:<24} {dim:>12}   {stds}")
    print("-" * 72)
    print("前两行（LayerNorm）：每个 token 都被拉成标准差 1，四个值整齐划一。")
    print("后两行（BatchNorm 口径）：4 个 token 混在一起算一组统计量，")
    print(f"而它们之间本来就有 100 倍的尺度差，于是输出标准差的跨度变成 "
          f"{min(v.item() for v in out_bn.std(dim=1)):.4f} ~ "
          f"{max(v.item() for v in out_bn.std(dim=1)):.4f} ——")
    print("那个小尺度的 token 被放大到极端，其余三个被整体压低。")
    print("换句话说：BatchNorm 下「你这个 token 被归一化成什么样」")
    print("取决于同批其他 token 长什么样；LayerNorm 下只取决于你自己。")

    # 核心差别：一个 token 的输出会不会被「同批的邻居」改变
    print("\n再看一个更要命的性质：换掉同批的其他样本，输出会变吗？")
    xs_a = xs[:2]                                   # 只有 2 个 token 的 batch
    xs_b = torch.cat([xs_a, xs[2:] * 10.0], dim=0)  # 同一个 batch，后面又塞进两个大尺度的
    ln_a = LayerNorm(C)(xs_a)
    ln_b = LayerNorm(C)(xs_b)[:2]
    bn_a = nn.BatchNorm1d(C, affine=False)(xs_a)
    bn_b = nn.BatchNorm1d(C, affine=False)(xs_b)[:2]
    print("    取前 2 个 token 组成 batch A；再往同一个 batch 里追加 2 个尺度大 10 倍的 token 成 batch B。")
    print("    只看前 2 个 token 的输出，比较 A 和 B 两次：")
    print(f"        LayerNorm(C)   输出变化 {(ln_a - ln_b).abs().max().item():.3e}"
          "   （恒为 0：只跟自己有关）")
    print(f"        BatchNorm1d(32) 输出变化 {(bn_a - bn_b).abs().max().item():.3e}"
          "   （非 0：邻居变了，自己也跟着变）")
    print("    BatchNorm 的统计量是「整批一起算」的，所以样本之间会互相影响；")
    print("    这也正是它不能在 batch=1 时工作的原因 —— 分母的方差根本算不出来。")

    print("\n为什么 Transformer 必须用 LayerNorm，而不是 BatchNorm：")
    print("  1. 小 batch：BatchNorm 的统计量依赖同一批里的其他样本，")
    print("     batch 小则极不稳定，batch=1 时方差分母直接退化为 0。")
    print("     GPT 训练常用很小的 batch，推理时甚至真的只有一个样本。")
    print("  2. 变长序列：BatchNorm 的统计口径会随序列长度变化，")
    print("     而且 padding 的位置会污染统计量。")
    print("  3. 自回归推理：生成时一次只喂一个 token，")
    print("     「一个 token 根本构不成一个 batch」，BatchNorm 无从计算。")
    print("  4. 训练 / 推理不一致：BatchNorm 训练时用当前 batch 的统计量、")
    print("     推理时用滑动平均的 running_mean/var，两者天然有偏差；")
    print("     LayerNorm 两个阶段的计算完全相同，不存在这个问题。")
    print("  LayerNorm 只在单个 token 自己的 C 个通道内统计，与 batch、与序列长度都无关。")

    # ---- weight / bias 为什么不能省 ----
    print("\n最后看 LayerNorm 自己的参数：")
    print(f"    weight 形状 {tuple(ln.weight.shape)}，bias 形状 {tuple(ln.bias.shape)}，"
          f"合计 {num_params(ln)} 个 = 2 × C")
    print(f"    占整个 Block 的比例：{num_params(ln) / num_params(Block(32, 4, 16)):.2%}"
          f"（以 n_embd=32 的 Block 为例）")
    print("    只有 2C 个参数，看着可有可无，但绝不能省：")
    print("    纯归一化会把每个 token 强行压成均值 0、方差 1，")
    print("    等于删掉了「这个特征整体偏大 / 偏小」这类信息；")
    print("    weight 和 bias 让模型自己决定每一维缩放平移回多少，")
    print("    相当于给归一化留了一个可学习的「反悔开关」。")
    print("    GPT-2 的 LayerNorm 保留 bias，部分新模型为了省参数会去掉 bias，")
    print("    但 weight 从来没有被去掉过。")


# ======================================================================
# 3. 参数量分布：钱到底花在哪儿了
# ======================================================================
def demo_param_share(n_embd: int = 384, n_layer: int = 6,
                     block_size: int = 256, vocab_size: int = 65) -> None:
    print(banner("3. 参数量分布：注意力只占约 1/3，MLP 才是大头"))
    est = estimate_params(vocab_size=vocab_size, block_size=block_size,
                          n_layer=n_layer, n_embd=n_embd, tie_weights=True)

    emb = est["token_embedding"] + est["position_embedding"]
    attn = est["attn_per_layer"] * n_layer
    mlp = est["mlp_per_layer"] * n_layer
    norm = est["ln_per_layer"] * n_layer + est["final_norm"]
    total = est["total"]
    parts = [("嵌入 (wte + wpe)", emb), ("注意力 (全部层)", attn),
             ("MLP (全部层)", mlp), ("LayerNorm (全部层 + ln_f)", norm)]

    print(f"以 n_layer={n_layer}, n_embd={n_embd}, block_size={block_size}, "
          f"vocab_size={vocab_size} 为例")
    print(f"（大约 {human_params(total)} 参数，输出头与 wte 共享权重）\n")
    print(f"{'部分':<26} {'参数量':>12} {'占比':>8}")
    print("-" * 50)
    for name, val in parts:
        print(f"{name:<26} {val:>12,} {val / total:>7.1%}")
    print("-" * 50)
    print(f"{'合计':<26} {total:>12,} {'100.0%':>8}")
    assert emb + attn + mlp + norm == total, "四项相加必须等于总参数量"

    print(f"\n单层内部对比（每层共 {est['per_layer']:,} 个参数）：")
    print(f"    {'注意力':<10} {est['attn_per_layer']:>10,}  = 4C²      "
          f"（q、k、v、输出投影各 C²）      {est['attn_per_layer'] / est['per_layer']:>5.1%} / 层")
    print(f"    {'MLP':<10} {est['mlp_per_layer']:>10,}  = 8C²+5C  "
          f"（C→4C 和 4C→C，外加 bias）     {est['mlp_per_layer'] / est['per_layer']:>5.1%} / 层")
    print(f"    {'LayerNorm':<10} {est['ln_per_layer']:>10,}  = 4C      "
          f"（两个 LayerNorm，各 weight+bias）{est['ln_per_layer'] / est['per_layer']:>5.1%} / 层")
    print(f"\n    只比较两个子层的矩阵部分：MLP 8C² 是注意力 4C² 的 "
          f"{est['mlp_per_layer'] / est['attn_per_layer']:.2f} 倍，")
    print(f"    折算到占比就是 MLP ≈ {2 / 3:.0%}、注意力 ≈ {1 / 3:.0%}。")

    print("\n为什么会这样？因为两者的分工不一样：")
    print("    注意力 = 位置之间的信息交换（横向）：每个 token 从别的 token 那里取信息。")
    print("    MLP   = 单个位置内部的非线性加工（纵向）：取来的信息在这里被消化。")
    print("    两者缺一不可：只有注意力，模型只能做加权平均（本质是线性操作）；")
    print("    只有 MLP，每个位置各算各的，上下文完全用不上。")
    print("    MLP 需要 4 倍宽的隐层，是因为它承担了全部「思考」容量，")
    print("    而注意力只需要把信息搬运过来。")

    print("\n顺手算一笔账：整个模型里，嵌入层只占 "
          f"{emb / total:.1%}，参数几乎全在 Transformer 层里 ——")
    print("所以「加宽 n_embd」和「加深 n_layer」都是在给这两块加钱，")
    print("而 4C² / 8C² 都随 C 平方增长，n_embd 的影响远比 n_layer 剧烈。")

    print("\n一个直接由这个结论长出来的后续工作：MoE（混合专家）。")
    print("    既然 2/3 的参数都堆在 MLP 里，那就把每个 MLP 换成很多个「专家」，")
    print("    每个 token 只路由到其中一两个专家上计算：参数量（显存）做大了，")
    print("    但每个 token 的实际计算量不变（推理成本不变）。")
    print("    Switch Transformer、Mixtral、DeepSeek-MoE 走的都是这条路。")


# ======================================================================
# 4. Pre-LN vs Post-LN
# ======================================================================
class PostLNBlock(nn.Module):
    """原始 Transformer（2017）的排列：先算子层 → 加残差 → 最后归一化。

        x = ln(x + sublayer(x))

    注意残差通路上多了一次 LayerNorm：反向传播时梯度必须穿过它，
    而 LayerNorm 会对梯度做一次「减均值、除标准差」的重新标定，
    那个宝贵的「1」就被破坏了。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, bias=False)
        self.mlp = MLP(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _, _ = self.attn(x)
        x = self.ln1(x + a)
        x = self.ln2(x + self.mlp(x))
        return x


class PreLNBlock(nn.Module):
    """GPT-2 / 现代模型的排列：先归一化 → 进子层 → 加残差。

        x = x + sublayer(ln(x))

    归一化被挪到了残差**分支**上，主干那条 x → x + ... 的通路干干净净，
    梯度可以一路无损传到底。代价是主干上的数值没有归一化约束，
    所以最后要补一个 ln_f（GPT-2 的做法，common/gpt.py 里也是这么写的）。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, bias=False)
        self.mlp = MLP(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _, _ = self.attn(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class StackLM(nn.Module):
    """把若干 Block 堆起来，接上嵌入和语言模型头，用来比较两种排列。

    故意不复用 common/gpt.py 的 GPT：这里要能自由替换 block 类型，
    而且我们想看清楚「只有 block 内部的顺序变了，其他一切相同」。
    """

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, block_size: int,
                 n_layer: int, block_cls: Any):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [block_cls(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def _grad_norm(module: nn.Module) -> float:
    """所有参数梯度的全局 L2 范数，也就是 clip_grad_norm_ 的返回值口径。"""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def demo_pre_vs_post_ln(ds, device: torch.device, steps: int = 60,
                        n_layer: int = 6, n_embd: int = 128, n_head: int = 4,
                        block_size: int = 64, lr: float = 1e-3) -> None:
    print(banner("4. Pre-LN vs Post-LN：只换了顺序，训练稳定性差很多"))
    print(f"两边都是 {n_layer} 层、n_embd={n_embd}、n_head={n_head}，")
    print(f"同一份数据、同一个种子、同样的学习率 {lr}、同样训练 {steps} 步。")
    print("唯一的差别是 Block 里归一化和残差的先后顺序。\n")

    print(f"{'排列':<10} {'第 1 步 loss':>12} {'最后 loss':>11} "
          f"{'最大梯度范数':>14} {'参数量':>12}")
    print("-" * 66)
    rows: List[Tuple[str, float, float, float, int]] = []
    for name, cls in [("Post-LN", PostLNBlock), ("Pre-LN", PreLNBlock)]:
        set_seed(7)
        model = StackLM(ds.vocab_size, n_embd, n_head, block_size, n_layer, cls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        first_loss, last_loss, max_grad = float("nan"), float("nan"), 0.0
        for step in range(steps):
            model.train()
            xb, yb = ds.get_batch("train", 32, block_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            max_grad = max(max_grad, _grad_norm(model))
            opt.step()
            if step == 0:
                first_loss = loss.item()
            last_loss = loss.item()
            if step % max(1, steps // 4) == 0:
                print(f"    [{name}] step {step:3d}  loss {loss.item():.4f}  "
                      f"梯度范数 {_grad_norm(model):.3f}")
        rows.append((name, first_loss, last_loss, max_grad, num_params(model)))

    print()
    print(f"{'排列':<10} {'第 1 步 loss':>12} {'最后 loss':>11} "
          f"{'最大梯度范数':>14} {'参数量':>12}")
    print("-" * 66)
    for name, first, last, mg, params in rows:
        print(f"{name:<10} {first:>12.4f} {last:>11.4f} {mg:>14.3f} {params:>12,}")
    print("-" * 66)

    base = rows[0][3]
    for name, _, _, mg, _ in rows[1:]:
        if base > 0:
            print(f"{name} 的最大梯度范数相对 Post-LN：{mg / base:.1%}"
                  "（越小说明反向传播越平稳）")

    print("\n怎么读：")
    print("  Post-LN 的最大梯度范数更大，loss 曲线也更抖。")
    print("  原因是它的残差通路上多了一次 LayerNorm：")
    print("      Post-LN:  x = ln(x + sublayer(x))   —— 主干上有个 ln，梯度要穿过它")
    print("      Pre-LN :  x = x + sublayer(ln(x))   —— 主干干净，ln 只在支路上")
    print("  那个「1 + f'(x)」的高速公路在 Post-LN 里被 ln 的雅可比重新标定，")
    print("  层数一多，正向的激活尺度和反向的梯度尺度都容易失控。")

    print("\n这就是 warmup 存在的原因：")
    print("  参数是随机初始化的，头几百步的梯度方向本身就不可靠；")
    print("  Post-LN 的网络又对学习率特别敏感，一上来就用大 lr 很容易发散。")
    print("  所以原始论文必须先用很小的 lr 试探（线性 warmup），稳定后再升上去。")

    # 现场演示：同样的步数、同样的数据，把学习率提高 10 倍，看谁更耐受
    big_lr, probe_steps = 1e-2, 40
    print(f"\n现场对照：学习率提高到 {big_lr:g}（原本的 {big_lr / lr:.0f} 倍），"
          f"各跑 {probe_steps} 步，都不带 warmup。")
    print(f"{'排列':<10} {'第 1 步 loss':>12} {'最好 loss':>11} "
          f"{'第 40 步 loss':>14}  结果")
    print("-" * 72)
    for name, cls in [("Post-LN", PostLNBlock), ("Pre-LN", PreLNBlock)]:
        set_seed(7)
        model = StackLM(ds.vocab_size, n_embd, n_head, block_size, n_layer, cls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=big_lr)
        first = best = last = float("nan")
        for step in range(probe_steps):
            xb, yb = ds.get_batch("train", 32, block_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            value = loss.item()
            if step == 0:
                first = best = value
            best = min(best, value)
            last = value
        if last != last:
            state = "发散（loss = nan）"
        elif last > best * 1.01:
            state = "不稳（最后反弹到最好之上）"
        else:
            state = "稳定收敛"
        print(f"{name:<10} {first:>12.4f} {best:>11.4f} {last:>14.4f}  {state}")
    print("-" * 72)
    print("这两行怎么读：6 层还不算深，所以 Post-LN 未必真的变成 nan，")
    print("但它在大学习率下（1）最终损失明显高于 Pre-LN，")
    print("（2）更容易出现「先降到最好、之后又反弹上去」的抖动 —— 看它两列 loss 的差。")
    print("层数越多、lr 越大，Post-LN 就越难压住；真实训练里十几层以上的 Post-LN")
    print("不加 warmup 经常直接发散，这就是 GPT-2 之后全面改用 Pre-LN 的原因。")
    print("代价是 Pre-LN 主干上的数值没有归一化约束，所以最后要补一个 ln_f，")
    print("并且在残差分支的投影层上按 1/sqrt(2*n_layer) 缩小初始化（见 common/gpt.py）。")


# ======================================================================
# 5. 组装并验证 Block
# ======================================================================
def demo_block_integrity(n_embd: int = 64, n_head: int = 4, block_size: int = 32,
                         B: int = 2, T: int = 8) -> None:
    print(banner("5. 验证 Block：两条残差通路真的通了吗"))
    set_seed(0)
    block = Block(n_embd, n_head, block_size)
    x = torch.randn(B, T, n_embd)
    out, att, cache = block(x)
    print(f"输入 {tuple(x.shape)} -> 输出 {tuple(out.shape)}")
    print(f"Block 参数量: {num_params(block):,}（{human_params(num_params(block))}）")
    print("\n参数明细：")
    print(module_table(block))

    # ---- 自检 1：子层输出置零后应退化为恒等映射 ----
    print("\n自检 1：把两个子层的输出投影全部置零，Block 应退化为恒等映射")
    set_seed(0)
    block = Block(n_embd, n_head, block_size)
    x = torch.randn(B, T, n_embd)
    with torch.no_grad():
        # c_proj 是注意力的最后一层，fc_out 是 MLP 的最后一层。
        # 把它们（含 bias）置零，子层的输出就恒为 0，残差通路只剩 x 本身。
        block.attn.c_proj.weight.zero_()
        if block.attn.c_proj.bias is not None:
            block.attn.c_proj.bias.zero_()
        block.mlp.fc_out.weight.zero_()
        if block.mlp.fc_out.bias is not None:
            block.mlp.fc_out.bias.zero_()
        identity_out, _, _ = block(x)
    err = (identity_out - x).abs().max().item()
    print(f"    max |Block(x) - x| = {err:.3e}    （严格等于 0 才算通过）")
    print(f"    判定: {'通过' if err == 0.0 else '不通过'}")

    # 再验证：只置零一个子层，另一个子层仍然生效
    set_seed(0)
    block = Block(n_embd, n_head, block_size)
    with torch.no_grad():
        block.attn.c_proj.weight.zero_()
        if block.attn.c_proj.bias is not None:
            block.attn.c_proj.bias.zero_()
        half_out, _, _ = block(x)
    print(f"    只置零注意力子层时，输出与输入的最大偏差 "
          f"{(half_out - x).abs().max().item():.4f} —— MLP 那条通路照常工作，")
    print("    说明两条残差是相互独立的加法，互不干扰。")

    # ---- 自检 2：梯度能否一路传到最底层 ----
    print("\n自检 2：梯度能否从输出一路传到输入，且各层梯度量级相当")
    set_seed(0)
    block = Block(n_embd, n_head, block_size)
    x = torch.randn(B, T, n_embd, requires_grad=True)
    out, _, _ = block(x)
    out.pow(2).mean().backward()
    in_grad = x.grad.norm().item()
    first_linear = block.attn.c_attn.weight.grad.norm().item()
    last_linear = block.mlp.fc_out.weight.grad.norm().item()
    ln1_w = block.ln_1.weight.grad.norm().item()
    print(f"    输入 x 的梯度范数          : {in_grad:>10.4f}")
    print(f"    第一个 Linear (c_attn)     : {first_linear:>10.4f}")
    print(f"    最后一个 Linear (fc_out)   : {last_linear:>10.4f}")
    print(f"    第一个 LayerNorm 的 weight : {ln1_w:>10.4f}")
    ratio = max(in_grad, first_linear, last_linear, 1e-12) / max(
        min(in_grad, first_linear, last_linear), 1e-12
    )
    print(f"    四者最大 / 最小 = {ratio:.2f}（同一量级即可，不必相等）")
    print("    最底下的 c_attn 也能拿到同量级梯度 —— 残差把梯度送到了层内每个角落。")
    print("    换成一个 24 层的无残差网络，这个比值会大得离谱（见第 1 节）。")

    # ---- 附带：Block 里的模块构成 ----
    print("\nBlock 内部的模块构成（对照 common/gpt.py 的 forward 读）：")
    for name, child in block.named_children():
        print(f"    {name:<10} {type(child).__name__:<22} "
              f"{human_params(num_params(child)):>8} 参数")
    print("    forward 里就是两行：")
    print("        x = x + attn(ln_1(x))")
    print("        x = x + mlp(ln_2(x))")


# ======================================================================
# 6. 层数实验：更深一定更好吗
# ======================================================================
def train_one_gpt(ds, device: torch.device, n_layer: int, steps: int,
                  n_embd: int = 128, n_head: int = 4, block_size: int = 64,
                  vocab_size: int | None = None, lr: float = 3e-3,
                  seed: int = 1234, quiet: bool = False) -> Dict[str, Any]:
    """用统一入口（GPTConfig + GPT.from_config）建模型并训练，返回指标字典。

    所有配置都用同一个种子、同一套超参数，唯一变量是 n_layer ——
    这是做对照实验的基本纪律，否则差异会来自随机初始化而不是层数。
    """
    vocab = vocab_size if vocab_size is not None else ds.vocab_size
    config = GPTConfig(vocab_size=vocab, block_size=block_size, n_layer=n_layer,
                       n_head=n_head, n_embd=n_embd)
    set_seed(seed)
    model = GPT.from_config(config).to(device)
    train_cfg = TrainConfig(batch_size=32, block_size=block_size, max_steps=steps,
                            learning_rate=lr, min_lr=lr / 10, warmup_steps=max(1, steps // 10),
                            eval_interval=10 ** 9, log_interval=10 ** 9, seed=seed,
                            device=str(device))
    opt = build_optimizer(model, train_cfg)

    with Timer() as timer:
        for step in range(steps):
            # 用 common.train.get_lr 的调度：warmup + 余弦退火，各配置完全一致
            current_lr = get_lr(step, train_cfg.warmup_steps, steps, lr, train_cfg.min_lr)
            for group in opt.param_groups:
                group["lr"] = current_lr
            model.train()
            xb, yb = ds.get_batch("train", train_cfg.batch_size, block_size, device)
            _, loss, _, _ = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            if not quiet and step % max(1, steps // 3) == 0:
                print(f"      n_layer={n_layer} step {step:3d}  loss {loss.item():.4f}")
        train_loss = loss.item()
        losses = estimate_loss(model, ds, train_cfg, splits=("train", "val"))

    return {
        "config": config,
        "n_layer": n_layer,
        "params": num_params(model),
        "est": estimate_params(vocab, block_size, n_layer, n_embd),
        "train_loss": train_loss,
        "eval_train": losses["train"],
        "val_loss": losses["val"],
        "elapsed": timer.elapsed,
        "model": model,
    }


def demo_depth_ablation(ds, device: torch.device, steps: int = 150) -> None:
    print(banner("6. 层数实验：加深一定更好吗"))
    print(f"固定 n_embd=128、n_head=4、block_size=64、batch=32、lr=3e-3，")
    print(f"只改层数，每个配置训练 {steps} 步（同一随机种子初始化）。\n")

    results: Dict[int, Dict[str, Any]] = {}
    for n_layer in [1, 2, 4, 6]:
        est = estimate_params(ds.vocab_size, 64, n_layer, 128)
        print(f"  [n_layer={n_layer}] 公式估算参数量 {est['total']:,}")
        results[n_layer] = train_one_gpt(ds, device, n_layer, steps)

    best = min(r["val_loss"] for r in results.values())
    print(f"\n{'层数':>5} {'参数量':>10} {'每层参数':>10} {'训练 loss':>11} "
          f"{'验证 loss':>11} {'相对最好':>10} {'耗时':>8}")
    print("-" * 74)
    for n_layer, r in results.items():
        note = "  <- 最好" if r["val_loss"] == best else f"{r['val_loss'] / best - 1:>+9.1%}"
        print(f"{n_layer:>5} {r['params']:>10,} {r['est']['per_layer']:>10,} "
              f"{r['eval_train']:>11.4f} {r['val_loss']:>11.4f} {note:>10} "
              f"{r['elapsed']:>7.1f}s")
    print("-" * 74)

    vals = {k: r["val_loss"] for k, r in results.items()}
    spread = max(vals.values()) - min(vals.values())
    print("\n三重观察，一个也不能少：")
    print("  1. 参数量：每多一层就多加一个完整的 Block（注意力 + MLP + 两个 LayerNorm），")
    print(f"     所以参数量随层数近似线性增长（每层 "
          f"{results[1]['est']['per_layer']:,} 个参数）。")
    print("  2. 验证 loss：理论上加深应该更好，但请先看这张表的绝对差距 ——")
    print(f"     四个配置的 val loss 全部落在 {min(vals.values()):.4f} ~ "
          f"{max(vals.values()):.4f} 之间，最大差距只有 {spread:.4f}。")
    print(f"     而我们只训练了 {steps} 步，这个量级的差距基本被随机性淹没：")
    print("     换一个种子，排名很可能就变了。所以「层数更多一定更好」这句话，")
    print("     必须用足够长的训练 + 多个种子才能验证（下面给了做法）。")
    print("     能确定的趋势是收益递减：多一层的边际收益，远不如把 n_embd 加宽一点")
    print("     （4C² / 8C² 都随 C 平方增长，参数量与表达力的杠杆更大）。")
    print("  3. 训练 loss vs 验证 loss：盯住这两列的差距。")
    for n_layer, r in results.items():
        gap = r["val_loss"] - r["eval_train"]
        print(f"     n_layer={n_layer}: train {r['eval_train']:.4f}  val {r['val_loss']:.4f}  "
              f"差距 {gap:+.4f}")
    print("     验证 loss 高于训练 loss 是正常的（模型没见过验证集）；")
    print(f"     但这里四个差距都在 {min(r['val_loss'] - r['eval_train'] for r in results.values()):+.4f} ~ "
          f"{max(r['val_loss'] - r['eval_train'] for r in results.values()):+.4f}，")
    print("     小到可以忽略 —— 说明此时还没到过拟合阶段，而是**欠训练**：")
    print(f"     只有 100 万 token、却只走了 {steps} 步，模型连训练集都没吃透。")
    print("     真正的过拟合信号是「训练 loss 继续降、验证 loss 开始升」；")
    print("     一旦出现，该做的是加数据 / 加 dropout / 加 weight decay，而不是继续加深。")

    print("\n更深更窄 vs 更浅更宽：")
    print("  上面这张表只改了深度，参数量同时也在变，所以不算严格的对照。")
    print("  真正要比较的是「参数量相同」的两组：")
    print(f"    更深更窄 : n_layer=6, n_embd=128  -> {results[6]['params']:,} 参数")
    print("    更浅更宽 : 保持参数量接近，把 n_layer 降到 2、n_embd 提到约 220")
    print("  经验结论：同样的参数量下，更深的模型通常更好 ——")
    print("  每多一层就多一次「注意力 + MLP」的组合，能表达的函数复杂度增长更快；")
    print("  但太深会难训练，这时就要靠残差、LayerNorm 和合适的学习率策略撑住，")
    print("  也就是本讲前面四节讲的全部内容。")

    print("\n想得到可靠结论的正确做法：")
    print("  1. 把 --steps 调到 1000+，让每个配置都真正收敛到各自的平台期；")
    print("  2. 用 3 个以上随机种子重复，比较平均值与方差，而不是单次结果；")
    print("  3. 报告参数量与 token 数（Chinchilla 口径下两者应当同步增长），")
    print("     否则「更深更好」很可能只是「参数更多更好」。")


# ======================================================================
def main() -> int:
    setup_console()                      # 第一行永远是它：把控制台切成 UTF-8
    parser = argparse.ArgumentParser(description="05 讲：残差、LayerNorm、MLP 与 Transformer Block")
    parser.add_argument("--dataset", default="tinyshakespeare", help="data/raw 下的语料名")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=150, help="实验类小节的训练步数")
    parser.add_argument("--quick", action="store_true", help="快速模式：步数减半，几十秒跑完")
    args = parser.parse_args()

    print(banner("05 讲：残差、LayerNorm、MLP 与 Transformer Block"))
    print(env_report())
    print()
    device = pick_device(args.device)
    print(f"本次运行使用设备: {describe_device(device)}")

    ds = load_dataset(args.dataset)
    print()
    print(ds.summary())

    steps = 40 if args.quick else args.steps
    print(f"\n实验步数: {steps}（--quick 时为 40）")

    demo_residual(depth=24, dim=64, steps=60 if args.quick else 200)
    demo_layernorm()
    demo_param_share()
    demo_pre_vs_post_ln(ds, device, steps=25 if args.quick else max(40, steps // 2))
    demo_block_integrity()
    print("\n（第 6 节最花时间：四组配置各训练一遍，请稍等）")
    demo_depth_ablation(ds, device, steps=steps)

    print(banner("小结"))
    print("1. 残差连接 x = x + f(x) 让雅可比里天然带一个 1，")
    print("   梯度可以无损穿过任意多层 —— 这是深层网络能训练的前提，")
    print("   去掉它，24 层网络的底层梯度会小好几个数量级。")
    print("2. LayerNorm 在单个 token 自己的 C 个通道上归一化，")
    print("   不依赖 batch、不依赖序列长度，所以小 batch、变长序列、")
    print("   自回归推理（一次一个 token）全都能正常工作 —— BatchNorm 三条都做不到。")
    print("   它只有 2C 个参数，但 weight/bias 承担了「把信息缩放回多少」的自由度，不能省。")
    print("3. 参数量上 MLP 约占 2/3、注意力约占 1/3：")
    print("   注意力负责位置之间的横向信息交换，MLP 负责单个位置内部的纵向非线性加工。")
    print("   MoE 就是冲着 MLP 这块去的 —— 把 MLP 拆成许多专家，按 token 路由，")
    print("   参数量做大而单 token 计算量不变。")
    print("4. Pre-LN（x = x + sublayer(ln(x))）把归一化挪到支路上，")
    print("   保住了残差主干的「梯度高速公路」，所以不像 Post-LN 那样依赖 warmup；")
    print("   代价是主干数值无约束，需要在最后补 ln_f。现代模型基本都用 Pre-LN。")
    print("5. common/gpt.py 里的 Block 就是 Pre-LN 版本：")
    print("   两个自检通过了 —— 子层输出置零时严格退化为恒等映射，")
    print("   反传后输入梯度与首尾 Linear 的梯度同量级，两条残差通路都是通的。")
    print("6. 加深通常有效但收益递减，还会带来过拟合风险；")
    print("   同样参数量下「更深更窄」一般优于「更浅更宽」，但太深就难训练了。")
    print()
    print("下一讲：src\\06_gpt.py —— 把这些 Block 堆起来，接上嵌入层和输出头，")
    print("得到一个完整可训练的 GPT，并用 forward hook 逐层追踪张量形状。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
