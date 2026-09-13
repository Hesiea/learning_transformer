r"""01 讲：张量、softmax、交叉熵，以及一个能跑的 bigram 基线模型

这一讲不碰注意力，目标是把后面所有代码都要用到的「地基」摸清楚：

    1. 环境自检        —— 确认 torch 真的能算
    2. 文本 -> 整数    —— 神经网络的输入只能是数字
    3. 张量的形状      —— (B, T, C) 这三个字母在本工程里反复出现
    4. softmax         —— 把任意实数变成概率分布（手写一遍）
    5. 交叉熵          —— 用「正确 token 的概率」衡量模型好坏（手写一遍，并和 torch 对照）
    6. bigram 基线模型 —— 完全没有注意力的最简语言模型，只有一张查表
    7. 训练前后对比    —— 看 loss 下降、看采样文本从乱码变成单词

跑法：
    .\.venv\Scripts\python.exe src\01_tensor_basics.py
    .\.venv\Scripts\python.exe src\01_tensor_basics.py --quick    # 快速版（300 步）

bigram 是本工程的「对照组」：后面几讲的注意力模型必须明显打赢它，否则说明哪里写错了。
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
    env_report,
    fmt_float,
    human_params,
    num_params,
    pick_device,
    set_seed,
    setup_console,
)


# ======================================================================
# 2. 文本 -> 整数
# ======================================================================
def demo_tokenizer(ds) -> None:
    print(banner("2. 文本 -> 整数：tokenizer 在做什么"))
    tok = ds.tokenizer
    text = tok.decode(ds.train[:160].astype(int).tolist())
    ids = tok.encode(text)

    print(f"词表大小 vocab_size = {tok.vocab_size}")
    print(f"前 40 个字符（按码点排序，所以顺序是确定的）: {''.join(tok.itos[:40])!r}")
    print()
    print(f"原文    : {text[:48]!r}")
    print(f"编码    : {ids[:24]}")
    print(f"解码回去: {tok.decode(ids[:24])!r}")
    print(f"是否可逆: {tok.decode(ids) == text}")
    print()

    print("逐个字符看「字符 <-> id」的映射（这是双向查表）：")
    print(f"{'字符':>6} {'id':>4}      {'id':>4} {'字符':>6}")
    print("-" * 30)
    for i in range(10):
        ch, idx = text[i], ids[i]
        back = tok.itos[idx]
        print(f"{ch!r:>6} {idx:>4}      {idx:>4} {back!r:>6}")
    print()
    print("关键认识：模型从头到尾看不到文字，只看到一串整数 id。")
    print("          stoi（字符->id）是词表，itos（id->字符）是它的反查表，")
    print("          tokenizer 就是这两张表加上一次列表推导式。")
    print()
    print("为什么先用字符级？词表小（这里只有 65），不需要外部依赖，也不会出现 [UNK]。")
    print("真实大模型用 BPE 做子词切分（词表 3 万 ~ 20 万），动机是：")
    print("    字符级序列太长（一个汉字一个 token，一段话几百步），")
    print("    词级词表太大且永远不够用，子词正好在两者之间取平衡。")


# ======================================================================
# 3. 张量的形状
# ======================================================================
def demo_shapes(ds, device: torch.device) -> None:
    print(banner("3. 张量的形状：本工程的三个核心字母 (B, T, C)"))
    set_seed(0)
    x, y = ds.get_batch("train", batch_size=4, block_size=8, device=device)

    print("用 ds.get_batch 抽一批数据：")
    print(f"    x.shape = {tuple(x.shape)}   # (B, T)：B 条序列，每条 T 个 token")
    print(f"    y.shape = {tuple(y.shape)}   # 与 x 同形：每个位置要预测的「下一个 token」")
    print(f"    x.dtype = {x.dtype}          # 整数 id，不是浮点")
    print()
    print("看第一条样本，理解 y 是怎么来的：")
    print(f"    x[0] = {x[0].tolist()}")
    print(f"    y[0] = {y[0].tolist()}   <- 正好是 x[0] 左移一位")
    print(f"    解码 x[0]: {ds.tokenizer.decode(x[0].tolist())!r}")
    print(f"    解码 y[0]: {ds.tokenizer.decode(y[0].tolist())!r}")
    print()
    print("语言模型的任务不需要人工标注：文本自己右移一位就是标签。")
    print("模型在这 8 个位置上同时做 8 道「下一个字符是什么」的题，")
    print("这就是所谓「并行训练」—— T 个位置的 loss 一次前向全部算出来。")
    print()

    print("往后再加一维 C（通道数 / 嵌入维度），整条流水线的形状是：")
    B, T = x.shape
    C, V = 32, ds.vocab_size
    print(f"    token id : (B, T)        = {tuple(x.shape)}    整数，每个元素 0..V-1")
    print(f"    embedding: (B, T, C)     = {(B, T, C)}    每个 token 变成 C 维向量")
    print(f"    logits   : (B, T, vocab) = {(B, T, V)}    每个位置对 V 个词的打分")
    print("    softmax  : (B, T, vocab) = "
          f"{(B, T, V)}    沿最后一维归一化成概率")
    print()
    print("一句话记住：(B, T, C) 是「一批序列、每个位置一个向量」，")
    print("            最后一维永远在做同一件事 —— 把向量映射成对词表的打分。")
    print()
    print(f"顺便看一眼维度对应的参数量：C={C} 时，把 C 维变成 V={V} 个打分需要 "
          f"{C * V:,} 个权重。")


# ======================================================================
# 4-5. softmax 与交叉熵
# ======================================================================
def my_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """手写 softmax：先减去最大值防止 exp 溢出，再归一化。

    为什么要减最大值？exp(100) 会直接变成 inf。减掉每行的最大值不改变结果
    （分子分母同乘一个常数），却让指数永远落在 (-inf, 0]，数值上稳定得多。
    torch 的内部实现也是这么做的。
    """
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    exps = shifted.exp()
    return exps / exps.sum(dim=dim, keepdim=True)


def my_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """手写交叉熵：-log(模型给正确答案的概率)，再对所有位置求平均。

    logits : (N, V)   targets : (N,)      N = B*T 个位置
    """
    probs = my_softmax(logits, dim=-1)
    log_probs = probs.log()
    picked = log_probs[torch.arange(targets.shape[0]), targets]
    return -picked.mean()


def demo_softmax_ce() -> None:
    print(banner("4-5. softmax 与交叉熵：手写版 vs torch 内置版"))
    set_seed(0)
    logits = torch.randn(6, 10)          # 6 个位置，词表大小 10
    targets = torch.randint(0, 10, (6,))

    print("输入 logits（6 个位置 × 10 个词）的第一行：")
    print(f"    {[round(v, 3) for v in logits[0].tolist()]}")
    print()

    # ---- softmax 对照 ----
    p_mine = my_softmax(logits)
    p_torch = F.softmax(logits, dim=-1)
    print("softmax 对照：")
    print(f"    手写第一行 : {[round(v, 4) for v in p_mine[0].tolist()]}")
    print(f"    torch 第一行: {[round(v, 4) for v in p_torch[0].tolist()]}")
    print(f"    最大误差   : {(p_mine - p_torch).abs().max().item():.3e}   <- 数值上完全一致")
    print(f"    每行求和   : {[round(v, 6) for v in p_mine.sum(-1).tolist()]}   (应为 1)")
    print()

    # ---- 交叉熵对照 ----
    ce_mine = my_cross_entropy(logits, targets)
    ce_torch = F.cross_entropy(logits, targets)
    print("交叉熵对照：")
    print(f"    手写  = {ce_mine.item():.6f}")
    print(f"    torch = {ce_torch.item():.6f}")
    print(f"    差值  = {abs(ce_mine.item() - ce_torch.item()):.3e}")
    print()

    # ---- 交叉熵的含义 ----
    print("交叉熵的直观含义 = -log(正确 token 的概率)：")
    demo_logits = torch.tensor([[math.log(0.9), math.log(0.1)],
                                [math.log(0.5), math.log(0.5)],
                                [math.log(0.1), math.log(0.9)]])
    demo_targets = torch.tensor([0, 0, 0])
    for k, (p, t) in enumerate(zip([0.9, 0.5, 0.1], demo_targets.tolist())):
        print(f"    给正确答案的概率 {p:.1f} -> loss = -log({p:.1f}) = "
              f"{my_cross_entropy(demo_logits[k:k + 1], demo_targets[k:k + 1]).item():.4f}")
    print("    给正确答案的概率 1.0 -> loss = 0（完美预测）")
    print()

    V = 10
    uniform = torch.zeros(4, V)                       # 所有权重为 0 => 完全均匀
    ce_uniform = F.cross_entropy(uniform, torch.zeros(4, dtype=torch.long))
    print("均匀分布（模型「什么都不会」）时：")
    print(f"    每个词概率 = 1/{V}，loss = -log(1/{V}) = ln({V}) = "
          f"{math.log(V):.6f}")
    print(f"    torch 实测                                  = {ce_uniform.item():.6f}")
    print()
    print("这条结论要牢牢记住：**训练刚开始时，loss 一定在 ln(vocab_size) 附近**。")
    print(f"本数据集的 vocab_size = {65}，所以基准是 ln(65) = {math.log(65):.4f}。")
    print("如果一开始 loss 就远高于这个数，说明初始化或数据出了问题；")
    print("如果远低于它，说明有信息泄漏（比如不小心把标签喂进了输入）。")


# ======================================================================
# 6. bigram 模型
# ======================================================================
class BigramLanguageModel(nn.Module):
    """最简语言模型：只用当前 token 预测下一个 token。

    它其实就是一张 (vocab_size, vocab_size) 的表：
        logits[b, t, :] = table[x[b, t], :]
    也就是一次 nn.Embedding 查表，没有任何上下文、没有任何注意力。

    为什么先写它？因为它是所有后续模型的「地板成绩」：
    如果加了注意力之后反而更差，那一定是实现有问题，而不是「注意力没用」。
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)                  # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """自回归采样：每次只生成一个 token，再把它接到输入后面继续。

        「自回归」= 自己生成的输出会成为下一步的输入。
        这也是生成看起来很慢的原因：要生成 T 个 token 就得跑 T 次前向
        （09 讲的 KV Cache 就是为了省掉这里的重复计算）。

        temperature 控制随机性：<1 更保守（接近贪心），>1 更发散，=1 就是原始分布。
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)     # 只关心最后一个位置
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)          # 按概率抽样
            idx = torch.cat([idx, nxt], dim=1)
        return idx


@torch.no_grad()
def estimate_bigram_val_loss(model: BigramLanguageModel, ds, device: torch.device,
                             batch_size: int, block_size: int, iters: int = 20) -> float:
    """在验证集上多抽几批取平均，减少单批的随机波动。"""
    model.eval()
    total = 0.0
    for _ in range(iters):
        xb, yb = ds.get_batch("val", batch_size, block_size, device)
        _, loss = model(xb, yb)
        total += loss.item()
    model.train()
    return total / iters


def demo_bigram(ds, device: torch.device, steps: int, batch_size: int,
                block_size: int) -> dict:
    print(banner("6-7. bigram 基线模型：训练前后对比"))
    set_seed(1337)

    model = BigramLanguageModel(ds.vocab_size).to(device)
    print(f"模型结构: nn.Embedding({ds.vocab_size}, {ds.vocab_size})  —— 就这一层")
    print(f"参数量  : {human_params(num_params(model))} "
          f"(= vocab x vocab = {ds.vocab_size}x{ds.vocab_size} = {ds.vocab_size ** 2:,})")
    print("注意：这张表是 vocab 的平方。英文 65 个字符时只有几千个参数，")
    print("      换成中文 5000 字词表就是 2500 万 —— 这就是 bigram 的天花板。")
    print()

    # 训练前的采样：权重是随机初始化的，查表结果纯粹是噪声
    start_ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    before = ds.tokenizer.decode(model.generate(start_ctx, 100)[0].tolist())
    print("训练前采样（完全是随机字符，看不出任何词语结构）：")
    print(f"    {before[:100]!r}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    print(f"开始训练 {steps} 步（batch_size={batch_size}, block_size={block_size}, "
          f"每步看 {batch_size * block_size} 个 token）...")
    losses: list[float] = []
    with Timer() as t:
        for step in range(steps):
            xb, yb = ds.get_batch("train", batch_size, block_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                print(f"    step {step:5d}   loss = {loss.item():.4f}")
    print(f"训练耗时 {t.elapsed:.1f}s（{steps / max(t.elapsed, 1e-9):.0f} 步/秒）")
    print()

    after = ds.tokenizer.decode(model.generate(start_ctx, 200, temperature=0.8)[0].tolist())
    print("训练后采样（开始出现英文单词和空格的节奏）：")
    print(f"    {after[:200]!r}")
    print()

    train_loss = sum(losses[-20:]) / len(losses[-20:])
    val_loss = estimate_bigram_val_loss(model, ds, device, batch_size, block_size)
    baseline = math.log(ds.vocab_size)

    print("数值对照：")
    print(f"    训练 loss（末 20 步均值）: {fmt_float(train_loss)}")
    print(f"    验证 loss（20 批均值）   : {fmt_float(val_loss)}")
    print(f"    随机猜测基准 ln(vocab)   : {fmt_float(baseline)}   = ln({ds.vocab_size})")
    print(f"    相对基准下降            : "
          f"{(1.0 - val_loss / baseline) * 100:.1f}%")
    print()
    print("怎么读这两个数：")
    print("    * loss 从 ln(V) 降到 3 以下，说明模型确实学到了「字符之间的转移规律」；")
    print("    * 但 bigram 只看当前一个字符，学到的只是「什么字母常跟在什么字母后」，")
    print("      所以它写不出连贯的句子 —— 这就是我们要上注意力的原因；")
    print("    * 验证 loss 略高于训练 loss 是正常的（模型没见过的文本更难预测），")
    print("      如果高出很多，说明过拟合了。")
    print()
    print("这个验证 loss 就是后续所有模型要超越的**基线**，请记进实验笔记。")
    return {"model": model, "train_loss": train_loss, "val_loss": val_loss,
            "baseline": baseline}


# ======================================================================
def main() -> int:
    setup_console()

    parser = argparse.ArgumentParser(description="01 讲：张量基础与 bigram 基线")
    parser.add_argument("--dataset", default="tinyshakespeare", help="data/raw 下的语料名")
    parser.add_argument("--quick", action="store_true", help="快速模式：只跑 300 步")
    parser.add_argument("--steps", type=int, default=3000, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    steps = 300 if args.quick else args.steps

    print(banner("01 讲：张量、softmax、交叉熵与 bigram 基线模型"))
    print(env_report())
    device = pick_device(args.device)
    print(f"使用设备: {describe_device(device)}")

    print(banner("1. 加载数据"))
    ds = load_dataset(args.dataset)
    print(ds.summary())
    print(f"运行配置: steps={steps}, batch_size={args.batch_size}, "
          f"block_size={args.block_size}, quick={args.quick}")

    demo_tokenizer(ds)
    demo_shapes(ds, device)
    demo_softmax_ce()
    demo_bigram(ds, device, steps=steps, batch_size=args.batch_size,
                block_size=args.block_size)

    print(banner("小结"))
    print("1. 语言模型的任务 = 预测下一个 token，标签就是文本自己右移一位，不用人工标注。")
    print("2. 数据在模型内的形状：(B,T) -> (B,T,C) -> (B,T,vocab) -> 概率。")
    print("3. softmax 把打分变成概率（手写版与 torch 完全一致）；")
    print("   交叉熵 = -log(正确 token 的概率)，均匀分布时恰好等于 ln(vocab_size)。")
    print("4. bigram 只是一张 vocab x vocab 的查表，上下文窗口为 1：")
    print("   loss 能明显低于 ln(V)，却写不出有语法的句子。")
    print("5. 下一讲先解决输入侧的问题：把 id 变成向量，并注入位置信息。")
    print()
    print("下一讲：src\\02_embeddings.py —— Embedding 与位置编码。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
