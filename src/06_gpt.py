r"""06 讲：组装完整 GPT —— 逐层追踪张量形状

前面几讲分别做好了零件：嵌入（02）、注意力（03/04）、Transformer Block（05）。
这一讲把它们拼成一台完整的机器，并且用「能看到内部」的方式检查它：

    0. 模型结构     —— 用 GPTConfig + GPT.from_config 建模型，看清整体骨架
    1. 前向传播追踪 —— 用 forward hook 记录每个子模块的输入输出形状，做成一张表
    2. 参数量账本   —— 子模块明细 + 与 estimate_params 公式逐项对照
    3. 权重共享     —— 用 data_ptr() 证明 lm_head 与 wte 真的是同一块内存
    4. 初始输出分布 —— 论证初始 loss 为什么必然 ≈ ln(vocab_size)
    5. 未训练生成   —— 只会输出随机字符（这是对的），顺便看温度的作用
    6. 设计选择对照 —— 固定其他条件，比较共享权重 vs 不共享

这一讲的核心方法是「先算账，再对数」：
    结构上每一块参数都能量化，凡是算出来的数和实际统计对不上，
    就说明你对那一块的理解还有漏洞。这正是第 2 节要演示的事情。

跑法：
    .\\.venv\\Scripts\\python.exe src\\06_gpt.py
    .\\.venv\\Scripts\\python.exe src\\06_gpt.py --preset micro --quick
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.attention import (  # noqa: E402
    CausalSelfAttention,      # Block 里的注意力实现，本讲第 1 节的形状追踪会看到它
    MultiHeadAttention,       # 好读版多头：与上面数学等价，04 讲做过数值对照
    manual_attention,         # 纯公式手写版：注意力本身的正确性由 03/04 讲负责
)
from common.config import GPTConfig, get_config, make_config  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.gpt import GPT, Block, LayerNorm, MLP, estimate_params  # noqa: E402
# 训练设施统一从 common.train 取。本讲第 6 节的对照实验自己写循环，
# 只为把中间量（每组每步的 loss、lr）打印出来；真要完整训练并存 checkpoint，
# 用 train(model, ds, config, train_cfg, path) 一行就够了（07 讲这么用）。
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
# 0. 模型结构
# ======================================================================
def build_model(preset: str, vocab_size: int, device: torch.device,
                seed: int = 1337) -> Tuple[GPT, GPTConfig]:
    """用工程统一入口建模：make_config（预设 + 词表） + GPT.from_config。

    为什么不直接写 GPT(**kwargs)？
        因为 checkpoint 里存的是 GPTConfig 的 dict。训练脚本、续训、推理
        全都走 from_config。课程脚本也用同一个入口，
        才能保证「怎么建的」和「怎么从 checkpoint 装回来的」完全一致。
    """
    config = make_config(preset, vocab_size=vocab_size)
    set_seed(seed)
    model = GPT.from_config(config).to(device)
    return model, config


def demo_architecture(model: GPT, config: GPTConfig, ds) -> None:
    print(banner("0. 模型结构：一个完整 GPT 由哪几块组成"))
    print(f"预设配置 {config.to_dict()}")
    total = num_params(model)
    print(f"\n参数量 {total:,}（{human_params(total)}）")

    print(f"\n{'配置项':<14} {'值':>10}  含义")
    print("-" * 78)
    rows = [
        ("vocab_size", config.vocab_size, "词表大小 = 嵌入矩阵的行数（字符级语料是 65）"),
        ("block_size", config.block_size, "上下文长度：一次最多能看多少个 token"),
        ("n_layer", config.n_layer, "Transformer Block 堆叠层数"),
        ("n_head", config.n_head, f"注意力头数，每头维度 = n_embd / n_head = {config.head_dim}"),
        ("n_embd", config.n_embd, "残差流宽度，也是所有中间张量的最后一维 C"),
        ("dropout", config.dropout, "dropout 比例，教学实验默认 0 便于复现"),
        ("bias", config.bias, "Linear 与 LayerNorm 是否带偏置"),
        ("tie_weights", model.tie_weights, "输出头是否与 token 嵌入共享同一个矩阵"),
    ]
    for name, value, note in rows:
        print(f"{name:<14} {str(value):>10}  {note}")

    print("\n骨架（对照 common/gpt.py 的模块定义读）：")
    skeleton = [
        ("wte", f"nn.Embedding({config.vocab_size}, {config.n_embd})", "token id -> 向量"),
        ("wpe", f"nn.Embedding({config.block_size}, {config.n_embd})", "位置 id -> 向量"),
        ("blocks", f"{config.n_layer} × Block",
         "每个 Block = 注意力子层 + MLP 子层，各带一层残差"),
        ("ln_f", f"LayerNorm({config.n_embd})", "最终归一化（Pre-LN 的必需收尾）"),
        ("lm_head", f"nn.Linear({config.n_embd}, {config.vocab_size})",
         "把向量投回词表打分，bias=False"),
    ]
    print(f"    {'模块':<9}{'定义':<34}作用")
    print("    " + "-" * 76)
    for name, definition, note in skeleton:
        print(f"    {name:<9}{definition:<34}{note}")
    print()
    print("forward 的返回值永远是 4 元组：")
    print("    logits, loss, weights, cache = model(idx, targets)")
    print("        logits  (B, T, vocab)                 每个位置对下一个 token 的打分")
    print("        loss    标量或 None                   传了 targets 才有")
    print("        weights (n_layer, B, nh, T, T) 或 None 注意力权重，开 return_weights 才有")
    print("        cache   长度 n_layer 的列表            每层的 K/V，09 讲增量推理用")
    print("    注意 logits 是「未归一化」的打分：想变概率要自己 softmax，")
    print("    算 loss 时 F.cross_entropy 内部会做 log_softmax，所以不用手动 softmax。")


# ======================================================================
# 1. 前向传播追踪
# ======================================================================
def trace_forward(model: GPT, ids: torch.Tensor,
                  max_depth: int = 1) -> List[Tuple[str, str, str, int]]:
    """用 register_forward_hook 记录每个子模块的输入输出形状与参数量。

    hook 是 PyTorch 的「观察者」机制：不用改模型代码，就能在每次前向时
    拿到中间张量。调试形状问题时这是最省事的办法。

    返回 [(缩进名, 输入形状, 输出形状, 参数量), ...]，顺序就是实际执行顺序。
    """
    records: List[Tuple[str, str, str, int]] = []

    def make_hook(label: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            # 有些模块（Block、CausalSelfAttention）返回元组，形状看第一个元素
            out = output[0] if isinstance(output, tuple) else output
            in_shape = tuple(inputs[0].shape) if inputs and hasattr(inputs[0], "shape") else ()
            out_shape = tuple(out.shape) if hasattr(out, "shape") else ()
            records.append((label, str(in_shape), str(out_shape), num_params(module)))
        return hook

    handles: List[Any] = []

    def register(module: nn.Module, label: str, depth: int) -> None:
        handles.append(module.register_forward_hook(make_hook(label)))
        if depth >= max_depth:
            return
        for child_name, child in module.named_children():
            if isinstance(child, nn.ModuleList):
                continue                    # blocks 那一层单独展开，避免刷屏
            register(child, f"{label}.{child_name}", depth + 1)

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList):
            handles.append(child.register_forward_hook(make_hook("blocks")))
            for i, block in enumerate(child):
                register(block, f"blocks[{i}]", 0)      # 每个 Block 再展开一层
            continue
        handles.append(child.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()                            # 关掉 dropout，形状与数值才可复现
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    return records


def demo_forward_trace(model: GPT, ds, device: torch.device) -> None:
    print(banner("1. 前向传播追踪：数据形状是怎么流动的"))
    set_seed(0)
    B, T = 2, 16
    ids = torch.randint(0, ds.vocab_size, (B, T), device=device)
    print(f"输入：{B} 条序列 × {T} 个 token，词表 {ds.vocab_size}")
    print(f"idx 形状 {tuple(ids.shape)}，dtype {ids.dtype}（是整数 id，不是浮点）\n")

    records = trace_forward(model, ids, max_depth=1)
    print(f"{'模块':<24} {'输入形状':<26} {'输出形状':<26} {'参数量':>10}")
    print("-" * 92)
    for label, in_shape, out_shape, params in records:
        p = human_params(params) if params else "-"
        print(f"{label:<24} {in_shape:<26} {out_shape:<26} {p:>10}")
    print("-" * 92)

    C, V = model.n_embd, ds.vocab_size
    hs = C // model.n_head
    print("\n把这条链路连起来看：")
    print(f"    ({B}, {T}){' ' * 12}整数 token id")
    print(f" -> ({B}, {T}, {C}){' ' * 8}wte + wpe：查表得到向量，残差流从这里开始")
    print(f" -> ({B}, {T}, {C}){' ' * 8}{model.n_layer} 层 Block，"
          f"**每层的输出形状都和输入完全一样**")
    print(f" -> ({B}, {T}, {C}){' ' * 8}ln_f：只改数值，不改形状")
    print(f" -> ({B}, {T}, {V}){' ' * 7}lm_head：最后一维从 C 变成词表大小")
    print()
    print(f"注意力内部还会临时变成 (B, n_head, T, head_size) = ({B}, {model.n_head}, {T}, {hs})：")
    print("    把 C 个通道切成 n_head 份，让每个头独立做 T × T 的相关性打分，")
    print("    算完再 transpose + view 拼回 (B, T, C)。04 讲已经验证过这条 reshape。")
    print("    也就是说：形状的「扰动」只发生在模块内部，模块之间永远是 (B, T, C)。")

    print("\n最重要的一条观察：残差流宽度全程不变。")
    print(f"    从 wte 的输出到 ln_f 的输出，形状始终是 (B, T, C)，C={C} 一次都没变过。")
    print("    这个设计带来三个很实用的好处：")
    print("      1. 层可以随意增删 —— 第 i 层的输出必然能喂给第 i+1 层，接口天然统一；")
    print("      2. 残差相加不需要任何投影对齐（若每层改宽度，就得多一个线性层）；")
    print("      3. 中间层可以被探针直接读取，也可以做层间拼接、")
    print("         甚至像 U-Net 那样把浅层特征跳接到深层 —— 形状一致才有这些操作空间。")
    print("    全流程只有两处改变形状：最开始的查表（整数 -> 向量）")
    print("    和最后的输出头（向量 -> 词表打分）。这也是 Transformer 结构上最优雅的一点。")


# ======================================================================
# 2. 参数量账本
# ======================================================================
def demo_param_ledger(model: GPT, config: GPTConfig) -> None:
    print(banner("2. 参数量账本：子模块明细 vs 解析公式"))
    print("先看子模块明细（module_table 直接统计每个子模块的实际参数量）：")
    print(module_table(model, top=8))

    print("\n再按公式拆一遍（estimate_params 不用建模型就能算出这些数）：")
    est = estimate_params(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        tie_weights=model.tie_weights,
    )
    C = config.n_embd
    per_layer_rows = [
        ("注意力 (c_attn + c_proj)", 4 * C * C, "4C²，注意力层故意不带 bias"),
        ("MLP (fc_in + fc_out)", 8 * C * C + 5 * C, "8C² + 5C（4C 与 C 两处 bias）"),
        ("LayerNorm (ln_1 + ln_2)", 4 * C, "4C（两个 LN，各 weight + bias）"),
    ]
    print(f"\n{'单项公式':<30} {'每层':>10}  说明")
    print("-" * 78)
    for name, value, note in per_layer_rows:
        print(f"{name:<30} {value:>10,}  {note}")
    print("-" * 78)
    print(f"{'每层合计':<30} {est['per_layer']:>10,}")

    ledger = [
        ("token 嵌入 wte", config.vocab_size * C, est["token_embedding"],
         "vocab × C"),
        ("位置嵌入 wpe", config.block_size * C, est["position_embedding"],
         "block_size × C"),
        ("所有 Transformer 层", est["per_layer"] * config.n_layer, est["all_layers"],
         f"每层 {est['per_layer']:,} × {config.n_layer} 层"),
        ("最终归一化 ln_f", 2 * C, est["final_norm"], "2C"),
        ("输出头 lm_head", 0 if model.tie_weights else config.vocab_size * C,
         est["lm_head"], "共享权重时为 0"),
    ]
    print(f"\n{'部分':<24} {'手算公式':>12} {'estimate_params':>17} {'一致':>6}  公式")
    print("-" * 88)
    all_match = True
    for name, mine, theirs, formula in ledger:
        same = mine == theirs
        all_match = all_match and same
        print(f"{name:<24} {mine:>12,} {theirs:>17,} {'是' if same else '否':>6}  {formula}")
    print("-" * 88)
    formula_total = sum(v for _, v, _, _ in ledger)
    actual_total = num_params(model)
    print(f"{'公式合计':<24} {formula_total:>12,}")
    print(f"{'estimate_params 合计':<24} {est['total']:>12,}")
    print(f"{'实际统计 num_params':<24} {actual_total:>12,}")

    print()
    if actual_total == est["total"] and formula_total == actual_total and all_match:
        print("三者完全吻合（差值 0）—— 每一块参数都解释清楚了：")
        print(f"    嵌入 {est['token_embedding'] + est['position_embedding']:,}"
              f" + 层 {est['all_layers']:,}"
              f" + 最终归一化 {est['final_norm']:,}"
              f" + 输出头 {est['lm_head']:,}"
              f" = {est['total']:,}")
    else:
        print(f"有差值 {actual_total - est['total']:,}，公式里漏了或多算了东西。")
        print("对不平时按这三处排查，它们是最常见的坑：")
        print("    1. bias 有没有算进去（Linear 的 bias、LayerNorm 的 bias 是两笔账）")
        print("    2. LayerNorm 的 weight/bias 容易被忽略（每个 LN 是 2C）")
        print("    3. 输出头是否共享权重（共享时它一个参数都不额外占）")

    emb = est["token_embedding"] + est["position_embedding"]
    print(f"\n占比视角（总参数 {est['total']:,}）：")
    for name, value in [("嵌入层 (wte + wpe)", emb),
                        ("Transformer 层", est["all_layers"]),
                        ("输出头 lm_head", est["lm_head"]),
                        ("最终归一化", est["final_norm"])]:
        print(f"    {name:<22} {value:>12,}  {value / est['total']:>7.1%}")
    attn_all = est["attn_per_layer"] * config.n_layer
    mlp_all = est["mlp_per_layer"] * config.n_layer
    print(f"\n    Transformer 层内部：注意力 {attn_all:,}"
          f"（{attn_all / est['all_layers']:.1%}） vs MLP {mlp_all:,}"
          f"（{mlp_all / est['all_layers']:.1%}）")
    print("    这个约 1:2 的比例正是 05 讲的结论：MLP 才是参数量的大头，")
    print("    注意力负责横向（位置之间）的信息搬运，MLP 负责纵向（位置内部）的非线性加工。")
    print("\n账本对得上，就说明你对 GPT 每一块参数的理解没有漏洞。")
    print("    以后看任何一个新模型（LLaMA、Qwen……），都可以先用同样的方法算一遍账。")


# ======================================================================
# 3. 权重共享
# ======================================================================
def demo_weight_tying(ds, device: torch.device, preset: str = "micro") -> None:
    print(banner("3. 权重共享：lm_head 和 wte 是不是同一块内存"))
    kwargs: Dict[str, Any] = dict(get_config(preset))
    kwargs["vocab_size"] = ds.vocab_size

    set_seed(7)
    tied = GPT(**kwargs, tie_weights=True).to(device)
    set_seed(7)
    untied = GPT(**kwargs, tie_weights=False).to(device)

    n_tied, n_untied = num_params(tied), num_params(untied)
    expected = ds.vocab_size * tied.n_embd
    print(f"词表 {ds.vocab_size}，n_embd={tied.n_embd}，其余配置完全相同，"
          f"唯一变量是 tie_weights。\n")
    print(f"{'配置':<20} {'参数量':>12}")
    print("-" * 34)
    print(f"{'tie_weights=True':<20} {n_tied:>12,}")
    print(f"{'tie_weights=False':<20} {n_untied:>12,}")
    print("-" * 34)
    print(f"{'差值':<20} {n_untied - n_tied:>12,}")
    print(f"vocab × n_embd = {ds.vocab_size} × {tied.n_embd} = {expected:,}"
          f"    差值相等: {n_untied - n_tied == expected}")

    print("\n再看内存地址（data_ptr() 返回张量底层存储的地址）：")
    p_lm = tied.lm_head.weight.data_ptr()
    p_wte = tied.wte.weight.data_ptr()
    print(f"    tied.lm_head.weight.data_ptr()   = {p_lm}")
    print(f"    tied.wte.weight.data_ptr()       = {p_wte}")
    print(f"    是同一块内存吗: {p_lm == p_wte}")
    u_lm = untied.lm_head.weight.data_ptr()
    u_wte = untied.wte.weight.data_ptr()
    print(f"    untied 版本: {u_lm} vs {u_wte} -> 同一块内存吗: {u_lm == u_wte}")

    print("\n联动验证：只在 wte 上写一个值，然后去读 lm_head 的同一个位置。")
    with torch.no_grad():
        before = tied.lm_head.weight[3, 0].item()
        tied.wte.weight[3, 0] = 12.345
        after = tied.lm_head.weight[3, 0].item()
    print(f"    lm_head.weight[3, 0] 改动前 : {before:+.4f}")
    print("    执行 tied.wte.weight[3, 0] = 12.345")
    print(f"    lm_head.weight[3, 0] 改动后 : {after:+.4f}")
    print(f"    确认联动: {abs(after - 12.345) < 1e-6}"
          "  —— 它们就是同一个 nn.Parameter 对象")
    print("    （实现上就是 common/gpt.py 里那一行：self.lm_head.weight = self.wte.weight）")

    print("\n为什么可以共享？")
    print("    输入嵌入在做「token id -> 向量」，输出头在做「向量 -> 每个 token 的分数」。")
    print("    前者是取出矩阵的某一行，后者是与矩阵每一行做点积 ——")
    print("    本质上是同一个映射的正反两个方向，用同一个矩阵完全说得通。")
    print("    收益是省掉 vocab × C 个参数，词表越大越可观：")
    print(f"    本例词表只有 {ds.vocab_size}，就省了 {expected:,} 个"
          f"（占不共享版总参数的 {expected / n_untied:.1%}）；")
    print("    换成中文 BPE 词表（几万），这一项就是几百万到上千万参数。")
    print("    GPT-2 全系以及绝大多数开源 LLM 都这么做。代价与风险见第 6 节的实测。")


# ======================================================================
# 4. 初始输出分布
# ======================================================================
def demo_initial_output(model: GPT, ds, device: torch.device,
                        B: int = 4, T: int = 16, trials: int = 20) -> None:
    print(banner("4. 初始输出分布：为什么 loss 必然 ≈ ln(vocab_size)"))
    V = ds.vocab_size
    uniform_p = 1.0 / V
    uniform_h = math.log(V)
    model.eval()

    # 单个 batch 的 loss 会有几毛钱的抖动，所以既要看一次，也要看多次的平均
    set_seed(0)
    ids = torch.randint(0, V, (B, T), device=device)
    with torch.no_grad():
        logits, loss, _, _ = model(ids, ids)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-12).log()).sum(-1).mean().item()

    print(f"输入 {B} × {T} 个随机 token，前向一次（此时模型还没训练过）。\n")
    print(f"{'指标':<24} {'实测':>12} {'均匀分布参考':>14}")
    print("-" * 54)
    print(f"{'logits 均值':<24} {logits.mean().item():>+12.5f} {'0':>14}")
    print(f"{'logits 标准差':<24} {logits.std().item():>12.5f} {'-':>14}")
    print(f"{'最大概率的平均值':<24} {probs.max(dim=-1).values.mean().item():>12.5f} "
          f"{uniform_p:>14.5f}")
    print(f"{'最大概率的最小值':<24} {probs.max(dim=-1).values.min().item():>12.5f} "
          f"{uniform_p:>14.5f}")
    print(f"{'平均熵（nats）':<24} {entropy:>12.5f} {uniform_h:>14.5f}")
    print(f"{'交叉熵 loss':<24} {loss.item():>12.5f} {uniform_h:>14.5f}")
    print("-" * 54)

    # 多抽几个 batch 求平均：这才是有意义的对照口径
    with torch.no_grad():
        draws = []
        for _ in range(trials):
            xb, yb = ds.get_batch("val", B, T, device)
            draws.append(model(xb, yb)[1].item())
    mean_loss = sum(draws) / len(draws)
    std_loss = (sum((d - mean_loss) ** 2 for d in draws) / len(draws)) ** 0.5
    print(f"\n单次前向的 loss 会抖：上面这一次是 {loss.item():.4f}，"
          f"与 ln(V) 差 {loss.item() - uniform_h:+.4f}。")
    print(f"在 {trials} 个随机 batch 上求平均才是稳定的对照口径：")
    print(f"    平均 loss = {mean_loss:.4f} ± {std_loss:.4f}（标准差），"
          f"ln(V) = {uniform_h:.4f}，偏差 {mean_loss - uniform_h:+.4f}")
    print(f"    均值与 ln(V) 的差只有 {abs(mean_loss - uniform_h):.4f} —— 这才是"
          "「初始 loss ≈ ln(V)」的准确含义。")
    print("    为什么会抖？因为 logits 并不是严格 0，而是方差很小的随机数，")
    print("    样本少的时候哪几个 token 恰好被高估一点，就会把这次 loss 拉开。")

    zero_logits = torch.zeros(2, 2, V, device=device)
    ce_uniform = F.cross_entropy(zero_logits.view(-1, V),
                                 torch.zeros(4, dtype=torch.long, device=device))
    print(f"\n极限情形直接验算：全 0 的 logits 做交叉熵 = {ce_uniform.item():.5f}，"
          f"ln({V}) = {uniform_h:.5f}")
    print("    两者严格相等。因为均匀分布下每个 token 的预测概率都是 1/V，")
    print("    交叉熵 = -log(1/V) = ln(V)。初始化时 logits 越接近 0，loss 就越贴近 ln(V)。")

    print("\n为什么初始化时输出会接近均匀？")
    print("    lm_head 的权重是均值 0、标准差 0.02 的随机数，残差流的数值也都很小，")
    print("    所以初始 logits 几乎全在 0 附近（看上面 logits 的均值与标准差），")
    print("    softmax 之后自然就是接近均匀的分布 —— 模型对下一个 token「毫无主见」。")
    print("    换句话说：一个刚初始化的 GPT，本质上就是在随机猜下一个字符。")

    print("\n这是一个极好用的自检指标：")
    print(f"    新写的模型第一轮前向，loss 应该落在 ln(V) = {uniform_h:.2f} 附近"
          f"（本例 {mean_loss:.2f}）。")
    print("    明显偏大（比如 8、10）：logits 初始化太大，或者缩放系数没做对；")
    print("    明显偏小（比如 1、2）：高度怀疑标签泄漏 —— 例如 y 忘了右移一位，")
    print("    模型在直接抄答案。这种「好成绩」是假的，训练下去会立刻打回原形。")
    print("    注意判据是「附近」而不是「精确等于」：差几个百分点很正常，")
    print("    差一大截才说明有问题。")
    print(f"\n所以训练的目标很明确：把这个 {uniform_h:.2f} 一步步压下去。")
    print("参考量级：字符级英文语料上，训练充分的 GPT 能把 val loss 压到 1.5 左右。")
    print(f"（下一讲 07 会真的训练一遍，看到 loss 从 {uniform_h:.2f} 一路下降。）")


# ======================================================================
# 5. 未训练模型的生成
# ======================================================================
def demo_untrained_generation(model: GPT, ds, device: torch.device) -> None:
    print(banner("5. 未经训练的模型能生成什么"))
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    start_char = ds.tokenizer.decode(start[0].tolist())
    print(f"起始 token {tuple(start.shape)}，id 0 -> 字符 {start_char!r}"
          f"（id 0 就是词表里排序最靠前的那个字符）\n")

    out = model.generate(start.clone(), max_new_tokens=120, temperature=1.0, seed=0)
    text = ds.tokenizer.decode(out[0].tolist())
    print("temperature=1.0 生成 120 个 token：")
    print(f"    {text!r}")
    print()
    print("看到的是随机字符 —— 这是**正确**的，不是 bug。")
    print("    模型此刻的权重还是随机初始化的，它学到的东西是「零」，")
    print("    输出自然接近从词表均匀采样。生成能力的唯一来源是训练。")
    print("    等 07 讲训上几百步再回来跑同样的代码，就会开始出现英文单词。")

    print("\n温度参数做了什么？它把 logits 除以 T 再 softmax：")
    print("    T < 1 分布更尖锐（高分 token 更容易被选中，保守、易重复）")
    print("    T > 1 分布更平坦（低分 token 也有机会，随机、容易胡说）\n")
    with torch.no_grad():
        logits, _, _, _ = model(start)
    last_logits = logits[0, -1]
    print(f"{'temperature':>12} {'平均熵':>10} {'最大概率':>10}  生成样例")
    print("-" * 84)
    for temp in [0.5, 1.0, 1.5]:
        p = F.softmax(last_logits / temp, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum().item()
        sample = model.generate(start.clone(), max_new_tokens=48, temperature=temp, seed=0)
        snippet = ds.tokenizer.decode(sample[0].tolist())[:46]
        print(f"{temp:>12.1f} {ent:>10.4f} {p.max().item():>10.4f}  {snippet!r}")
    print("-" * 84)
    print(f"（对照：均匀分布的熵是 ln({ds.vocab_size}) = {math.log(ds.vocab_size):.4f}，"
          f"最大概率是 {1 / ds.vocab_size:.4f}）")
    print("\n读这张表：温度越低，熵越小、最大概率越接近 1，采样越保守；")
    print("    温度越高，熵越接近 ln(V)，分布越接近均匀，文字越乱。")
    print("    但请注意：温度并不改变模型「知道什么」，只改变采样的激进程度。")
    print("    模型没训练好时，怎么调温度都是噪声 —— 它调不出不存在的知识。")
    print("    temperature=1.0 表示「严格按模型自己学到的分布采样」，是默认值；")
    print("    实际使用中更常见的是 0.7~0.9 再配合 top_k / top_p（09 讲会实测）。")


# ======================================================================
# 6. 设计选择对照
# ======================================================================
def train_variant(ds, device: torch.device, preset: str, steps: int,
                  tie_weights: bool, tag: str, seed: int = 4321,
                  quiet: bool = False) -> Dict[str, Any]:
    """用 common.train 的统一设施训练一个变体，返回指标字典。

    纪律：同一个种子、同一套超参数（含学习率调度与梯度裁剪），
    唯一变量是 tie_weights —— 否则差异可能来自随机性而不是这个设计选择。
    """
    kwargs: Dict[str, Any] = dict(get_config(preset))
    kwargs["vocab_size"] = ds.vocab_size
    set_seed(seed)
    model = GPT(**kwargs, tie_weights=tie_weights).to(device)

    train_cfg = TrainConfig(
        batch_size=32, block_size=model.block_size, max_steps=steps,
        learning_rate=3e-3, min_lr=3e-4, warmup_steps=max(1, steps // 10),
        grad_clip=1.0, eval_interval=10 ** 9, log_interval=10 ** 9,
        eval_batches=20, seed=seed, device=str(device),
    )
    opt = build_optimizer(model, train_cfg)

    with Timer() as timer:
        for step in range(steps):
            # 用 common.train.get_lr 的调度：warmup + 余弦退火，两组完全一致
            lr = get_lr(step, train_cfg.warmup_steps, steps,
                        train_cfg.learning_rate, train_cfg.min_lr)
            for group in opt.param_groups:
                group["lr"] = lr
            model.train()
            xb, yb = ds.get_batch("train", train_cfg.batch_size, train_cfg.block_size, device)
            _, loss, _, _ = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            if not quiet and step % max(1, steps // 4) == 0:
                print(f"      [{tag}] step {step:3d}  loss {loss.item():.4f}  lr {lr:.2e}")
        final = estimate_loss(model, ds, train_cfg, splits=("train", "val"))

    return {
        "tag": tag,
        "params": num_params(model),
        "train_loss": final["train"],
        "val_loss": final["val"],
        "elapsed": timer.elapsed,
        "model": model,
    }


def demo_design_choices(ds, device: torch.device, steps: int = 150,
                        preset: str = "micro") -> None:
    print(banner("6. 设计选择对照：共享权重 vs 不共享"))
    print("固定数据集与全部超参数（lr=3e-3 + warmup + 余弦退火，grad_clip=1.0），")
    print(f"两组使用同一个随机种子、各训练 {steps} 步，唯一变量是 tie_weights。\n")

    rows = [train_variant(ds, device, preset, steps, True, "共享权重"),
            train_variant(ds, device, preset, steps, False, "不共享")]

    print(f"\n{'配置':<12} {'参数量':>10} {'训练 loss':>11} {'验证 loss':>11} "
          f"{'相对基线':>10} {'耗时':>8}")
    print("-" * 68)
    base = rows[0]["val_loss"]
    for i, r in enumerate(rows):
        note = "（基准）" if i == 0 else f"{r['val_loss'] / base - 1:+.1%}"
        print(f"{r['tag']:<12} {r['params']:>10,} {r['train_loss']:>11.4f} "
              f"{r['val_loss']:>11.4f} {note:>10} {r['elapsed']:>7.1f}s")
    print("-" * 68)
    n_embd = rows[0]["model"].n_embd
    print(f"参数量差距 {rows[1]['params'] - rows[0]['params']:,} 个"
          f"（= vocab × n_embd = {ds.vocab_size} × {n_embd}"
          f" = {ds.vocab_size * n_embd:,}）")

    print("\n怎么读这张表：")
    print("  1. 参数量：不共享那组精确地多出 vocab × n_embd 个参数，")
    print("     与第 3 节的账完全一致。")
    print("  2. 验证 loss：多出来的参数**未必**换来更好的验证 loss。")
    print("     共享权重强制输入嵌入与输出头使用同一个矩阵，等于给模型加了一层约束；")
    print("     在中小规模数据上，这种约束常常刚好起到正则化的作用。")
    print("  3. 把这条经验推而广之：参数量与效果不是一回事。")
    print("     能省的参数省下来，拿去多加几层或加宽 n_embd，通常更划算。")
    print("\n提醒：这个规模下的对照只能看趋势。想得出可靠结论，")
    print("      请把 --steps 调到 1000+，并用 3 个以上种子重复，比较均值与方差；")
    print("      单次几百步的差距很可能被随机性淹没。")


# ======================================================================
def main() -> int:
    setup_console()                      # 第一行永远是它：把控制台切成 UTF-8
    parser = argparse.ArgumentParser(description="06 讲：组装完整 GPT 并逐层追踪形状")
    parser.add_argument("--dataset", default="tinyshakespeare", help="data/raw 下的语料名")
    parser.add_argument("--preset", default="tiny",
                        choices=["micro", "tiny", "mini", "small"], help="模型规模预设")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=150, help="第 6 节对照组每组训练步数")
    parser.add_argument("--quick", action="store_true", help="快速模式：步数减半，几十秒跑完")
    args = parser.parse_args()

    print(banner("06 讲：组装完整 GPT —— 逐层追踪张量形状"))
    print(env_report())
    print()
    device = pick_device(args.device)
    print(f"本次运行使用设备: {describe_device(device)}")

    ds = load_dataset(args.dataset)
    print()
    print(ds.summary())

    model, config = build_model(args.preset, ds.vocab_size, device)
    print(f"\n已建立模型：预设 {args.preset}，参数量 {num_params(model):,}"
          f"（{human_params(num_params(model))}）")

    demo_architecture(model, config, ds)
    demo_forward_trace(model, ds, device)
    demo_param_ledger(model, config)
    demo_weight_tying(ds, device, preset=args.preset)
    demo_initial_output(model, ds, device)
    demo_untrained_generation(model, ds, device)

    steps = 50 if args.quick else args.steps
    demo_design_choices(ds, device, steps=steps, preset=args.preset)

    print(banner("小结"))
    print("1. 完整 GPT = 嵌入（wte + wpe） + N × Block + 最终归一化 ln_f + 输出头 lm_head；")
    print("   用 GPTConfig + GPT.from_config 建模，与 checkpoint 的存取口径完全一致。")
    print("2. 形状链路 (B,T) -> (B,T,C) -> 各层不变 -> (B,T,vocab)；")
    print("   残差流宽度全程不变，所以层可以随意增删、中间层能被直接读取或跳接。")
    print("3. 参数量账本对得上：嵌入 + 每层(4C² + 8C²+5C + 4C) + ln_f + 输出头，")
    print("   手算公式、estimate_params、实际统计三者完全吻合 ——")
    print("   账能算平，说明结构理解没有漏洞。")
    print("4. 权重共享省下 vocab × n_embd 个参数，data_ptr() 证明它就是同一块内存，")
    print("   改一个另一个立刻联动；在中小模型上通常还能略微提升效果。")
    print("5. 初始化时 logits ≈ 0、输出接近均匀分布，所以在多个 batch 上平均之后，")
    print("   初始 loss ≈ ln(vocab_size)；这是个极好用的自检指标：")
    print("   偏大说明初始化有问题，偏小要怀疑标签泄漏。")
    print("6. 未训练的模型只能输出随机字符 —— 生成能力全部来自训练；")
    print("   温度只改变采样的激进程度，变不出模型没学到的知识。")
    print()
    print("下一讲：src\\07_training.py —— 真正开始训练。")
    print("你会看到 loss 从 ln(65) ≈ 4.17 一路降到 1.5 左右，")
    print("并学会学习率调度、梯度裁剪、验证集评估和 checkpoint 的保存与恢复。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
