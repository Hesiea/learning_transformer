r"""07 讲：训练一个真正的 GPT

前面六讲把模型搭好了，这一讲让它学会说话。训练的核心其实只有五行：

    logits, loss, _, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

但「训得好」需要一整套工程细节。本讲按顺序讲清楚七件事：

    1. 训练成本预估   —— 先用 estimate_params 把账算清楚：多少参数、吃多少 token
    2. 学习率调度     —— warmup 升温 + 余弦退火 + min_lr 兜底，各自解决什么问题
    3. AdamW 参数分组 —— 为什么 bias 与 LayerNorm 参数不做 weight decay
    4. 训练循环       —— 直接复用 common.train.train，一行循环都不自己写
    5. 训练曲线       —— history 存成 json，并画 train/val loss 与学习率两条曲线
    6. 断点续训       —— 从 checkpoint 恢复模型、优化器状态、已训练步数与历史最佳
    7. 结果解读       —— 和随机猜测 ln(V) 对比，看困惑度、过拟合差距与吞吐

打印顺序是 1 -> 2 -> 3 -> 6 -> 4 -> 5 -> 7：
断点续训（第 6 节）必须排在训练循环（第 4 节）之前，因为它决定「从第几步开始跑」。

跑法（CPU 上几分钟）：
    .\.venv\Scripts\python.exe src\07_training.py                        # 默认 1200 步
    .\.venv\Scripts\python.exe src\07_training.py --quick                # 200 步，几分钟跑完
    .\.venv\Scripts\python.exe src\07_training.py --steps 150 --quick --no-plot
    .\.venv\Scripts\python.exe src\07_training.py --steps 200 --quick --no-plot --resume
    .\.venv\Scripts\python.exe src\07_training.py --dataset zh_poetry    # 换语料

产物：
    out/<run_name>.pt               最终 checkpoint
    out/<run_name>_best.pt          验证 loss 最低的 checkpoint（推理就用它）
    out/<run_name>_history.json     训练曲线数据（step / train_loss / val_loss / lr）
    out/figures/07_training_curves.png   训练曲线图
"""

from __future__ import annotations

import argparse
import math
import sys
import unicodedata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import GPTConfig, make_config  # noqa: E402
from common.data import TokenDataset, load_dataset  # noqa: E402
from common.gpt import GPT, estimate_params  # noqa: E402
from common.train import (  # noqa: E402
    TrainConfig,
    build_optimizer,
    get_lr,
    load_model_from_checkpoint,
    save_history,
    train,
)
from common.utils import (  # noqa: E402
    Timer,
    banner,
    describe_device,
    env_report,
    human_params,
    human_time,
    num_params,
    pick_device,
    set_seed,
    setup_console,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "out"
FIG_PATH = OUT_DIR / "figures" / "07_training_curves.png"


# ----------------------------------------------------------------------
# 表格对齐小工具：中英文混排时 str.ljust 按「字符数」补齐，一个汉字占两列，
# 表格就会歪。这里按终端显示宽度补齐，输出才对得整齐。
# ----------------------------------------------------------------------
def disp_width(text: str) -> int:
    """按终端显示宽度数：全角（W/F）算 2 列，其余算 1 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int, align: str = "l") -> str:
    fill = " " * max(0, width - disp_width(text))
    if align == "r":
        return fill + text
    return text + fill


# ======================================================================
# 1. 训练成本预估
# ======================================================================
def demo_cost_estimate(config: GPTConfig, train_cfg: TrainConfig,
                       ds: TokenDataset, actual_params: int) -> None:
    """先算账再开跑：参数量、每步 token 数、总 token 数、6ND 的 FLOPs 量级。

    estimate_params 是纯公式（不用真的建模型），这里顺手拿它和真实参数量对一次数，
    符合本工程的老规矩：结论要用数值对照证明，而不是只写在注释里。
    """
    print(banner("1. 训练成本预估：先算账，再开跑"))
    est = estimate_params(config.vocab_size, config.block_size, config.n_layer,
                          config.n_embd, tie_weights=True)
    tokens_per_step = train_cfg.batch_size * train_cfg.block_size
    total_tokens = tokens_per_step * train_cfg.max_steps
    train_tokens = len(ds.train)
    epochs = total_tokens / max(train_tokens, 1)

    print(f"模型规格：n_layer={config.n_layer}  n_head={config.n_head}  "
          f"n_embd={config.n_embd}  block_size={config.block_size}  "
          f"vocab_size={config.vocab_size}  dropout={config.dropout}")
    print()
    print("参数量拆分（公式估算 vs 真的建一个模型数出来）：")
    est_rows = [
        ("token 嵌入", est["token_embedding"], ""),
        ("位置嵌入", est["position_embedding"], ""),
        ("每层：注意力（4C²）", est["attn_per_layer"], ""),
        ("每层：MLP（8C² + 5C）", est["mlp_per_layer"], ""),
        ("每层：LayerNorm（4C）", est["ln_per_layer"], ""),
        ("每层小计", est["per_layer"], f"x {config.n_layer} 层"),
        ("所有层合计", est["all_layers"], ""),
        ("最终 LayerNorm", est["final_norm"], ""),
        ("语言模型头", est["lm_head"], "与 token 嵌入共享权重，所以不计入额外参数"),
    ]
    for label, value, extra in est_rows:
        print("    " + pad(label, 28) + pad(f"{value:,}", 12, "r")
              + ("   " + extra if extra else ""))
    print("    " + "-" * 46)
    print("    " + pad("公式估算合计", 28) + pad(f"{est['total']:,}", 12, "r")
          + f"   ({human_params(est['total'])})")
    print("    " + pad("实测参数量", 28) + pad(f"{actual_params:,}", 12, "r")
          + f"   ({human_params(actual_params)})    差值 {actual_params - est['total']:+d}")
    print("    结论：MLP 约占 2/3 的参数，注意力约 1/3，嵌入层可以忽略 —— 这就是「参数都在哪」。")
    print()
    print("本次训练要吃多少数据：")
    print(f"    每步 token 数 = batch_size({train_cfg.batch_size}) x "
          f"block_size({train_cfg.block_size}) = {tokens_per_step:,}")
    print(f"    总步数        = {train_cfg.max_steps:,}")
    print(f"    总 token 数   = {total_tokens:,}")
    print(f"    训练集 token  = {train_tokens:,}")
    print(f"    => 相当于把训练集过了 {epochs:.2f} 遍（epoch）")
    print()
    flops = 6 * est["total"] * total_tokens
    print("粗略计算量（Chinchilla 常用的 6ND 口径）：")
    print(f"    6 x 参数量 x token 数 = 6 x {est['total']:,} x {total_tokens:,}"
          f" ≈ {flops:.3e} FLOPs（约 {flops / 1e12:.3f} TFLOPs）")
    print("    系数 6 的来历：前向约 2ND，反向约 4ND（反向是前向的两倍），加起来 6ND。")
    print()
    print("「过了几遍」这个数字要留意：")
    print("    不到 1 遍：模型还没看完全部数据，加步数通常还有收益。")
    print("    大于 1 遍：同一批文本会被反复看到，要盯着 val loss 会不会先降后升。")
    print("    实践中的大模型训练普遍只过 1 遍左右 —— 语料规模远大于模型规模。")


# ======================================================================
# 2. 学习率调度
# ======================================================================
def demo_lr_schedule(train_cfg: TrainConfig) -> None:
    """打印学习率曲线（字符画），并解释三段设计各自解决什么问题。"""
    print(banner("2. 学习率调度：warmup + 余弦退火 + min_lr"))
    peak = train_cfg.learning_rate
    min_lr = train_cfg.min_lr
    warmup = train_cfg.warmup_steps
    max_steps = train_cfg.max_steps

    # 均匀取样若干行，再补上两个关键点：warmup 结束那一步、最后一步
    rows = 12
    grid = {int(round(i * (max_steps - 1) / max(rows - 1, 1))) for i in range(rows)}
    grid.update({0, warmup - 1, warmup, max_steps - 1})
    grid = sorted(s for s in grid if 0 <= s < max_steps)

    print(f"峰值学习率 {peak:.2e}，warmup {warmup} 步，最低学习率 {min_lr:.2e}，总步数 {max_steps}")
    print()
    print(f"{'step':>6}  {'lr':>10}  曲线（满格 = 峰值）")
    print("-" * 76)
    for s in grid:
        lr = get_lr(s, warmup, max_steps, peak, min_lr)
        bar = "#" * int(round(lr / peak * 40))
        note = ""
        if s == warmup - 1:
            note = "  <- warmup 结束，达到峰值"
        elif s == max_steps - 1:
            note = "  <- 最后一步，落到 min_lr"
        print(f"{s:>6}  {lr:>10.3e}  {bar}{note}")
    print()
    print("数值对照（用 common.train.get_lr 直接验证，不靠肉眼看曲线）：")
    checks = [
        ("第 0 步", get_lr(0, warmup, max_steps, peak, min_lr),
         "峰值 x 1/warmup，warmup 特意从小值起步"),
        (f"第 {warmup - 1} 步（warmup 结束）", get_lr(warmup - 1, warmup, max_steps, peak, min_lr),
         "正好等于峰值"),
        (f"第 {max_steps - 1} 步（最后一步）", get_lr(max_steps - 1, warmup, max_steps, peak, min_lr),
         "正好等于 min_lr"),
    ]
    for label, lr, note in checks:
        print("    " + pad(label, 28) + f": {lr:.3e}  （{note}）")
    print()
    print("三段设计各自解决什么问题：")
    print("    warmup（升温）：初始参数是随机的，梯度方向很不可靠。")
    print("        一上来就用大学习率，容易把参数一步带进坏区域再也回不来；")
    print("        先用小学习率试探几十到几百步，等梯度方向稳定了再加速。")
    print("        对 Post-LN 结构尤其关键（05 讲对比过），Pre-LN 容错更好但仍建议保留。")
    print("    余弦退火（降温）：后期用较小的学习率精调，更容易收敛到平坦的极小值。")
    print("        比阶梯式下降平滑、无需手工设里程碑，是 GPT 系列的默认选择。")
    print("    min_lr（兜底）：不降到 0，保留一点探索能力，避免后期完全走不动。")
    print()
    print("经验值参考（nanoGPT / GPT-2 量级）：")
    print("    peak lr : 小模型 1e-3 ~ 6e-3，大模型 1e-4 ~ 6e-4（模型越大越小）")
    print("    warmup  : 总步数的 1% ~ 5%")
    print("    min_lr  : 峰值的 1/10 左右")
    print(f"    本讲实际: peak={peak:.1e}，warmup={warmup} 步"
          f"（占 {warmup / max(max_steps, 1):.1%}），min_lr={min_lr:.1e}")


# ======================================================================
# 3. AdamW 参数分组
# ======================================================================
def demo_optimizer_groups(model: GPT, train_cfg: TrainConfig) -> None:
    """用 build_optimizer 建一个优化器，把两组的张量数 / 参数量 / weight_decay 打出来。"""
    print(banner("3. AdamW 参数分组：谁该被 weight decay"))
    optimizer = build_optimizer(model, train_cfg)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    labels = ["权重矩阵（dim>=2）", "bias 与 LayerNorm（dim<2）"]

    print(f"优化器：{type(optimizer).__name__}，betas=({train_cfg.beta1}, {train_cfg.beta2})，"
          f"lr={train_cfg.learning_rate:.1e}，共 {len(optimizer.param_groups)} 组")
    print()
    print(pad("组", 4) + pad("张量数", 10, "r") + pad("参数量", 14, "r")
          + pad("占比", 8, "r") + pad("weight_decay", 14, "r") + "   说明")
    print("-" * 92)
    for i, group in enumerate(optimizer.param_groups):
        n_tensors = len(group["params"])
        n_params = sum(p.numel() for p in group["params"])
        share = n_params / max(total, 1)
        desc = "做衰减：抑制权重变大" if group["weight_decay"] > 0 else "不衰减：让它自由取值"
        print(pad(str(i), 4) + pad(f"{n_tensors:,}", 10, "r") + pad(f"{n_params:,}", 14, "r")
              + pad(f"{share:.1%}", 8, "r") + pad(f"{group['weight_decay']:.2f}", 14, "r")
              + "   " + f"{labels[i]}，{desc}")
    print(pad("合计", 4)
          + pad(f"{sum(len(g['params']) for g in optimizer.param_groups):,}", 10, "r")
          + pad(f"{total:,}", 14, "r") + pad("100.0%", 8, "r") + pad("-", 14, "r"))
    print()
    no_decay_names = [n for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    print(f"不衰减组里都是些什么（共 {len(no_decay_names)} 个张量），前 8 个：")
    for name in no_decay_names[:8]:
        print(f"    {name}")
    if len(no_decay_names) > 8:
        print(f"    ...（其余 {len(no_decay_names) - 8} 个同理：各层的 bias 与 LayerNorm 参数）")
    print()
    print("为什么 bias 和 LayerNorm 参数不衰减？")
    print("    LayerNorm 的 weight 初始化是 1，它的职责是把归一化后的信号缩放回去。")
    print("    对它做 weight decay 等于每步都把它往 0 推，归一化信号会被逐渐压扁，")
    print("    模型的表达力直接受损 —— 这类参数本来就没有「应该尽量小」的理由。")
    print("    bias 同理：它的作用是平移，衰减它只是白白削弱拟合能力。")
    print("    所以 GPT-2/3 的标准做法是：只衰减二维及以上的权重矩阵，")
    print("    也就是 nn.Linear.weight 和 nn.Embedding.weight。")
    print()
    print(f"betas=({train_cfg.beta1}, {train_cfg.beta2}) 与默认的 (0.9, 0.999) 差在哪：")
    print("    beta2 控制二阶矩（梯度平方）滑动平均的记忆长度，约等于 1/(1-beta2) 步：")
    print("        0.999 -> 记忆约 1000 步，估计很稳但反应迟钝；")
    print("        0.95  -> 记忆约 20 步，跟得上变化，适合 batch 小、梯度噪声大的训练。")
    print("    小模型 / 小 batch 场景下 0.95 通常收敛更快，这是 GPT 系列的常见改动。")
    print("    beta1=0.9 保留默认：一阶动量看的是长期方向，没有必要动。")


# ======================================================================
# 4. 训练循环
# ======================================================================
def make_sample_fn(ds: TokenDataset, device: torch.device, prompt: str = "\n"):
    r"""构造一个「训练途中顺手看一眼生成效果」的回调，交给 common.train.train 调用。

    回调签名固定为 fn(model, step) -> str，train() 每次评估后会打印它的返回值。
    """
    try:
        prompt_ids = ds.tokenizer.encode(prompt)
    except KeyError:
        # 语料里没有换行符（比如某些中文语料），退化成用 id 0 开头
        prompt_ids = [0]

    def sample_fn(model, step):
        was_training = model.training
        with torch.no_grad():
            start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out = model.generate(start, max_new_tokens=80, temperature=0.8, top_k=40,
                                 seed=1337 + step)
        # generate 内部会调用 model.eval()，这里必须把训练状态还原回去，
        # 否则后面的 dropout 就不生效了（mini / small 预设带 dropout）。
        model.train(was_training)
        text = ds.tokenizer.decode(out[0].tolist())
        return text.replace("\n", " ")

    return sample_fn


def run_training(ds: TokenDataset, model: GPT, model_config: GPTConfig,
                 train_cfg: TrainConfig, device: torch.device, run_name: str,
                 start_step: int, best_val: float, history: list,
                 optimizer) -> dict:
    """第 4 节：训练循环。核心实现全部来自 common.train.train，本函数只负责组织与打印。"""
    print(banner("4. 训练循环：复用 common.train.train"))
    ckpt_path = OUT_DIR / f"{run_name}.pt"
    best_path = OUT_DIR / f"{run_name}_best.pt"
    set_seed(train_cfg.seed)

    if start_step >= train_cfg.max_steps:
        print(f"  跳过训练：checkpoint 已经训练到第 {start_step} 步，"
              f"不小于本次设定的 max_steps={train_cfg.max_steps}。")
        print(f"  想继续训练请看：--steps {start_step + 200} --resume")
        return {
            "skipped": True,
            "history": history,
            "steps": start_step,
            "start_step": start_step,
            "steps_run": 0,
            "elapsed": 0.0,
            "best_val": best_val,
            "final": {},
            "num_params": num_params(model),
            "checkpoint": str(ckpt_path),
            "best_checkpoint": str(best_path),
        }

    print(f"  本次要跑 {train_cfg.max_steps - start_step} 步（第 {start_step} -> "
          f"{train_cfg.max_steps} 步），每步 {train_cfg.batch_size} x {train_cfg.block_size}"
          f" = {train_cfg.batch_size * train_cfg.block_size:,} token")
    print(f"  每 {train_cfg.eval_interval} 步评估一次，每次评估各取 "
          f"{train_cfg.eval_batches} 个 train / val batch 求平均")
    print("  评估时会调用 model.generate 采样一段，直观看看模型现在会写什么。")
    print()

    with Timer() as timer:
        result = train(
            model, ds, model_config, train_cfg,
            checkpoint_path=ckpt_path,
            start_step=start_step,
            start_best=best_val,
            history=history,
            optimizer=optimizer,
            sample_fn=make_sample_fn(ds, device),
        )
    # 注意：common.utils.Timer 只在 with 语句里才会更新 elapsed，
    # 而 train() 是直接用 Timer() 读的，所以它返回的 elapsed 恒为 0。
    # 耗时 / 吞吐以本脚本自己测的墙钟时间为准。
    result["elapsed"] = timer.elapsed
    result["skipped"] = False
    result["start_step"] = start_step
    result["steps_run"] = train_cfg.max_steps - start_step
    result["num_params"] = num_params(model)
    return result


# ======================================================================
# 5. 训练曲线
# ======================================================================
def plot_curves(history: list, path: Path) -> None:
    """画两张子图：train/val loss 与学习率。没装 matplotlib 就优雅跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")                      # 无窗口后端，服务器上也能存图
        import matplotlib.pyplot as plt
    except Exception as exc:                       # 缺包、缺后端都算「跳过画图」
        print(f"  未安装 matplotlib（{type(exc).__name__}: {exc}），已跳过画图。")
        print("  想要图片的话先装一次：python -m pip install matplotlib")
        return

    if not history:
        print("  history 是空的，没有东西可画。")
        return

    steps = [h["step"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    lrs = [h["lr"] for h in history]

    # 坐标轴与图例一律用英文：默认字体没有中文字形，写中文会变成一排方块。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(steps, train_loss, marker="o", ms=3, label="train loss")
    axes[0].plot(steps, val_loss, marker="s", ms=3, label="val loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("train / val loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, lrs, color="tab:orange", marker="^", ms=3)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("learning rate schedule")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  训练曲线图: {path}")


# ======================================================================
# 6. 断点续训
# ======================================================================
def prepare_model(config: GPTConfig, train_cfg: TrainConfig, device: torch.device,
                  ckpt_path: Path, resume: bool, ds: TokenDataset):
    """决定这次是从头训练，还是接着上次的 checkpoint 跑。

    返回 (model, model_config, start_step, best_val, history, optimizer)。
    """
    print(banner("6. 断点续训：从上次的 checkpoint 继续"))
    best_path = ckpt_path.with_name(ckpt_path.stem + "_best.pt")

    if resume and ckpt_path.exists():
        model, cfg_loaded, tokenizer, payload = load_model_from_checkpoint(ckpt_path, device)
        start_step = int(payload.get("step") or 0)
        best_val = float(payload.get("best_val", float("inf")))
        history = list(payload.get("history") or [])
        meta = payload.get("meta") or {}

        print(f"  读取 checkpoint: {ckpt_path}")
        print(f"    保存时间 {meta.get('saved_at', '未知')}，"
              f"当时参数量 {meta.get('num_params', '未知')}")
        print(f"    已训练 {start_step} 步，历史最佳 val loss = {best_val:.4f}，"
              f"已有 {len(history)} 条评估记录")
        if tokenizer is not None:
            print(f"    词表大小 {tokenizer.vocab_size}"
                  "（推理要用同一份词表，所以它也存在 checkpoint 里）")

        if cfg_loaded.vocab_size != ds.vocab_size:
            print(f"    警告：checkpoint 的 vocab_size={cfg_loaded.vocab_size} 与数据集 "
                  f"{ds.name} 的 {ds.vocab_size} 不一致，建议换个 --run-name 重新训练。")

        # 优化器状态必须一起恢复：Adam 的动量属于训练状态，
        # 丢掉它会让续训开头几步出现明显的 loss 抖动。
        optimizer = build_optimizer(model, train_cfg)
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
            print("    优化器状态已恢复（一阶 / 二阶动量都没丢）")
        else:
            print("    警告：checkpoint 里没有优化器状态，只能从零开始累积动量")

        # load_model_from_checkpoint 会把模型切成 eval()（推理用），
        # 训练前必须切回 train()，否则 dropout 不生效、BatchNorm 类层也不会更新统计量。
        model.train(True)

        model_config = config
        if cfg_loaded.to_dict() != config.to_dict():
            print("    注意：checkpoint 里的模型结构与本次命令行参数不一致，")
            print("          继续训练必须沿用 checkpoint 的结构，否则权重装不上：")
            print(f"          checkpoint: {cfg_loaded.to_dict()}")
            print(f"          命令行    : {config.to_dict()}")
            model_config = cfg_loaded

        print(f"  >>> 从第 {start_step} 步继续训练（本次最多跑到第 {train_cfg.max_steps} 步）")
        return model, model_config, start_step, best_val, history, optimizer

    if resume:
        print(f"  --resume 已指定，但没有找到 {ckpt_path}，改为从头训练。")
        print(f"  （{best_path.name} 只用于推理；续训读的是 {ckpt_path.name}）")
    else:
        print("  没有指定 --resume：从头训练一个全新的模型。")
        print(f"  中途想接着跑，加 --resume 即可；checkpoint 会写到 {ckpt_path}")

    model = GPT.from_config(config).to(device)
    model.train(True)
    print(f"  新建模型：{num_params(model):,} 参数（{human_params(num_params(model))}）")
    print(f"  >>> 从第 0 步开始训练（本次最多跑到第 {train_cfg.max_steps} 步）")
    return model, config, 0, float("inf"), [], None


# ======================================================================
# 7. 结果解读
# ======================================================================
def report_results(result: dict, ds: TokenDataset, train_cfg: TrainConfig) -> None:
    """把 loss 换算成人类能判断的指标：ln(V) 基准、困惑度、过拟合差距、吞吐。"""
    print(banner("7. 结果解读：怎么判断训练得好不好"))
    history = result.get("history") or []
    if not history:
        print("  没有评估记录，无法解读（训练可能被跳过了）。")
        return

    best = min(history, key=lambda h: h["val_loss"])
    last = history[-1]
    unigram = math.log(ds.vocab_size)
    gap = last["val_loss"] - last["train_loss"]
    steps_run = int(result.get("steps_run", 0))
    total_steps = int(result.get("steps", last["step"]))
    tokens_run = steps_run * train_cfg.batch_size * train_cfg.block_size
    total_tokens = total_steps * train_cfg.batch_size * train_cfg.block_size
    elapsed = float(result.get("elapsed", 0.0))
    throughput = tokens_run / elapsed if elapsed > 0 else 0.0
    run_note = f"{steps_run} 步"
    if result.get("skipped"):
        run_note += "（本次跳过了训练）"

    rows = [
        ("随机猜测基准 ln(V)", f"{unigram:.4f}",
         f"V={ds.vocab_size}，完全不学时交叉熵的理论值"),
        ("最佳 val loss", f"{best['val_loss']:.4f}",
         f"step {best['step']}，checkpoint 记录的 best_val="
         f"{float(result.get('best_val', float('nan'))):.4f}"),
        ("最佳 val 困惑度", f"{math.exp(best['val_loss']):.2f}",
         "exp(loss)，平均在多少个候选里犹豫"),
        ("最终 train loss", f"{last['train_loss']:.4f}", f"step {last['step']}"),
        ("最终 val loss", f"{last['val_loss']:.4f}", ""),
        ("过拟合差距 val-train", f"{gap:+.4f}", "越小说明泛化越好"),
        ("本次运行耗时（墙钟）", human_time(elapsed), run_note),
        ("本次吞吐", f"{throughput:,.0f}", "token/秒，只算本次真正训练的步数"),
        ("累计训练量", f"{total_steps:,} 步", f"{total_tokens:,} token"),
    ]
    for label, value, note in rows:
        print(pad(label, 22) + pad(value, 12, "r") + "   " + note)

    # 先把结论算成字符串，再打印：这样 f-string 里不会出现嵌套引号，读起来也更清楚
    gain = unigram - best["val_loss"]
    gain_pct = gain / unigram * 100.0
    if gain > 0:
        verdict1 = (f"比随机猜测低 {gain:.4f}（降了 {gain_pct:.1f}%），"
                    "说明模型确实学到了东西。")
    else:
        verdict1 = "没有低于随机猜测，训练基本失败：先检查学习率是否太大或太小。"
    if gap < 0.3:
        verdict2 = "差距正常，泛化还不错。"
    elif gap < 0.8:
        verdict2 = "差距偏大，可以加一点 dropout 或 weight decay。"
    else:
        verdict2 = "明显过拟合：训练集快被背下来了，验证集没跟上。"
    if len(history) >= 2:
        recent = history[-1]["val_loss"] - history[max(0, len(history) - 3)]["val_loss"]
        trend = f"最近几次评估的 val loss 变化 {recent:+.4f}。"
    else:
        trend = "评估记录太少，看不出趋势。"

    print()
    print("怎么判断训练得好不好（按重要性排序）：")
    print(f"  1. 先和 ln(V)={unigram:.4f} 比：{verdict1}")
    print("  2. 再看 train 与 val 的差距：")
    print("       两者同步下降、差距很小 -> 欠拟合，加步数 / 加参数 / 加大 block_size 都可能有收益；")
    print("       val 不降而 train 还在降 -> 过拟合，该加正则化或者换更多数据。")
    print(f"       当前 {verdict2}")
    print(f"  3. 看 loss 是否还在下降：{trend}")
    print("       还在降就别停；已经平台化就该换更大的模型或更多数据，继续加步数没用。")
    print("  4. 最后看生成样例像不像人话（每次评估上面都打了一段）—— loss 是手段，不是目的。")
    print()
    print("横向对照：把这里的 val loss 和 01 讲 bigram 模型的结果比一比，")
    print("差值就是「自注意力 + 深层堆叠」相对「只看前一个字符」的净收益。")


# ======================================================================
def build_train_config(args, steps: int, device: torch.device) -> TrainConfig:
    """把命令行参数翻译成 TrainConfig。warmup 与评估间隔都按总步数按比例取。"""
    return TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_steps=steps,
        learning_rate=args.lr,
        min_lr=args.lr / 10.0,
        warmup_steps=max(10, steps // 20),          # 约 5% 的步数用来升温
        eval_interval=max(20, steps // 12),         # 一次训练至少看到 12 个点
        eval_batches=10 if args.quick else 20,      # 快速模式少评估几个 batch
        device=str(device),
    )


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="07 讲：训练一个真正的 GPT")
    parser.add_argument("--dataset", default="tinyshakespeare",
                        help="data/raw 下的语料名，默认 tinyshakespeare")
    parser.add_argument("--preset", default="tiny",
                        choices=["micro", "tiny", "mini", "small"],
                        help="模型规模预设，取值来自 common.config.get_config")
    parser.add_argument("--steps", type=int, default=None,
                        help="训练步数，默认 1200；--quick 且没显式指定时为 200")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3, help="峰值学习率")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--run-name", default=None, help="默认 gpt_<preset>_<dataset>")
    parser.add_argument("--resume", action="store_true",
                        help="从 out/<run_name>.pt 继续训练（含优化器状态）")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：200 步（未显式给 --steps 时）、更少的评估 batch")
    parser.add_argument("--no-plot", action="store_true", help="不画训练曲线")
    args = parser.parse_args()

    print(banner("07 讲：训练一个真正的 GPT"))
    print("本讲打印顺序：1 -> 2 -> 3 -> 6 -> 4 -> 5 -> 7")
    print("  第 6 节（断点续训）排在前面，因为它决定训练从哪里开始。")
    print()
    print(env_report())
    device = pick_device(args.device)
    print(f"使用设备：{describe_device(device)}")

    ds = load_dataset(args.dataset)
    print()
    print(ds.summary())

    steps = args.steps if args.steps is not None else (200 if args.quick else 1200)
    if steps < 1:
        print(f"--steps 必须大于 0，当前是 {steps}")
        return 1
    config = make_config(args.preset, vocab_size=ds.vocab_size, block_size=args.block_size)
    train_cfg = build_train_config(args, steps, device)
    run_name = args.run_name or f"gpt_{args.preset}_{args.dataset}"
    ckpt_path = OUT_DIR / f"{run_name}.pt"
    history_path = OUT_DIR / f"{run_name}_history.json"

    print()
    print(f"本次配置：preset={args.preset}  steps={steps}  batch_size={args.batch_size}  "
          f"block_size={args.block_size}  lr={args.lr:.1e}  resume={args.resume}  "
          f"run_name={run_name}")

    # 临时模型只干两件事：给第 1 节做参数量对账、给第 3 节演示优化器分组，不参与训练。
    demo_model = GPT.from_config(config)
    demo_cost_estimate(config, train_cfg, ds, num_params(demo_model))       # 1
    demo_lr_schedule(train_cfg)                                            # 2
    demo_optimizer_groups(demo_model, train_cfg)                           # 3
    del demo_model

    model, model_config, start_step, best_val, history, optimizer = prepare_model(
        config, train_cfg, device, ckpt_path, args.resume, ds)             # 6
    result = run_training(ds, model, model_config, train_cfg, device,      # 4
                          run_name, start_step, best_val, history, optimizer)

    print(banner("5. 训练曲线：存成 json 与图片"))                          # 5
    save_history(result["history"], history_path)
    print(f"  曲线数据: {history_path}（{len(result['history'])} 条评估记录）")
    if result.get("skipped"):
        print("  本次没有执行训练，曲线沿用 checkpoint 里的历史记录。")
    if args.no_plot:
        print("  --no-plot 已指定，跳过画图。")
    else:
        plot_curves(result["history"], FIG_PATH)

    report_results(result, ds, train_cfg)                                  # 7

    print(banner("小结与下一步"))
    print("1. 训练的核心五行很简单，真正难的是学习率、正则化、评估这些工程细节。")
    print("2. warmup + 余弦退火、分组 weight decay、梯度裁剪，是 GPT 训练的三件套。")
    print("3. 判断好坏一定看验证集：train loss 下降不代表模型变好。")
    print("4. checkpoint 里存了模型、优化器、配置和词表，所以能随时续训、随时推理。")
    print()
    print("现在你有了一个训练好的模型，下一讲看怎么把它用好：")
    print(f"  .\\.venv\\Scripts\\python.exe src\\08_generate.py "
          f"--checkpoint out\\{run_name}_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
