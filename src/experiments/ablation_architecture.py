r"""消融实验：结构选择到底值多少？

这是一个**可复现的对照实验脚本**：每次只改一个变量，其它全部固定，
用同一个随机种子与同一批数据训练，最后比较验证 loss。

它会回答几个工程上很实际的问题：
    1. 权重共享（tie_weights）值不值？
    2. Pre-LN 与 Post-LN 差多少？
    3. 激活函数 GELU / ReLU / SiLU 有区别吗？
    4. dropout 在什么时候才有用？

跑法：
    .\\.venv\\Scripts\\python.exe src\\experiments\\ablation_architecture.py --quick
    .\\.venv\\Scripts\\python.exe src\\experiments\\ablation_architecture.py --steps 800 --seeds 3

产物：out\experiments\ablation_architecture.json（含每组配置、参数量、loss、耗时）
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.attention import CausalSelfAttention  # noqa: E402
from common.config import GPTConfig  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.gpt import GPT, MLP, Block, LayerNorm  # noqa: E402
from common.train import TrainConfig, build_optimizer, estimate_loss  # noqa: E402
from common.utils import (  # noqa: E402
    banner,
    human_params,
    num_params,
    pick_device,
    set_seed,
    setup_console,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "experiments"


# ======================================================================
# 结构变体：通过继承 Block 来换排列顺序，其它一律不动
# ======================================================================
class PostLNBlock(Block):
    """原始 Transformer 的排列：先算子层再加残差，最后归一化。"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 use_fused_attention: bool = True):
        super().__init__(n_embd, n_head, block_size, dropout,
                         use_fused_attention=use_fused_attention)
        # Post-LN 的两条支路各需要一个独立的归一化层，不能沿用父类建好的同一个实例
        self.ln_1 = LayerNorm(n_embd)
        self.ln_2 = LayerNorm(n_embd)

    def forward(self, x, return_weights: bool = False, cache=None):
        a, att, new_cache = self.attn(x, return_weights=return_weights, cache=cache)
        x = self.ln_1(x + a)
        x = self.ln_2(x + self.mlp(x))
        return x, att, new_cache


def build_variant(name: str, vocab_size: int, block_size: int, **base):
    """按变体名构造模型。每个分支只改一个变量。"""
    kwargs = dict(vocab_size=vocab_size, block_size=block_size,
                  n_layer=base["n_layer"], n_head=base["n_head"],
                  n_embd=base["n_embd"], dropout=base.get("dropout", 0.0))

    if name == "baseline":
        return GPT(**kwargs), "基线：Pre-LN + GELU + 权重共享"
    if name == "no_tie":
        return GPT(**kwargs, tie_weights=False), "不共享输入/输出权重"
    if name == "silu":
        model = GPT(**kwargs)
        for m in model.modules():
            if isinstance(m, MLP):
                m.activation = "silu"
        return model, "激活函数换成 SiLU"
    if name == "relu":
        model = GPT(**kwargs)
        for m in model.modules():
            if isinstance(m, MLP):
                m.activation = "relu"
        return model, "激活函数换成 ReLU"
    if name == "post_ln":
        model = GPT(**kwargs)
        # 逐层替换 Block 的实现，保持权重初始化方式一致
        model.blocks = torch.nn.ModuleList([
            PostLNBlock(base["n_embd"], base["n_head"], block_size, base.get("dropout", 0.0),
                        use_fused_attention=True)
            for _ in range(base["n_layer"])
        ])
        return model, "Post-LN（先残差后归一化）"
    if name == "dropout":
        kwargs["dropout"] = 0.2
        return GPT(**kwargs), "dropout = 0.2"
    raise KeyError(f"未知变体 {name!r}")


# ======================================================================
def train_once(model, ds, device, steps: int, batch_size: int, block_size: int,
               seed: int, lr: float = 3e-3) -> dict:
    """训练一个模型并返回评估结果。整个过程不使用验证集做任何选择。"""
    set_seed(seed)
    cfg = TrainConfig(batch_size=batch_size, block_size=block_size, max_steps=steps,
                      learning_rate=lr, min_lr=lr / 10, warmup_steps=max(10, steps // 20),
                      eval_interval=10 ** 9, eval_batches=16, device=str(device))
    opt = build_optimizer(model, cfg)
    model.train()
    for _ in range(steps):
        xb, yb = ds.get_batch("train", batch_size, block_size, device)
        _, loss, _, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    losses = estimate_loss(model, ds, cfg, splits=("train", "val"))
    return {"train_loss": losses["train"], "val_loss": losses["val"],
            "params": num_params(model)}


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="结构消融实验")
    parser.add_argument("--dataset", default="tinyshakespeare")
    parser.add_argument("--preset", default="tiny", choices=["micro", "tiny", "mini"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=1, help="每个配置重复几次（取平均）")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true", help="快速模式：120 步、1 个种子")
    args = parser.parse_args()

    steps = 120 if args.quick else args.steps
    seeds = [1337, 2024, 7][: (1 if args.quick else args.seeds)]
    device = pick_device(args.device)

    print(banner("结构消融实验"))
    ds = load_dataset(args.dataset)
    print(ds.summary())
    print(f"\n设备: {device}   步数: {steps}   种子: {seeds}")

    preset = {
        "micro": dict(n_layer=2, n_head=2, n_embd=64),
        "tiny": dict(n_layer=4, n_head=4, n_embd=128),
        "mini": dict(n_layer=6, n_head=6, n_embd=192),
    }[args.preset]
    base = dict(**preset, dropout=0.0)
    print(f"基础配置: {preset}，block_size={args.block_size}，batch={args.batch_size}")

    variants = ["baseline", "no_tie", "silu", "relu", "post_ln", "dropout"]
    results = {}
    print(f"\n{'变体':<12} {'说明':<28} {'参数量':>10} {'验证 loss':>12} {'标准差':>9}")
    print("-" * 76)
    for name in variants:
        vals, params, desc = [], 0, ""
        for seed in seeds:
            model, desc = build_variant(name, ds.vocab_size, args.block_size, **base)
            out = train_once(model, ds, device, steps, args.batch_size, args.block_size, seed)
            vals.append(out["val_loss"])
            params = out["params"]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        results[name] = {"desc": desc, "params": params, "val_loss_mean": mean,
                         "val_loss_std": std, "runs": vals}
        print(f"{name:<12} {desc:<28} {human_params(params):>10} {mean:>12.4f} {std:>9.4f}")

    baseline = results["baseline"]["val_loss_mean"]
    print("\n相对基线的变化：")
    for name, r in results.items():
        if name == "baseline":
            continue
        delta = r["val_loss_mean"] - baseline
        pct = delta / baseline * 100
        verdict = "更好" if delta < -0.005 else ("更差" if delta > 0.005 else "基本无差别")
        print(f"  {name:<12} {delta:+.4f} ({pct:+.1f}%)  {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ablation_architecture.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset, "preset": args.preset, "steps": steps,
        "seeds": seeds, "batch_size": args.batch_size, "block_size": args.block_size,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")
    print("\n提醒：小步数下的差异有相当一部分来自随机性。")
    print("      要得出可靠结论，请把 --steps 调到 800 以上、--seeds 设为 3 以上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
