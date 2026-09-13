r"""消融实验：规模与数据 —— 到底该把预算花在哪？

两个经典问题，用同一个训练流程各跑一遍就有答案：

    实验 A「模型规模」：数据固定，把参数量从 0.1M 加到 5M，验证 loss 怎么变？
    实验 B「数据规模」：模型固定，只用 5% / 20% / 100% 的训练数据，验证 loss 怎么变？

这两个实验合起来就是「Chinchilla 式」思维的入门版：
在固定算力预算下，模型太大而数据太少（或反之）都不是最优的。

跑法：
    .\\.venv\\Scripts\\python.exe src\\experiments\\ablation_scale.py --quick
    .\\.venv\\Scripts\\python.exe src\\experiments\\ablation_scale.py --steps 600

产物：out\experiments\ablation_scale.json，以及（若装了 matplotlib）一张曲线图
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.config import GPTConfig  # noqa: E402
from common.data import load_dataset  # noqa: E402
from common.gpt import GPT, estimate_params  # noqa: E402
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


class SubsetDataset:
    """只暴露训练集前 fraction 比例的包装器，用来模拟数据量不足的情形。

    验证集始终用完整的 —— 我们要比较的是泛化能力，评估口径必须一致。
    """

    def __init__(self, ds, fraction: float):
        self._ds = ds
        self.name = ds.name
        self.tokenizer = ds.tokenizer
        n = max(int(len(ds.train) * fraction), 1024)
        self.train = ds.train[:n]
        self.val = ds.val
        self.meta = ds.meta

    @property
    def vocab_size(self) -> int:
        return self._ds.vocab_size

    def split(self, which: str = "train"):
        return self._ds.split(which) if which != "train" else self.train

    def get_batch(self, split: str = "train", batch_size: int = 32,
                  block_size: int = 64, device="cpu"):
        data = self.split(split)
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
        ])
        return x.to(device), y.to(device)


def run_one(ds, device, n_layer: int, n_head: int, n_embd: int, steps: int,
            batch_size: int, block_size: int, seed: int, lr: float = 3e-3) -> dict:
    """建模型、训练、评估，返回一份可比较的记录。"""
    set_seed(seed)
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=block_size, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=0.0)
    model = GPT.from_config(cfg).to(device)
    tcfg = TrainConfig(batch_size=batch_size, block_size=block_size, max_steps=steps,
                       learning_rate=lr, min_lr=lr / 10, warmup_steps=max(10, steps // 20),
                       eval_interval=10 ** 9, eval_batches=16, device=str(device))
    opt = build_optimizer(model, tcfg)
    model.train()
    for _ in range(steps):
        xb, yb = ds.get_batch("train", batch_size, block_size, device)
        _, loss, _, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
    losses = estimate_loss(model, ds, tcfg, splits=("train", "val"))
    tokens_seen = steps * batch_size * block_size
    return {
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "params": num_params(model),
        "train_tokens_total": int(len(ds.train)),
        "tokens_seen": tokens_seen,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "perplexity": math.exp(losses["val"]),
    }


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="规模与数据消融实验")
    parser.add_argument("--dataset", default="tinyshakespeare")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true", help="快速模式：120 步、更小的网格")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    steps = 120 if args.quick else args.steps
    device = pick_device(args.device)

    print(banner("规模与数据消融实验"))
    ds = load_dataset(args.dataset)
    print(ds.summary())
    print(f"\n设备: {device}   每组训练步数: {steps}")
    print(f"随机猜测基准 ln(V) = {math.log(ds.vocab_size):.4f}")

    results = {"dataset": args.dataset, "steps": steps, "model_scale": [], "data_scale": []}

    # ---------------- 实验 A：模型规模 ----------------
    print(banner("实验 A：模型规模（数据固定）"))
    model_grid = ([("0.1M", 2, 2, 64), ("0.8M", 4, 4, 128)] if args.quick
                  else [("0.1M", 2, 2, 64), ("0.4M", 4, 4, 96),
                        ("0.8M", 4, 4, 128), ("2.5M", 6, 6, 192)])
    print(f"{'配置':<22} {'参数量':>10} {'train loss':>12} {'验证 loss':>12} {'困惑度':>10}")
    print("-" * 70)
    for label, nl, nh, ne in model_grid:
        r = run_one(ds, device, nl, nh, ne, steps, args.batch_size, args.block_size, 1337)
        r["label"] = label
        results["model_scale"].append(r)
        print(f"{label + f' (L{nl}/H{nh}/C{ne})':<22} {human_params(r['params']):>10} "
              f"{r['train_loss']:>12.4f} {r['val_loss']:>12.4f} {r['perplexity']:>10.2f}")

    # ---------------- 实验 B：数据规模 ----------------
    print(banner("实验 B：数据规模（模型固定）"))
    data_fracs = [0.05, 1.0] if args.quick else [0.05, 0.2, 0.5, 1.0]
    nl, nh, ne = (4, 4, 128)
    print(f"固定模型 L{nl}/H{nh}/C{ne}（约 {human_params(estimate_params(ds.vocab_size, args.block_size, nl, ne)['total'])}）")
    print(f"\n{'训练数据占比':>14} {'训练 token':>14} {'见到的 token':>14} {'验证 loss':>12}")
    print("-" * 60)
    full_train_tokens = len(ds.train)
    for frac in data_fracs:
        sub = SubsetDataset(ds, frac)
        r = run_one(sub, device, nl, nh, ne, steps, args.batch_size, args.block_size, 1337)
        r["fraction"] = frac
        r["train_tokens_used"] = int(len(sub.train))
        results["data_scale"].append(r)
        print(f"{frac:>13.0%} {len(sub.train):>14,} {r['tokens_seen']:>14,} "
              f"{r['val_loss']:>12.4f}")

    # ---------------- 解读 ----------------
    print(banner("怎么读这些结果"))
    a = results["model_scale"]
    if len(a) >= 2:
        big, small = a[-1], a[0]
        gain = small["val_loss"] - big["val_loss"]
        ratio = big["params"] / small["params"]
        print(f"实验 A：参数量放大 {ratio:.1f} 倍，验证 loss 降低 {gain:.4f}")
        if gain > 0.5:
            print("        -> 这个任务上「模型还不够大」，继续加参数仍有明显收益。")
        elif gain > 0.1:
            print("        -> 收益递减但还没饱和，可以继续加参数或加步数。")
        else:
            print("        -> 已经接近瓶颈，此时加数据或加步数比加参数更划算。")
        print(f"        （注意每组只训练了 {steps} 步，小模型可能尚未充分收敛，")
        print("          这是本实验最大的混杂因素：要更公平，应让每组都训练到收敛。）")

    b = results["data_scale"]
    if len(b) >= 2:
        first, last = b[0], b[-1]
        print(f"\n实验 B：数据量从 {first['fraction']:.0%} 增到 {last['fraction']:.0%}，"
              f"验证 loss 从 {first['val_loss']:.4f} 变到 {last['val_loss']:.4f}")
        gap_small = first["train_loss"] - first["val_loss"]
        gap_full = last["train_loss"] - last["val_loss"]
        print(f"        train 与 val 的差距：小数据 {gap_small:+.4f}，全量 {gap_full:+.4f}")
        if gap_small > gap_full + 0.05:
            print("        -> 数据少时泛化差距明显更大，这就是过拟合的直接证据。")
        else:
            print("        -> 差距变化不明显，说明模型容量还没大到会背下数据。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ablation_scale.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")

    if not args.no_plot:
        plot(results)
    return 0


def plot(results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("（未安装 matplotlib，跳过绘图）")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    a = results["model_scale"]
    axes[0].plot([r["params"] for r in a], [r["val_loss"] for r in a], marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("参数量（对数轴）")
    axes[0].set_ylabel("验证 loss")
    axes[0].set_title("模型规模 vs 验证 loss")
    axes[0].grid(alpha=0.3)

    b = results["data_scale"]
    axes[1].plot([r["fraction"] for r in b], [r["val_loss"] for r in b],
                 marker="s", label="val")
    axes[1].plot([r["fraction"] for r in b], [r["train_loss"] for r in b],
                 marker="^", label="train")
    axes[1].set_xlabel("使用的训练数据比例")
    axes[1].set_ylabel("loss")
    axes[1].set_title("数据规模 vs loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path = OUT_DIR / "ablation_scale.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"曲线图: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
