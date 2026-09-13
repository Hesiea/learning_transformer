"""训练基础设施：学习率调度、评估、checkpoint 读写、训练循环。

为什么把这些从课程脚本里抽出来？
    它们属于工程约定，不是知识点：
      * checkpoint 的字段结构必须全工程一致，07 讲存的模型 08 讲要能直接加载
      * 学习率调度、梯度裁剪、评估这些逻辑每讲都要用，重复写容易写歪
    课程脚本里只保留这一步在讲什么，实现细节放这里。

checkpoint 格式（全工程唯一约定）：
    {
        "model":     state_dict,
        "optimizer": state_dict,          # 续训才需要
        "config":    GPTConfig 的 dict,   # 重建模型结构
        "tokenizer": {"type": "char", "itos": [...]},
        "step":      int,
        "best_val":  float,
        "history":   [...],
        "meta":      {...},
    }
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from common.utils import Timer, num_params


# ======================================================================
# 学习率调度
# ======================================================================
def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """线性 warmup + 余弦退火，GPT 系列训练的标配。

    三个阶段各自解决一个问题：
      1. warmup（线性升温）：初始参数是随机的，梯度方向不可靠，
         直接上大学习率容易把模型带进坏区域，先用几百步试探更稳。
      2. 余弦退火：后期小学习率有利于收敛到更平坦的极小值。
      3. min_lr：不降到 0，保留一点探索能力（通常是 max_lr 的 1/10）。
    """
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)          # +1 避免第 0 步学习率为 0
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))          # 从 1 平滑降到 0
    return min_lr + coeff * (max_lr - min_lr)


@dataclass
class TrainConfig:
    """训练超参数。与模型结构（GPTConfig）分开，改训练策略时不用动模型。"""

    batch_size: int = 32
    block_size: int = 64
    max_steps: int = 1000
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95            # GPT 系列常用 0.95 而非默认的 0.999
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 20
    seed: int = 1337
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW，并且对不同参数用不同的 weight decay。

    为什么要分组？
        weight decay 的作用是把权重往 0 拉，用来抑制过拟合。
        但 LayerNorm 的 weight/bias 和所有 bias 向量本来就该自由取值，
        对它们做衰减反而有害。GPT-2/3 的标准做法是只衰减二维及以上的权重矩阵。
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))


# ======================================================================
# 评估
# ======================================================================
@torch.no_grad()
def estimate_loss(model: nn.Module, ds, cfg: TrainConfig,
                  splits: Iterable[str] = ("train", "val")) -> Dict[str, float]:
    """在若干 batch 上求平均 loss，比单个 batch 稳定得多。

    注意必须 model.eval()：否则 dropout 还在起作用，评估结果会偏高且每次都不同。
    结束后恢复 model.train()，避免影响调用方的训练状态。
    """
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}
    for split in splits:
        losses = []
        for _ in range(cfg.eval_batches):
            xb, yb = ds.get_batch(split, cfg.batch_size, cfg.block_size, cfg.device)
            _, loss, _, _ = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    if was_training:
        model.train()
    return out


# ======================================================================
# checkpoint
# ======================================================================
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: Any,
    dataset: Any,
    step: int,
    best_val: float,
    history: Optional[List[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """按统一格式保存。返回实际写入的路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = getattr(dataset, "tokenizer", None)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config),
        "tokenizer": {
            "type": "char",
            "itos": list(tokenizer.itos) if tokenizer is not None else None,
        },
        "step": step,
        "best_val": best_val,
        "history": history or [],
        "meta": {
            "dataset": getattr(dataset, "name", None),
            "num_params": num_params(model),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(extra or {}),
        },
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """读取 checkpoint。只负责加载，不负责重建模型（那一步在 load_model_from_checkpoint）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{path}\n"
            "请先训练一个模型，例如：\n"
            "  .\\.venv\\Scripts\\python.exe src\\07_training.py --steps 400"
        )
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(path: str | Path,
                               map_location: str | torch.device = "cpu"
                               ) -> Tuple[nn.Module, Any, Any, Dict[str, Any]]:
    """从 checkpoint 重建模型与词表。

    返回 (model, gpt_config, tokenizer, payload)。这个函数是 08/09 讲的入口，
    保证训练时怎么存的、推理时就怎么装回来。
    """
    from common.config import GPTConfig
    from common.gpt import GPT
    from common.tokenizer import CharTokenizer

    payload = load_checkpoint(path, map_location)
    cfg = GPTConfig.from_dict(payload["config"])
    model = GPT.from_config(cfg).to(map_location)
    model.load_state_dict(payload["model"])
    model.eval()

    tok_payload = payload.get("tokenizer") or {}
    tokenizer = CharTokenizer(tok_payload["itos"]) if tok_payload.get("itos") else None
    return model, cfg, tokenizer, payload


# ======================================================================
# 主训练循环
# ======================================================================
def train(
    model: nn.Module,
    ds,
    model_config: Any,
    cfg: TrainConfig,
    checkpoint_path: str | Path,
    start_step: int = 0,
    start_best: float = float("inf"),
    history: Optional[List[Dict[str, Any]]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    verbose: bool = True,
    sample_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """训练循环。所有课程脚本与实验脚本都走这一个入口。

    参数
    ----
    start_step : 续训时的起始步数
    optimizer  : 续训时传入，保证动量状态不丢
    sample_fn  : 可选的采样回调，形如 fn(model, step) -> str，用于定期看生成效果
    """
    device = torch.device(str(cfg.device))
    model.to(device)
    optimizer = optimizer or build_optimizer(model, cfg)
    history = history if history is not None else []

    best_val = start_best
    best_path = Path(checkpoint_path).with_name(Path(checkpoint_path).stem + "_best.pt")
    timer = Timer()
    running_loss: List[float] = []
    last_eval: Dict[str, float] = {}

    if verbose:
        print(f"\n开始训练：{cfg.max_steps} 步，batch={cfg.batch_size}，"
              f"block={cfg.block_size}，lr={cfg.learning_rate}")
        print(f"checkpoint 将保存到 {checkpoint_path}\n")

    for step in range(start_step, cfg.max_steps):
        # ---- 设定本步学习率 ----
        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.learning_rate, cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---- 前向 + 反向 ----
        xb, yb = ds.get_batch("train", cfg.batch_size, cfg.block_size, device)
        _, loss, _, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            # 梯度裁剪：防止偶发的梯度爆炸把参数一把带飞
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running_loss.append(loss.item())

        # ---- 定期评估 ----
        should_eval = (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1
        if should_eval:
            last_eval = estimate_loss(model, ds, cfg)
            history.append({
                "step": step + 1,
                "train_loss": last_eval["train"],
                "val_loss": last_eval["val"],
                "lr": lr,
                "elapsed": round(timer.elapsed, 2),
            })

            saved_best = False
            if last_eval["val"] < best_val:
                best_val = last_eval["val"]
                save_checkpoint(best_path, model, optimizer, model_config, ds,
                                step + 1, best_val, history)
                saved_best = True

            if verbose:
                flag = "  * 新的最佳" if saved_best else ""
                print(f"  step {step + 1:>5}/{cfg.max_steps}  "
                      f"train {last_eval['train']:.4f}  val {last_eval['val']:.4f}  "
                      f"lr {lr:.2e}  {timer.elapsed:.1f}s{flag}")
            if sample_fn is not None and verbose:
                text = sample_fn(model, step + 1)
                if text:
                    print(f"        样例: {text[:100]!r}")

    # ---- 收尾：保存最终 checkpoint ----
    save_checkpoint(checkpoint_path, model, optimizer, model_config, ds,
                    cfg.max_steps, best_val, history,
                    extra={"elapsed_sec": round(timer.elapsed, 1),
                           "final_train_loss": last_eval.get("train"),
                           "final_val_loss": last_eval.get("val")})

    result = {
        "steps": cfg.max_steps,
        "best_val": best_val,
        "final": last_eval,
        "elapsed": timer.elapsed,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_path),
        "mean_train_loss": sum(running_loss) / max(len(running_loss), 1),
    }
    if verbose:
        final_train = last_eval.get("train")
        final_val = last_eval.get("val")
        print(f"\n训练完成，用时 {timer.elapsed:.1f}s")
        if final_train is not None:
            print(f"  最终 train/val loss: {final_train:.4f} / {final_val:.4f}")
        print(f"  最佳 val loss       : {best_val:.4f}")
        print(f"  已保存: {checkpoint_path}")
        print(f"          {best_path}")
    return result


def save_history(history: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
