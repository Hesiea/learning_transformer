"""公共库与 01-03 讲的自检测试。

把课程里手推的公式和工程实现做数值对照：任何一条 FAIL 都说明对应的讲解或实现有问题，
应当先修好再往下学。

跑法：
    .\\.venv\\Scripts\\python.exe tests\\test_common.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.attention import (  # noqa: E402
    CausalSelfAttention,
    Head,
    MultiHeadAttention,
    build_causal_mask,
    manual_attention,
)
from common.config import GPTConfig, get_config  # noqa: E402
from common.gpt import GPT, MLP, Block, LayerNorm, estimate_params  # noqa: E402
from common.tokenizer import CharTokenizer  # noqa: E402
from common.train import TrainConfig, build_optimizer, get_lr  # noqa: E402
from common.utils import num_params, set_seed, setup_console  # noqa: E402

RESULTS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))


def close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> bool:
    return a.shape == b.shape and (a - b).abs().max().item() < tol


# ======================================================================
# tokenizer / 数据
# ======================================================================
def test_tokenizer() -> None:
    text = "hello 世界\n你好，世界！"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    check("tokenizer: 编码长度等于字符数", len(ids) == len(text), f"{len(ids)} vs {len(text)}")
    check("tokenizer: 解码可逆", tok.decode(ids) == text)
    check("tokenizer: 词表去重", tok.vocab_size == len(set(text)), f"vocab={tok.vocab_size}")
    try:
        tok.encode("这个词表里没有→")
        check("tokenizer: 词表外字符应报错", False)
    except KeyError:
        check("tokenizer: 词表外字符应报错", True)

    # 存档 / 读档
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tok.json"
        tok.save(p)
        tok2 = CharTokenizer.load(p)
        check("tokenizer: 存取后完全一致",
              tok2.itos == tok.itos and tok2.decode(ids) == text)


# ======================================================================
# softmax / 交叉熵
# ======================================================================
def test_softmax_and_ce() -> None:
    set_seed(0)
    logits = torch.randn(8, 16)
    targets = torch.randint(0, 16, (8,))

    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exps = shifted.exp()
    mine = exps / exps.sum(-1, keepdim=True)
    check("softmax: 与 torch 一致", close(mine, F.softmax(logits, -1)))
    check("softmax: 每行和为 1", close(mine.sum(-1), torch.ones(8)))

    ce_mine = -mine.log()[torch.arange(8), targets].mean()
    check("cross_entropy: 与 torch 一致",
          abs(ce_mine.item() - F.cross_entropy(logits, targets).item()) < 1e-5,
          f"{ce_mine.item():.6f}")
    uniform = torch.zeros(4, 16)
    check("cross_entropy: 均匀分布等于 ln(V)",
          abs(F.cross_entropy(uniform, torch.zeros(4, dtype=torch.long)).item()
              - math.log(16)) < 1e-6)


# ======================================================================
# 注意力
# ======================================================================
def test_attention_shapes_and_causality() -> None:
    set_seed(1)
    B, T, hs = 2, 8, 16
    q, k, v = torch.randn(B, T, hs), torch.randn(B, T, hs), torch.randn(B, T, hs)
    out, att = manual_attention(q, k, v, causal=True)

    check("attention: 输出形状 (B,T,hs)", out.shape == (B, T, hs), str(tuple(out.shape)))
    check("attention: 权重形状 (B,T,T)", att.shape == (B, T, T), str(tuple(att.shape)))
    check("attention: 权重每行和为 1", close(att.sum(-1), torch.ones(B, T)))
    check("attention: 上三角为 0（看不到未来）",
          att.triu(diagonal=1).abs().max().item() == 0.0)
    check("attention: 非负", att.min().item() >= 0.0)

    out_torch = F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
    ).squeeze(1)
    check("attention: 与 scaled_dot_product_attention 一致", close(out, out_torch, 1e-5),
          f"max_err={(out - out_torch).abs().max().item():.2e}")

    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, -1] += 3.0
    k2[:, -1] += 3.0
    v2[:, -1] += 3.0
    out2, _ = manual_attention(q2, k2, v2, causal=True)
    check("attention: 因果性（未来不影响过去）",
          (out[:, :-1] - out2[:, :-1]).abs().max().item() == 0.0)


def test_scaling_matters() -> None:
    set_seed(2)
    d_k, T = 256, 64
    scores = torch.randn(1, 1, d_k) @ torch.randn(1, T, d_k).transpose(-2, -1)
    raw_peak = F.softmax(scores, -1).max().item()
    scaled_peak = F.softmax(scores / math.sqrt(d_k), -1).max().item()
    check("scaling: 缩放后分布更平缓", scaled_peak < raw_peak,
          f"未缩放 {raw_peak:.3f} -> 缩放 {scaled_peak:.3f}")


def test_head_module() -> None:
    set_seed(3)
    n_embd, hs, block, B, T = 32, 32, 16, 2, 10
    head = Head(n_embd, hs, block)
    x = torch.randn(B, T, n_embd)
    out, att = head(x, return_weights=True)
    check("Head: 输出形状", out.shape == (B, T, hs), str(tuple(out.shape)))
    check("Head: 权重上三角为 0", att.triu(diagonal=1).abs().max().item() == 0.0)

    q, k, v = head.query(x), head.key(x), head.value(x)
    out_manual, att_manual = manual_attention(q, k, v, causal=True)
    check("Head: 与手写公式一致",
          close(out, out_manual, 1e-5) and close(att, att_manual, 1e-5))

    n = sum(p.numel() for p in head.parameters())
    check("Head: 参数量 = 3*n_embd*head_size", n == 3 * n_embd * hs, f"{n}")


def test_multihead() -> None:
    set_seed(4)
    n_embd, n_head, block, B, T = 32, 4, 16, 2, 9
    mha = MultiHeadAttention(n_embd, n_head, block)
    csa = CausalSelfAttention(n_embd, n_head, block)
    x = torch.randn(B, T, n_embd)

    out_mha, att_mha = mha(x, return_weights=True)
    out_csa, att_csa, _ = csa(x, return_weights=True)
    check("MHA: 输出形状", out_mha.shape == (B, T, n_embd), str(tuple(out_mha.shape)))
    check("MHA: 权重形状 (B, nh, T, T)", att_mha.shape == (B, n_head, T, T),
          str(tuple(att_mha.shape)))
    check("MHA: 每个头每行和为 1", close(att_mha.sum(-1), torch.ones(B, n_head, T)))
    check("CSA: 权重形状与 MHA 一致", att_csa.shape == att_mha.shape)
    check("MHA 与 CSA: 参数量相同",
          num_params(mha) == num_params(csa), f"{num_params(mha)} vs {num_params(csa)}")


def test_causal_mask_helper() -> None:
    m = build_causal_mask(5)
    check("mask: 形状 (T,T)", m.shape == (5, 5))
    check("mask: 与 tril 一致",
          bool((m == torch.tril(torch.ones(5, 5, dtype=torch.bool))).all()))
    check("mask: 上三角为 False", not bool(m.triu(diagonal=1).any()))
    check("mask: 对角线为 True", bool(m.diagonal().all()))


def test_kv_cache_equivalence() -> None:
    """KV Cache 必须与一次性全量前向给出完全相同的结果（09 讲的前提）。"""
    set_seed(5)
    n_embd, n_head, block = 32, 4, 16
    for module in (CausalSelfAttention(n_embd, n_head, block),
                   Block(n_embd, n_head, block)):
        module.eval()
        T = 8
        x = torch.randn(1, T, n_embd)
        with torch.no_grad():
            if isinstance(module, Block):
                full, _, _ = module(x)
                outs, cache = [], None
                for t in range(T):
                    o, _, cache = module(x[:, t:t + 1], cache=cache)
                    outs.append(o)
            else:
                full, _, _ = module(x)
                outs, cache = [], None
                for t in range(T):
                    o, _, cache = module(x[:, t:t + 1], cache=cache)
                    outs.append(o)
            stepwise = torch.cat(outs, dim=1)
        err = (full - stepwise).abs().max().item()
        name = type(module).__name__
        check(f"KV Cache: {name} 逐 token 与全量前向一致", err < 1e-5, f"max_err={err:.2e}")


# ======================================================================
# GPT 组装
# ======================================================================
def test_gpt_shapes_and_estimate() -> None:
    set_seed(6)
    kwargs = dict(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(**kwargs)
    B, T = 3, 16
    idx = torch.randint(0, 65, (B, T))
    logits, loss, weights, cache = model(idx, idx, return_weights=True)

    check("GPT: logits 形状 (B,T,vocab)",
          tuple(logits.shape) == (B, T, 65), str(tuple(logits.shape)))
    check("GPT: loss 是标量", loss is not None and loss.dim() == 0)
    check("GPT: 权重形状 (n_layer,B,nh,T,T)",
          tuple(weights.shape) == (2, B, 2, T, T), str(tuple(weights.shape)))
    check("GPT: cache 长度等于层数", len(cache) == 2)
    check("GPT: 初始 loss 接近 ln(V)",
          abs(loss.item() - math.log(65)) < 0.5,
          f"{loss.item():.3f} vs {math.log(65):.3f}")

    est = estimate_params(vocab_size=65, block_size=32, n_layer=2, n_embd=64)
    check("GPT: 参数量与公式估算一致", num_params(model) == est["total"],
          f"{num_params(model)} vs {est['total']}")


def test_weight_tying() -> None:
    tied = GPT(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64, tie_weights=True)
    untied = GPT(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64, tie_weights=False)
    check("权重共享: 是同一块内存",
          tied.lm_head.weight.data_ptr() == tied.wte.weight.data_ptr())
    check("权重共享: 省下 vocab*n_embd 个参数",
          num_params(untied) - num_params(tied) == 65 * 64,
          f"{num_params(untied) - num_params(tied)}")


def test_block_residual_identity() -> None:
    """把两个子层的输出投影置零后，Block 应退化为恒等映射。"""
    set_seed(7)
    block = Block(32, 4, 16)
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        block.attn.c_proj.weight.zero_()
        if block.attn.c_proj.bias is not None:
            block.attn.c_proj.bias.zero_()
        block.mlp.fc_out.weight.zero_()
        if block.mlp.fc_out.bias is not None:
            block.mlp.fc_out.bias.zero_()
        out, _, _ = block(x)
    check("Block: 子层输出为零时是恒等映射", (out - x).abs().max().item() == 0.0)


def test_layernorm() -> None:
    ln = LayerNorm(32)
    x = torch.randn(4, 6, 32) * torch.tensor([0.1, 1.0, 10.0, 100.0]).view(4, 1, 1)
    y = ln(x)
    check("LayerNorm: 输出均值约为 0", y.mean(dim=(1, 2)).abs().max().item() < 1e-5)
    check("LayerNorm: 输出标准差约为 1",
          (y.std(dim=(1, 2), unbiased=False) - 1).abs().max().item() < 1e-3)
    check("LayerNorm: 参数量 = 2C", num_params(ln) == 64)


def test_mlp_param_share() -> None:
    """MLP 的参数量应当约等于注意力（4C²）的两倍（8C²），这也是参数分布的基本结论。"""
    n_embd = 64
    mlp = MLP(n_embd)
    attn = CausalSelfAttention(n_embd, 2, 16)
    check("MLP: 参数量约是注意力的 2 倍",
          abs(num_params(mlp) - 2 * num_params(attn)) < 0.05 * num_params(mlp),
          f"MLP={num_params(mlp)}, attn={num_params(attn)}")


# ======================================================================
# 训练基础设施
# ======================================================================
def test_lr_schedule() -> None:
    lrs = [get_lr(s, warmup=10, max_steps=100, max_lr=1e-3, min_lr=1e-4) for s in range(101)]
    check("lr: 第 0 步最小", lrs[0] < lrs[1])
    check("lr: warmup 结束达到峰值", abs(lrs[9] - 1e-3) < 1e-9, f"{lrs[9]:.2e}")
    check("lr: 单调下降到 min_lr", abs(lrs[-1] - 1e-4) < 1e-9, f"{lrs[-1]:.2e}")
    check("lr: 全程不超过峰值", max(lrs) <= 1e-3 + 1e-12)


def test_optimizer_groups() -> None:
    model = GPT(vocab_size=65, block_size=32, n_layer=1, n_head=2, n_embd=32)
    cfg = TrainConfig()
    opt = build_optimizer(model, cfg)
    check("优化器: 分为两组", len(opt.param_groups) == 2)
    check("优化器: 组 0 有 weight decay", opt.param_groups[0]["weight_decay"] == cfg.weight_decay)
    check("优化器: 组 1 无 weight decay", opt.param_groups[1]["weight_decay"] == 0.0)
    check("优化器: betas=(0.9, 0.95)",
          opt.param_groups[0]["betas"] == (cfg.beta1, cfg.beta2))
    covered = sum(len(g["params"]) for g in opt.param_groups)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    check("优化器: 覆盖全部可训练参数", covered == total, f"{covered}/{total}")


def test_config_validation() -> None:
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=64)
    check("config: head_dim 计算正确", cfg.head_dim == 16)
    check("config: 往返 dict 一致", GPTConfig.from_dict(cfg.to_dict()) == cfg)
    try:
        GPTConfig(n_embd=64, n_head=5)
        check("config: 非法 n_head 应报错", False)
    except ValueError:
        check("config: 非法 n_head 应报错", True)
    check("config: 预设 tiny 可用", get_config("tiny")["n_layer"] == 4)


def test_end_to_end_training() -> None:
    """最小规模的端到端训练：loss 必须明显下降。"""
    set_seed(11)
    vocab, block = 32, 16
    model = GPT(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # 造一个可学的规律：下一个 token 等于当前 token 加一（模 vocab）
    def batch():
        x = torch.randint(0, vocab - 1, (16, block))
        return x, x + 1

    first = last = None
    for step in range(120):
        xb, yb = batch()
        _, loss, _, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    check("端到端: 训练后 loss 显著下降", last < first * 0.35,
          f"{first:.3f} -> {last:.3f}")

    # 生成接口能跑通
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10)
    check("端到端: generate 输出长度正确", out.shape == (1, 11), str(tuple(out.shape)))
    out_c = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10,
                           use_cache=True)
    check("端到端: KV Cache 生成形状一致", out_c.shape == (1, 11))


# ======================================================================
def main() -> int:
    setup_console()
    print("=" * 72)
    print("公共库与 01-03 讲自检测试")
    print("=" * 72)
    tests = [
        test_tokenizer,
        test_softmax_and_ce,
        test_attention_shapes_and_causality,
        test_scaling_matters,
        test_head_module,
        test_multihead,
        test_causal_mask_helper,
        test_kv_cache_equivalence,
        test_gpt_shapes_and_estimate,
        test_weight_tying,
        test_block_residual_identity,
        test_layernorm,
        test_mlp_param_share,
        test_lr_schedule,
        test_optimizer_groups,
        test_config_validation,
        test_end_to_end_training,
    ]
    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} 抛出异常", False, f"{type(e).__name__}: {e}")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 72)
    print(f"结果: {passed}/{total} 通过")
    failed = [n for n, ok, _ in RESULTS if not ok]
    if failed:
        print("失败项:")
        for n in failed:
            print(f"  - {n}")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
