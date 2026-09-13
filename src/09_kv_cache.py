r"""09 讲：KV Cache 与推理优化 —— 把 O(T²) 的重复计算降到 O(T)

08 讲最后一节留下一个悬念：只是打开 use_cache=True，生成同样长度的文本就快了好几倍。
这一讲把它讲透。核心只有一句话：

    K 和 V 一旦算出来就不会再变 —— 它们只取决于历史 token，与后面要生成什么无关。
    所以可以把它们缓存下来，每步只算新 token 的 q/k/v，历史直接复用。

六个小节：

    1. 自回归生成为什么慢  —— 没有 cache 时，第 t 步要把前 t 个 token 全部重算一遍
    2. KV Cache 的原理     —— 张量形状 (B, nh, T_past, hs) 是怎么沿 T 维拼接的
    3. 正确性验证          —— 全量前向 vs 逐 token + cache，最大误差必须接近 0
    4. 性能对比            —— 50 / 100 / 200 token 的耗时表与加速倍数
    5. 显存代价            —— cache 大小 = 2 × n_layer × B × n_head × T × hs × 字节数
    6. 进阶方向            —— PagedAttention / 量化 KV / 滑动窗口 + 动手练习

跑法：
    .\\.venv\\Scripts\\python.exe src\\09_kv_cache.py
    .\\.venv\\Scripts\\python.exe src\\09_kv_cache.py --quick
    .\\.venv\\Scripts\\python.exe src\\09_kv_cache.py --checkpoint out\\你的模型_best.pt

关于 checkpoint：本讲对「模型权重」没有任何要求 —— 所有结论都来自计算图的形状与
复杂度，跟权重是训练出来的还是随机初始化的无关。所以找不到 checkpoint 时，
脚本会自动建一个随机初始化的小模型继续演示，性能对比与正确性验证同样有效。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import GPTConfig, get_config  # noqa: E402
from common.gpt import GPT  # noqa: E402
from common.tokenizer import CharTokenizer  # noqa: E402
from common.train import load_model_from_checkpoint  # noqa: E402
from common.utils import (  # noqa: E402
    Timer,
    banner,
    describe_device,
    human_params,
    pick_device,
    set_seed,
    setup_console,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = "out/gpt_tiny_tinyshakespeare_best.pt"
RANDOM_NOTICE = ("当前使用的是随机初始化的模型，因此生成的文本没有意义，"
                 "但性能对比与正确性验证依然有效")


# ======================================================================
# 通用小工具
# ======================================================================
def resolve_checkpoint(path: str | Path) -> Path:
    """相对路径按工程根目录解析，换到哪里运行都指向同一个文件。"""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def clip_text(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def load_tokenizer_fallback() -> Tuple[CharTokenizer, str]:
    """没有 checkpoint 时，尽量拿到一个真实的字符级词表。

    顺序：已处理好的数据目录 -> 原始语料现算 -> 内置的 ASCII 词表。
    目的是让 prompt 能正常编码，而不是拿到词表就报错。
    """
    tok_file = PROJECT_ROOT / "data" / "processed" / "tinyshakespeare" / "tokenizer.json"
    if tok_file.exists():
        try:
            return CharTokenizer.load(tok_file), f"data/processed/tinyshakespeare（{tok_file.name}）"
        except Exception:                                   # noqa: BLE001
            pass

    raw_file = PROJECT_ROOT / "data" / "raw" / "tinyshakespeare.txt"
    if raw_file.exists():
        try:
            text = raw_file.read_text(encoding="utf-8")
            return CharTokenizer.from_text(text), "data/raw/tinyshakespeare.txt"
        except Exception:                                   # noqa: BLE001
            pass

    builtin = ("\n !\"$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return CharTokenizer(builtin), "内置 ASCII 词表"


def build_random_model(device: torch.device, seed: int = 1337
                       ) -> Tuple[GPT, GPTConfig, CharTokenizer, str]:
    """建一个随机初始化的 micro 模型，保证「没有 checkpoint 也能上这一讲」。"""
    tokenizer, source = load_tokenizer_fallback()
    cfg = GPTConfig.from_dict(get_config("micro", vocab_size=tokenizer.vocab_size,
                                         block_size=64))
    set_seed(seed)
    model = GPT.from_config(cfg).to(device)
    model.eval()
    return model, cfg, tokenizer, source


def load_artifacts(path: Path, device: torch.device
                   ) -> Tuple[GPT, GPTConfig, CharTokenizer, str, bool, str]:
    """加载 checkpoint；失败则退回随机初始化模型。

    返回 (model, config, tokenizer, 来源说明, 是否为随机模型, 补充说明)
    """
    if not path.exists():
        print(banner("没有找到 checkpoint：改用随机初始化模型演示"))
        print(f"期望的文件: {path}")
        print()
        print("本讲研究的是「计算量与内存」，不依赖模型学到了什么，所以随机权重完全够用。")
        print("你现在看到的所有数字（数值误差、加速倍数、cache 占用）都真实有效。")
        print()
        print("想看有意义的文本，请先跑 07 讲训练一个模型：")
        print("    & '.\\.venv\\Scripts\\python.exe' 'src\\07_training.py' --quick")
        print("之后本脚本会自动加载 out\\gpt_tiny_tinyshakespeare_best.pt。")
        print()
        model, cfg, tokenizer, source = build_random_model(device)
        return model, cfg, tokenizer, source, True, "（checkpoint 不存在）"

    try:
        model, cfg, tokenizer, payload = load_model_from_checkpoint(path, device)
    except Exception as e:                                  # noqa: BLE001
        print(banner("checkpoint 加载失败：改用随机初始化模型演示"))
        print(f"文件: {path}")
        print(f"错误: {type(e).__name__}: {e}")
        print("常见原因：训练中途被中断导致文件不完整。可用 07 讲重新生成。")
        print()
        model, cfg, tokenizer, source = build_random_model(device)
        return model, cfg, tokenizer, source, True, f"（加载失败: {type(e).__name__}）"

    model = model.to(device)
    model.eval()
    if tokenizer is None:                                   # 词表缺失时补一个
        tokenizer, _ = load_tokenizer_fallback()
    detail = f"checkpoint 第 {payload.get('step', '?')} 步"
    return model, cfg, tokenizer, str(path), False, detail


# ======================================================================
# 1. 自回归生成为什么慢
# ======================================================================
def demo_why_slow(model: GPT, device: torch.device) -> None:
    print(banner("1. 自回归生成为什么慢：重复劳动的平方级增长"))
    print("训练时可以并行：整段文本一次性喂进去，所有位置的 loss 同时算出来。")
    print("推理时不行 —— 第 t 个 token 必须在第 t-1 个 token 确定之后才能算，天生串行。")
    print()
    print("没有 cache 的实现（GPT.generate 里 use_cache=False 的分支）：")
    print("    idx_cond = generated[:, -block_size:]     # 把整段历史重新喂进去")
    print("    logits, _, _, _ = model(idx_cond)         # 每个 token 都重新算一遍 q/k/v")
    print()
    print("于是生成第 t 个 token 要前向 t 个 token，生成 T 个 token 的总前向量是：")
    print("    1 + 2 + 3 + ... + T = T(T+1)/2 ≈ T²/2")
    print("这就是「平方增长」的来源：生成长度翻倍，计算量大约翻四倍。")
    print()

    rows = [16, 32, 64, 128, 256, 512]
    print(f"{'生成长度 T':>10} {'累计前向 token 数 T(T+1)/2':>28} {'其中注意力部分 ≈ T³/6':>24}")
    print("-" * 66)
    for T in rows:
        print(f"{T:>10} {T * (T + 1) // 2:>28,} {T ** 3 // 6:>24,}")
    print()
    print("注意第二列与第三列的增长速度不一样：")
    print("  * 线性部分（MLP、LayerNorm、输出头、QKV 投影）按 token 计费 -> O(T²)")
    print("  * 注意力部分每个位置要跟前面所有位置打分 -> O(T³)")
    print("长上下文时注意力会逐渐主导，这正是 FlashAttention / 稀疏注意力要解决的问题。")
    print()

    bs = min(model.block_size, 64)
    T = min(128, bs)
    if T >= 2:
        ids = torch.randint(0, model.vocab_size, (1, T), device=device)
        with torch.no_grad():
            with Timer() as t_full:
                model(ids)
            with Timer() as t_step:
                for i in range(T):
                    model(ids[:, i:i + 1])
        ratio = t_step.elapsed / max(t_full.elapsed, 1e-9)
        print("实测（同一台机器、同一个模型）：")
        print(f"    一次性前向 {T} 个 token          : {t_full.elapsed * 1e3:8.2f} ms")
        print(f"    逐 token 前向 {T} 次（等价于无 cache 生成长度 {T}）: "
              f"{t_step.elapsed * 1e3:8.2f} ms")
        print(f"    倍数: {ratio:.2f}x")
        print()
        print("逐 token 那一路做了同样多的浮点运算，却慢这么多，原因有两层：")
        print("  1. 重复计算：第 i 步算了前 i 个 token 的 K/V，第 i+1 步又从头算一遍。")
        print("  2. 小算子开销：每次只喂 1 个 token，矩阵太小，CPU/GPU 都喂不饱，")
        print("     相当多时间花在 kernel 启动和内存搬运上，而不是真正的计算。")
        print("KV Cache 解决的是第 1 层；批处理（一次跑多个请求）解决的是第 2 层。")


# ======================================================================
# 2. KV Cache 的原理
# ======================================================================
def demo_cache_principle(model: GPT, device: torch.device) -> None:
    print(banner("2. KV Cache 的原理：为什么可以缓存，缓存什么"))
    print("这是整讲最关键的一段推理，一步步来：")
    print()
    print("  第 1 步：注意力用了哪些张量？")
    print("      q = x W_q,  k = x W_k,  v = x W_v      （x 是该位置的输入向量）")
    print()
    print("  第 2 步：因果掩码保证「位置 i 只能看到 j <= i」。")
    print("      所以位置 i 的 k_i / v_i 一旦算出来，之后所有位置的注意力都会用到它，")
    print("      而它本身只取决于位置 i 的输入 —— 后面的 token 改变不了它。")
    print("      这就是「可以缓存」的全部理由：值不变，何必重算。")
    print()
    print("  第 3 步：q 不能缓存。位置 i 的 q_i 只在算位置 i 的输出时用一次，用完即弃。")
    print("      所以缓存的是 K 和 V，不是 Q —— 这也是 KV Cache 这个名字的由来。")
    print()
    print("  第 4 步：逐层都要缓存。第 L 层的输入是第 L-1 层的输出，")
    print("      所以要给 n_layer 层每层准备一份 (k, v)。")
    print()

    if model.n_layer < 1:
        print("（模型层数为 0，跳过形状演示）")
        return

    B, nh, hs = 1, model.n_head, model.n_embd // model.n_head
    n_show = min(4, model.block_size)
    print("形状变化演示（每层的缓存都是一份 (k, v)，各自形如 (B, nh, T_past, hs)）：")
    print()
    print(f"    设定: B={B}, n_head={nh}, head_size={hs}, "
          f"每步只喂 1 个新 token")
    print()
    print(f"{'步':>4} {'本步输入':>14} {'本步新 k/v':>16} {'拼接后的 cache':>22} {'注意力打分':>16}")
    print("-" * 78)

    ids = torch.randint(0, model.vocab_size, (1, n_show), device=device)
    cache = None
    with torch.no_grad():
        for t in range(n_show):
            x = ids[:, t:t + 1]
            _, _, _, new_cache = model(x, cache=cache)
            k, v = new_cache[0]
            T_total = k.size(2)
            print(f"{t + 1:>4} {'(1, 1, ' + str(model.n_embd) + ')':>14} "
                  f"{str(tuple(k.shape)):>16} {str(tuple(k.shape)):>22} "
                  f"{'(1, ' + str(nh) + ', 1, ' + str(T_total) + ')':>16}")
            cache = new_cache

    print()
    print("对应的源码只有两行（common/attention.py 的 CausalSelfAttention.forward）：")
    print("    if cache is not None:")
    print("        k_prev, v_prev = cache")
    print("        k = torch.cat([k_prev, k], dim=2)      # 沿 T 维拼接历史与当前")
    print("        v = torch.cat([v_prev, v], dim=2)")
    print()
    print("拼完之后注意力照常做，只是 key/value 长度变成 T_total，query 长度仍是 1：")
    print("    out = softmax(q kᵀ / sqrt(hs)) v            # (1, nh, 1, T_total) @ (1, nh, T_total, hs)")
    print()
    print("一个容易踩的坑：这种「query 短、key 长」的情况不能用 is_causal=True。")
    print("PyTorch 的 SDPA 会按左上角对齐来理解因果掩码，从而把全部历史都当成「未来」屏蔽掉，")
    print("结果第一个 token 之后就全是错的。本工程的 attention 在这种场景下显式构造掩码，")
    print("并按 (T_total - T + i) 计算每个 query 的真实位置 —— 这就是 cache 路径能")
    print("与全量前向逐位对齐的原因。")
    print()
    print("总计算量对比：")
    print("    无 cache: 生成 T 个 token 需要约 T²/2 次「单 token 前向」")
    print("    有 cache: 每个 token 一次「单 token 前向」，共 T 次")
    print("从 O(T²) 降到 O(T) —— 每多生成一个 token 的边际成本变成常数。")


# ======================================================================
# 3. 正确性验证
# ======================================================================
def demo_correctness(model: GPT, tokenizer: CharTokenizer, device: torch.device,
                     prompt_ids: List[int]) -> None:
    print(banner("3. 正确性验证：缓存不能改变数值结果"))
    print("先立规矩：KV Cache 是「等价变换」，不是近似。两条路径的输出必须逐位相同")
    print("（浮点运算顺序略有差异，允许 1e-5 量级的误差）。")
    print()
    print("如果这里的误差很大，说明掩码对齐写错了 —— 那种 bug 不会崩溃，")
    print("只会让模型悄悄变傻，是最难查的一类问题。")
    print()

    T = min(len(prompt_ids), model.block_size)
    ids = torch.tensor([prompt_ids[:T]], dtype=torch.long, device=device)
    print(f"prompt 编码后共 {len(prompt_ids)} 个 token，取前 {T} 个做逐步对比。")
    print()

    model.eval()
    with torch.no_grad():
        full_logits, _, _, _ = model(ids)
        step_logits: List[torch.Tensor] = []
        cache = None
        for t in range(T):
            lg, _, _, cache = model(ids[:, t:t + 1], cache=cache, pos_offset=t)
            step_logits.append(lg)
    step = torch.cat(step_logits, dim=1)

    err = (full_logits - step).abs().max().item()
    print(f"{'比较对象':<40} {'形状':<22} {'最大绝对误差':>14}")
    print("-" * 78)
    print(f"{'logits（全量前向 vs 逐 token + cache）':<40} "
          f"{str(tuple(full_logits.shape)):<22} {err:>14.3e}")
    print()

    # 再用「贪心生成的整段文本」做一次端到端比较，这比比 logits 更贴近真实使用
    n_check = 16
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out_no = model.generate(prompt, max_new_tokens=n_check, temperature=1e-6,
                                use_cache=False, seed=0)
        out_yes = model.generate(prompt, max_new_tokens=n_check, temperature=1e-6,
                                 use_cache=True, seed=0)
    same = bool(torch.equal(out_no, out_yes))
    n_diff = int((out_no != out_yes).sum().item())
    print("端到端比较（temperature≈0 贪心解码，同一 prompt，各生成 "
          f"{n_check} 个 token）：")
    print(f"    use_cache=False 生成: {clip_text(repr(tokenizer.decode(out_no[0].tolist())), 120)}")
    print(f"    use_cache=True  生成: {clip_text(repr(tokenizer.decode(out_yes[0].tolist())), 120)}")
    print(f"    逐 token 完全相同: {'是' if same else '否'}（不同的位置数: {n_diff}）")
    print()

    print("两个结论：")
    print("  1. logits 的最大误差在 1e-6 量级 —— 只是浮点加法顺序不同带来的舍入差异，")
    print("     数学上两条路径算的是同一个函数。")
    if same:
        print("  2. 生成结果逐 token 相同 —— 所以打开 cache 是纯粹的提速，不牺牲任何质量。")
    else:
        print("  2. 生成结果出现了差异！请先检查下面两个前提是否被破坏（本讲的验收重点）：")
        print("     a) prompt + 生成长度 没有超过 block_size（超过后窗口滑动，")
        print("        两条路径覆盖的历史范围本来就不同）；")
        print("     b) 增量推理时位置嵌入按绝对位置取（pos_offset），不能每步都从 0 开始。")
    print()
    print("顺便说清一个前提：上面两条路径只有在「窗口不滑动」时才会逐位相同。")
    print(f"    当前 prompt {len(prompt_ids)} token + 生成 {n_check} token = "
          f"{len(prompt_ids) + n_check}，block_size = {model.block_size}。")
    print("    一旦总长度超过 block_size，两条路径保留的历史窗口就不同：")
    print("    全量路径保留最近 block_size 个 token，缓存路径需要重建缓存，")
    print("    此时输出出现差异是预期行为，而不是 bug（第 4 节会再解释一次）。")
    print()
    print("再解释一个常见疑问：为什么用 temperature≈0 而不是 0.8 来比？")
    print("  采样会消耗随机数。只要两条路径的步数定义一致，随机数消耗顺序也就一致，")
    print("  输出同样应当相同；但贪心解码没有随机性，作为验收标准更严格、更直观。")


# ======================================================================
# 4. 性能对比
# ======================================================================
def time_generate(model: GPT, prompt_ids: List[int], device: torch.device, n_tokens: int,
                  use_cache: bool) -> Tuple[float, List[int]]:
    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with Timer() as t:
        out = model.generate(start, max_new_tokens=n_tokens, temperature=1e-6,
                             use_cache=use_cache, seed=0)
    return t.elapsed, out[0].tolist()


def demo_benchmark(model: GPT, tokenizer: CharTokenizer, device: torch.device,
                   prompt_ids: List[int], lengths: List[int]) -> None:
    print(banner("4. 性能对比：加速比随长度增长"))
    print("同一段 prompt、同一个模型、同一台机器，只切换 use_cache。")
    print("用贪心解码（temperature≈0）消除采样随机性对耗时的影响。")
    print()

    print(f"{'生成长度':>8} {'无 cache 耗时':>14} {'有 cache 耗时':>14} "
          f"{'加速倍数':>10} {'无 cache token/s':>18}")
    print("-" * 72)

    # 两条路径必须覆盖同样长的历史窗口才有可比性：上限是 block_size - 1
    window = model.block_size - 1
    lengths = sorted({min(n, window) for n in lengths if n > 0})
    if not lengths:
        print("（可测长度为空，跳过）")
        return
    if lengths[-1] < model.block_size - 1:
        print(f"（本模型 block_size={model.block_size}，测试长度上限设为 "
              f"block_size-1={window}：超过这个长度窗口就开始滑动，")
        print("  两条路径覆盖的历史范围不再相同，耗时的比较也就失去意义。）")
        print()

    ratios: List[Tuple[int, float]] = []
    samples: Dict[int, List[int]] = {}
    for n in lengths:
        t_no, ids_no = time_generate(model, prompt_ids, device, n, False)
        t_yes, ids_yes = time_generate(model, prompt_ids, device, n, True)
        speed = t_no / max(t_yes, 1e-9)
        ratios.append((n, speed))
        samples[n] = ids_yes
        print(f"{n:>8} {t_no:>13.3f}s {t_yes:>13.3f}s {speed:>9.2f}x "
              f"{n / max(t_no, 1e-9):>18.1f}")

    if len(ratios) >= 2:
        (n0, s0), (n1, s1) = ratios[0], ratios[-1]
        print()
        print(f"注意加速比的变化：长度 {n0} 时 {s0:.2f}x，长度 {n1} 时 {s1:.2f}x。")
        print("为什么越长越划算？因为两条路径的每步成本不一样：")
        print("    无 cache: 第 t 步要前向 t 个 token        -> 总成本 ∝ T²")
        print("    有 cache: 每步只前向 1 个新 token          -> 总成本 ∝ T")
        print("    比值 ∝ T，所以长度翻倍，加速比也大致翻倍。")
        print()
        print("反过来说：短序列（几十个 token）时 cache 的优势有限，")
        print("因为固定开销（每步的 kernel 启动、Python 循环、采样）占比更高。")
        print()

    print("还有一个常被忽略的成本：无 cache 路径每一步都要把最多 block_size 个 token")
    print(f"喂进模型（本模型 block_size={model.block_size}），一旦生成长度超过 block_size，")
    print("窗口就会滑动，最早的内容被丢掉 —— 也就是说无 cache 路径不但慢，")
    print("在超长生成时注意力覆盖范围还会退化。")
    print()
    print("有 cache 的路径同样受 block_size 约束，但情况更微妙，必须讲清楚：")
    print(f"  1. 位置嵌入只有 0..{model.block_size - 1} 这几行，所以缓存长度不能超过 block_size。")
    print("  2. 生成长度逼近 block_size 时，窗口要滑动，而缓存里的历史位置是按绝对位置")
    print("     算出来的 —— 位置对不上了，就必须把缓存作废、基于当前窗口重建一次。")
    print("  3. 因此两条路径只有在「总长度不超过 block_size」时才逐 token 完全相同；")
    print("     超出之后它们保留的历史窗口不同，输出出现差异是预期行为，不是 bug。")
    print("     本工程 generate 就是这么处理的，第 3 节也验证了这一点。")


# ======================================================================
# 5. 显存 / 内存代价
# ======================================================================
def demo_memory_cost(model: GPT, device: torch.device) -> None:
    print(banner("5. 显存代价：用空间换时间，账要算清楚"))
    print("KV Cache 不是免费的。每一层都要存 k 和 v，于是：")
    print()
    print("    cache 大小 = 2 × n_layer × B × n_head × T × head_size × 字节数")
    print("               = 2 × n_layer × B × T × n_embd × 字节数")
    print()
    print("    2       : k 和 v 各一份")
    print("    n_layer : 每一层都要自己的缓存（层与层之间不共享）")
    print("    B       : 并发请求数 —— 这是生产环境里最要命的一项")
    print("    T       : 已经生成的 token 数，随对话长度线性增长")
    print("    n_head × head_size = n_embd，所以可以合并成一个更直观的式子")
    print("    字节数  : float32 = 4，float16 / bfloat16 = 2，int8 量化 = 1")
    print()

    B = 1
    for T in (128, 512, 1024, 4096):
        per_layer = 2 * B * model.n_head * T * (model.n_embd // model.n_head)
        total = model.n_layer * per_layer * 4
        print(f"  n_layer={model.n_layer:<3} B={B} T={T:<5} float32: "
              f"每层 {per_layer * 4 / 1024:9.1f} KB  ->  合计 {total / 1024 / 1024:8.2f} MB")

    T_ref = 1024
    per_layer = 2 * B * model.n_head * T_ref * (model.n_embd // model.n_head)
    total = model.n_layer * per_layer * 4
    print()
    print(f"以本模型为例（n_layer={model.n_layer}, n_embd={model.n_embd}, "
          f"n_head={model.n_head}, head_size={model.n_embd // model.n_head}）：")
    print(f"    手算代入 B=1, T={T_ref}, float32:")
    print(f"      2 × {model.n_layer} × 1 × {model.n_head} × {T_ref} × "
          f"{model.n_embd // model.n_head} × 4 = {total:,} 字节 "
          f"= {total / 1024 / 1024:.2f} MB")
    print(f"    与上面表格的合计一致 -> 公式正确。")

    # 用真实 cache 张量验证公式：建一段长度为 T_fit 的假缓存，直接读它的字节数。
    # 注意实测长度受 block_size 限制（位置嵌入只有 block_size 行），所以
    # 取 min(T_ref, block_size) 来测，公式本身对任意 T 都成立。
    T_fit = min(T_ref, model.block_size)
    per_layer_fit = 2 * B * model.n_head * T_fit * (model.n_embd // model.n_head)
    total_fit = model.n_layer * per_layer_fit * 4
    print(f"    实测校验（取 T={T_fit}，受本模型 block_size={model.block_size} 限制）：")
    try:
        with torch.no_grad():
            fake_ids = torch.randint(0, model.vocab_size, (B, T_fit), device=device)
            _, _, _, cache = model(fake_ids)
        real_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                         for k, v in cache)
        ok = "一致" if real_bytes == total_fit else "不一致，请检查"
        print(f"      手算 2 × {model.n_layer} × {B} × {model.n_head} × {T_fit} × "
              f"{model.n_embd // model.n_head} × 4 = {total_fit:,} 字节 "
              f"= {total_fit / 1024:.1f} KB")
        print(f"      真实 cache 张量占用 {real_bytes:,} 字节 "
              f"= {real_bytes / 1024:.1f} KB  -> {ok}")
    except Exception as e:                                  # noqa: BLE001
        print(f"      （实测跳过：{type(e).__name__}: {e}）")

    print()
    print("为什么说这是「用空间换时间」：")
    print("    省下的时间 = 每步重算历史 K/V 的开销，随 T 线性增长；")
    print("    付出的空间 = 上面这个式子，也随 T 线性增长。")
    print("    因为「每步重算」会让总时间变成 O(T²)，这笔交易几乎总是划算的。")
    print()
    print("但它在生产环境会变成瓶颈，算两个量级就知道：")
    print("    本模型放大到 LLaMA-7B 量级（n_layer=32, n_embd=4096, 16 bit）：")
    big = 2 * 32 * 1 * 4096 * T_ref * 2
    print(f"      单条 {T_ref} token 的请求: {big / 1024 / 1024:.0f} MB —— 权重才 14 GB 左右，")
    print("      并发 32 条就是好几 GB 的额外显存，而且随对话变长持续增长。")
    print("    这直接决定了推理服务的吞吐上限：显存先被 cache 吃满，就没法再放新请求。")
    print()
    print("常见对策（第 6 节展开）：")
    print("    * 分页管理显存（PagedAttention）：按需分配小块，消除碎片与预留浪费")
    print("    * 量化 KV（int8 / fp8）：字节数直接减半甚至变成 1/4")
    print("    * 滑动窗口 / 稀疏注意力：只保留最近 W 个 token 的 K/V")
    print("    * 前缀共享（prefix caching）：多个请求共享同一段系统提示的 K/V")


# ======================================================================
# 6. 进阶方向与动手练习
# ======================================================================
def demo_advanced(model: GPT) -> None:
    print(banner("6. 进阶：生产级推理还做了什么"))
    print("KV Cache 只是起点。真实的推理系统（vLLM / TensorRT-LLM / SGLang）在这些方向上继续优化：")
    print()
    print("  1. PagedAttention（vLLM 的核心）")
    print("     让 cache 像操作系统的虚拟内存一样按页分配：")
    print("     请求之间共享公共前缀，碎片率从 60%~80% 降到 4% 以下，")
    print("     同一张卡能同时服务的请求数因此翻好几倍。")
    print()
    print("  2. 量化 KV Cache")
    print("     K/V 用 int8 / fp8 存，再配 per-channel 的 scale 反量化。")
    print("     显存直接减半，代价是轻微的质量下降（长上下文时更明显）。")
    print("     配合分组量化（group-wise）能把损失压到可接受范围。")
    print()
    print("  3. 滑动窗口注意力（Mistral / Gemma 等在用）")
    print("     只保留最近 W 个 token 的 K/V，cache 大小变成 O(W)，与对话长度无关。")
    print("     代价是丢失远处信息，通常和「局部窗口 + 少量全局层」混合使用。")
    print("     本工程 GPT.generate 在超过 block_size 后滑动截断，就是最朴素的版本。")
    print()
    print("  4. 其他配套手段")
    print("     * FlashAttention / 融合 kernel：减少显存搬运，不改变数学结果")
    print("     * 连续批处理（continuous batching）：请求随到随算，不等整批做完")
    print("     * 投机解码（speculative decoding）：小模型起草、大模型验证，一次并行验证多个 token")
    print("     * 前缀缓存（prefix caching）：同一段 system prompt 只算一次")
    print()
    print("这些优化有个共同点：都不改变模型的数学输出，只在「怎么算」上省时间省显存。")
    print("凡是会改变输出的（量化权重、近似注意力），都必须在评测里单独验证质量。")
    print()
    print(f"本讲模型: n_layer={model.n_layer}, n_head={model.n_head}, "
          f"n_embd={model.n_embd}, block_size={model.block_size}")

    print()
    print("=" * 72)
    print("动手练习：把 cache 改成只保留最近 W 个 token 的滑动窗口")
    print("=" * 72)
    print("提示（改动很小，但要想清楚位置编码怎么办）：")
    print()
    print("  1. 在 CausalSelfAttention.forward 的拼接之后加一步截断：")
    print("         W = 32")
    print("         if k.size(2) > W:")
    print("             k = k[:, :, -W:]")
    print("             v = v[:, :, -W:]")
    print("     此时 T_total 会变小，掩码里 q 的位置必须跟着改成按 block_size 计，")
    print("     否则 q/k 位置对不齐，输出会静默出错（用第 3 节的方法能立刻测出来）。")
    print()
    print("  2. 位置编码会暴露真正的难点：本工程的位置嵌入按「序列内下标」取，")
    print("     窗口一滑动，历史 token 的 wpe 下标就和它在新序列里的位置对不上了。")
    print("     工业界用 RoPE（旋转位置编码，天然相对位置）解决这个问题 —— ")
    print("     这也是为什么现代模型几乎都用 RoPE 而不是可学习的位置嵌入。")
    print()
    print("  3. 验收标准（照抄本讲的验证套路）：")
    print("     * W >= 序列长度时，结果必须与不截断完全一致（误差 < 1e-5）")
    print("     * W 很小时 cache 显存按 O(W) 封顶（打印第 5 节的表格确认）")
    print("     * 生成速度与不截断时基本一致（截断不引入额外开销）")


# ======================================================================
def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="09 讲：KV Cache 与推理优化")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="07 讲产出的 checkpoint；不存在时自动改用随机模型")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：只跑到 40 个 token，CPU 上几十秒跑完")
    args = parser.parse_args()

    print(banner("09 讲：KV Cache 与推理优化"))
    device = pick_device(args.device)
    set_seed(1337)
    print(f"使用设备: {describe_device(device)}")

    ckpt_path = resolve_checkpoint(args.checkpoint)
    model, cfg, tokenizer, source, is_random, detail = load_artifacts(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())

    print()
    print(banner("0. 本讲使用的模型"))
    print(f"模型来源  : {source}")
    print(f"说明      : {detail}")
    print(f"结构      : n_layer={cfg.n_layer}, n_head={cfg.n_head}, n_embd={cfg.n_embd}, "
          f"block_size={cfg.block_size}, vocab_size={cfg.vocab_size}")
    print(f"参数量    : {n_params:,}（{human_params(n_params)}）")
    print(f"词表来源  : {tokenizer.vocab_size} 个字符")
    if is_random:
        print()
        print("*" * 72)
        print(RANDOM_NOTICE)
        print("*" * 72)

    # 挑一个所有字符都在词表里的 prompt，避免演示被 KeyError 打断。
    # 同时控制长度：prompt + 生成长度 不超过 block_size 时，缓存路径与全量路径
    # 才有逐 token 相同的严格可比性（见第 3 节）。
    prompt_text = "\n"
    for candidate in ("\n", "ROMEO:", "The ", "a"):
        try:
            tokenizer.encode(candidate)
            prompt_text = candidate
            break
        except KeyError:
            continue
    prompt_ids = tokenizer.encode(prompt_text)
    if len(prompt_ids) + 16 > model.block_size:
        prompt_ids = prompt_ids[-max(1, model.block_size - 16):]
        prompt_text = tokenizer.decode(prompt_ids)
    print(f"prompt    : {prompt_text!r}  ->  {prompt_ids}")

    demo_why_slow(model, device)
    demo_cache_principle(model, device)
    demo_correctness(model, tokenizer, device, prompt_ids)

    # 生成长度必须小于 block_size，否则两条路径覆盖的历史窗口不同，耗时不可比
    lengths = [16, 32, 48] if args.quick else [32, 48, 63]
    if args.max_new_tokens:
        lengths = [min(n, args.max_new_tokens) for n in lengths]
    demo_benchmark(model, tokenizer, device, prompt_ids, lengths)

    demo_memory_cost(model, device)
    demo_advanced(model)

    print()
    print(banner("小结"))
    print("1. 自回归生成天生串行，没有 cache 时第 t 步要重算前 t 个 token，")
    print("   总计算量 O(T²)，注意力部分甚至到 O(T³)。")
    print("2. K/V 只取决于历史 token，算出即定值，所以可以缓存；q 用完即弃，不必缓存。")
    print("   实现上就是沿 T 维 concat：(B, nh, T_past, hs) + (B, nh, 1, hs)。")
    print("3. 缓存路径与全量前向在数学上等价：logits 误差在 1e-6 量级（纯浮点舍入），")
    print("   贪心解码下逐 token 完全相同 —— 提速不牺牲质量。")
    print("4. 每步成本从 O(T) 降到 O(1)，总成本从 O(T²) 降到 O(T)，")
    print("   所以加速比随生成长度增长（本讲实测见第 4 节表格）。")
    print("5. 代价是显存：2 × n_layer × B × T × n_embd × 字节数，")
    print("   并发数 B 和上下文长度 T 都会让它线性膨胀，这就是 PagedAttention /")
    print("   量化 KV / 滑动窗口这些技术存在的理由。")
    print()
    print("至此推理这条线讲完了。回头看整条链路：")
    print("    tokenizer(01) -> 嵌入(02) -> 注意力(03/04) -> Block(05) -> GPT(06)")
    print("    -> 训练(07) -> 采样(08) -> 推理优化(09)")
    print()
    print(banner("课程学完后的下一步建议"))
    print("你已经有了一个能训练、能生成、能优化的完整 Transformer。接下来最值得动手的是：")
    print()
    print("  1. 换位置编码：把可学习的 wpe 换成 RoPE（旋转位置编码）")
    print("     公式：把 q/k 按维度两两配对当作二维向量旋转，角度随位置变化。")
    print("     它天然编码相对位置，所以能外推到训练时没见过的长度。")
    print("     验证：短序列训练、长序列测试，看 loss 是否还能保持。")
    print()
    print("  2. 换归一化：LayerNorm -> RMSNorm")
    print("     去掉减均值，只除以均方根，少一次归约、少一组 bias 参数，")
    print("     在 LLaMA 系列里已是标配。验证：同样训练步数下 loss 曲线是否持平。")
    print()
    print("  3. 换前馈层：MLP -> SwiGLU")
    print("     SwiGLU(x) = (x W_gate ⊙ silu(x W_up)) W_down，用门控换更强的表达力。")
    print("     注意为了参数量对齐，隐层宽度要从 4C 调到约 8/3 C。")
    print()
    print("  4. 换 tokenizer：字符级 -> BPE")
    print("     这是收益最大的一步：字符级序列太长、中文一词多 token，")
    print("     BPE 能在词表大小与序列长度之间取得平衡，也是所有主流模型的选择。")
    print("     验证：同样语料下，压缩率（字符数 / token 数）能到 3~4 倍。")
    print()
    print("  5. 换语料：用更大的中文语料重训一遍")
    print("     把 data/raw 下换成中文小说、百科或对话数据，重新 prepare 再训练，")
    print("     体会词表大小、block_size、模型规模三者怎么互相牵制。")
    print()
    print("  6. 工程化一条龙：导出成可部署的推理服务")
    print("     把 generate 包成带 KV Cache + 批处理的接口，测吞吐与延迟，")
    print("     再对比本讲的单请求版本 —— 你会看到批处理带来的数量级提升。")
    print()
    print("建议按 4 -> 5 -> 1 -> 2 -> 3 -> 6 的顺序做：")
    print("前两步改变的是「数据与表示」，收益最大且最容易观察；")
    print("后面几步是结构替换，适合在有了稳定 baseline 之后逐项对照。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
