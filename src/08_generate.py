r"""08 讲：推理与采样策略 —— 让训练好的模型开口说话

07 讲结束时你手上有一个 checkpoint。这一讲把它用起来，重点回答一个问题：

    模型每一步输出的是「整个词表上的分数（logits）」，怎么从这些分数里挑出下一个 token？

这个「怎么挑」就是采样策略，它直接决定生成文本的性格：

    temperature  调概率分布的「陡峭程度」：越小越保守，越大越野
    top-k        只在概率最高的 k 个候选里挑 —— 砍掉长尾的胡言乱语
    top-p        只在累计概率达到 p 的最小集合里挑 —— 候选数量随上下文自适应

本讲的六个小节：

    1. 加载模型        —— 从 checkpoint 还原模型、词表、训练步数与最佳 val loss
    2. 采样策略的原理  —— 用同一组 logits 演示三种策略对「候选数」和「熵」的影响
    3. 温度扫描        —— 0.2 / 0.5 / 0.8 / 1.0 / 1.5 的连续对比
    4. top-k / top-p   —— 温度固定，换不同的截断方式
    5. 条件生成        —— 换几段 prompt，看模型会不会「顺着往下写」
    6. 采样速度        —— use_cache=False / True 的耗时对照（09 讲的引子）

跑法：
    .\\.venv\\Scripts\\python.exe src\\08_generate.py
    .\\.venv\\Scripts\\python.exe src\\08_generate.py --temperature 0.5 --top-k 20
    .\\.venv\\Scripts\\python.exe src\\08_generate.py --compare --quick
    .\\.venv\\Scripts\\python.exe src\\08_generate.py --prompt "ROMEO:"

如果还没有训练好的模型（out/ 下没有 .pt），本脚本会打印提示并正常退出，
不会抛异常 —— 先跑 07 讲拿到 checkpoint 再回来。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import GPTConfig  # noqa: E402
from common.gpt import GPT, _apply_top_k_top_p  # noqa: E402
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


# ======================================================================
# 通用小工具
# ======================================================================
def resolve_checkpoint(path: str | Path) -> Path:
    """把 checkpoint 路径解析成绝对路径：相对路径按「工程根目录」而不是当前目录算。

    这样无论你在工程根目录还是 src/ 下运行，--checkpoint out/xxx.pt 都指向同一个文件。
    """
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def escape_text(text: str) -> str:
    """把换行、制表符转成可见写法，否则打印出来的样例会破坏表格排版。"""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def clip_text(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def repeat_ratio(ids: List[int], n: int = 3) -> float:
    """重复率：长度为 n 的连续片段里，重复出现的比例。

    这是衡量「生成是否陷入复读」的简单指标：
        0   表示完全没有重复的 n-gram（越随机越接近这个值）
        1   表示整段就是同一个 n-gram 在循环
    """
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def entropy_of(logits: torch.Tensor) -> float:
    """把 logits 变成概率后求熵（单位 nat），用来量化「分布有多平」。

    注意：采样策略会把不合格的 token 分数设成 -inf，softmax 之后它们的概率正好是 0，
    对熵没有贡献 —— 这正是我们想要的：截断越狠，熵越小。
    """
    probs = F.softmax(logits.float(), dim=-1)
    safe = torch.where(probs > 0, probs, torch.ones_like(probs))
    return float(-(probs * safe.log()).sum().item())


def cand_count(logits: torch.Tensor) -> int:
    """仍然可被采样到的 token 数量（分数没有被设成 -inf 的那些）。"""
    return int(torch.isfinite(logits).sum().item())


def generate_ids(model: GPT, prompt_ids: List[int], device: torch.device,
                 max_new_tokens: int, temperature: float, top_k: Optional[int],
                 top_p: Optional[float], use_cache: bool, seed: int) -> List[int]:
    """统一走 GPT.generate，保证 08 与 09 讲的采样行为完全一致。"""
    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, use_cache=use_cache, seed=seed)
    return out[0].tolist()


def nll_of_continuation(model: GPT, ids: List[int], device: torch.device) -> float:
    """算「模型自己有多意外」：把整段文本喂进去，取每个位置的交叉熵平均。

    数值越小说明模型越「胸有成竹」。配合温度扫描看很有意思：
    低温生成重复文本，模型反而非常自信（loss 很低）—— 自信不等于写得好。
    """
    if len(ids) < 2:
        return float("nan")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, loss, _, _ = model(x, y)
    return float(loss.item())


# ======================================================================
# 1. 加载模型
# ======================================================================
def load_artifacts(ckpt_path: Path, device: torch.device
                   ) -> Optional[Tuple[GPT, GPTConfig, CharTokenizer, Dict[str, Any]]]:
    """加载 checkpoint。找不到或加载失败时返回 None，让调用方优雅退出。"""
    if not ckpt_path.exists():
        print(banner("找不到 checkpoint：先训练一个模型"))
        print(f"期望的文件: {ckpt_path}")
        print()
        print("08 讲的所有演示都需要一个训练好的模型（它讲的是「怎么用」，不是「怎么训」）。")
        print("请先花几分钟跑一遍 07 讲，它会顺手在 out/ 下存好 checkpoint：")
        print()
        print("    & '.\\.venv\\Scripts\\python.exe' 'src\\07_training.py' --steps 400")
        print()
        print("想快速看效果就加 --quick（200 步）；也可以换中文语料：")
        print("    & '.\\.venv\\Scripts\\python.exe' 'src\\07_training.py' --dataset zh_poetry --quick")
        print()
        print("训练完成后 out/ 下会出现两个文件，其中 *_best.pt 是验证 loss 最低的那个：")
        print("    out\\gpt_tiny_tinyshakespeare.pt")
        print("    out\\gpt_tiny_tinyshakespeare_best.pt")
        print()
        print("然后回到本讲：")
        print("    & '.\\.venv\\Scripts\\python.exe' 'src\\08_generate.py' --compare")
        print()
        print("（如果你把 checkpoint 存在别处，用 --checkpoint 指定路径即可。）")
        return None

    try:
        model, cfg, tokenizer, payload = load_model_from_checkpoint(ckpt_path, device)
    except Exception as e:                                  # noqa: BLE001
        print(banner("checkpoint 加载失败"))
        print(f"文件: {ckpt_path}")
        print(f"错误: {type(e).__name__}: {e}")
        print()
        print("常见原因：文件损坏（训练中途被中断）、或用不兼容的 torch 版本存的。")
        print("解决办法：重新运行 07 讲生成一份 checkpoint。")
        return None

    model = model.to(device)
    model.eval()
    if tokenizer is None:
        print("警告: checkpoint 里没有词表，无法把文本编码成 id，本讲无法继续。")
        print("      请用 07 讲重新训练并保存（save_checkpoint 会自动带上词表）。")
        return None
    return model, cfg, tokenizer, payload


def demo_load(model: GPT, cfg: GPTConfig, tokenizer: CharTokenizer, payload: Dict[str, Any],
              ckpt_path: Path, device: torch.device) -> None:
    print(banner("1. 加载模型：checkpoint 里都有什么"))
    n_params = sum(p.numel() for p in model.parameters())

    print(f"checkpoint : {ckpt_path}")
    print(f"文件大小   : {ckpt_path.stat().st_size / 1e6:.2f} MB")
    print(f"设备       : {describe_device(device)}")
    print()
    print("模型配置（GPTConfig）：")
    print(f"    vocab_size = {cfg.vocab_size:<6} 词表大小（字符级 tokenizer 就是不同字符数）")
    print(f"    block_size = {cfg.block_size:<6} 上下文长度上限，一次最多看这么多 token")
    print(f"    n_layer    = {cfg.n_layer:<6} Transformer Block 层数")
    print(f"    n_head     = {cfg.n_head:<6} 注意力头数（每头 {cfg.head_dim} 维）")
    print(f"    n_embd     = {cfg.n_embd:<6} 残差流宽度")
    print(f"    dropout    = {cfg.dropout}")
    print()
    print(f"参数量     : {n_params:,}（{human_params(n_params)}）")
    print(f"训练步数   : {payload.get('step', '?')}")
    best_val = payload.get("best_val")
    if best_val is not None and math.isfinite(float(best_val)):
        print(f"最佳 val loss: {float(best_val):.4f}"
              f"（困惑度 perplexity ≈ {math.exp(float(best_val)):.2f}）")
        print(f"    对照：随机猜测的 loss = ln(V) = ln({cfg.vocab_size}) = "
              f"{math.log(cfg.vocab_size):.4f}")
    else:
        print("最佳 val loss: checkpoint 里没有记录")

    meta = payload.get("meta") or {}
    if meta.get("dataset"):
        print(f"训练语料   : {meta['dataset']}（保存于 {meta.get('saved_at', '未知时间')}）")
    print()
    print("词表：")
    preview = "".join(tokenizer.itos[:40]).replace("\n", "\\n")
    print(f"    大小 {tokenizer.vocab_size}，前 40 个字符: {preview!r}")
    print()
    print("注意：字符级 tokenizer 只会认词表里的字符，遇到没见过的字会直接报错（见第 5 节）。")


# ======================================================================
# 2. 采样策略的原理
# ======================================================================
def demo_sampling_math(vocab_size: int) -> None:
    print(banner("2. 采样策略的原理：候选数与熵怎么变"))
    print("为了看清策略本身的效果，这里不依赖真实模型，而是造一组固定的 logits：")
    print("它们模拟模型在一个位置上「比较确定」的输出（前几个 token 分数明显更高）。")
    print()

    g = torch.Generator().manual_seed(1337)
    logits = torch.randn(1, vocab_size, generator=g) * 1.5
    logits[0, 3] += 4.0
    logits[0, 10] += 2.5
    logits[0, 17] += 1.5
    logits[0, 25] += 0.8

    print("三种策略各自在做什么：")
    print("  temperature : logits /= T。T<1 把差距放大（分布更尖，几乎总选 top-1）；")
    print("                T>1 把差距压平（长尾也有机会）；T=1 保持模型原始分布。")
    print("  top-k       : 只保留分数最高的 k 个，其余置为 -inf（概率 0）。")
    print("                候选数是固定的 k，不随上下文变化。")
    print("  top-p       : 按概率从大到小累加，凑够 p 就截断（核采样）。")
    print("                模型越确定，截断后剩下的候选越少；越犹豫留下的越多 —— 自适应。")
    print()

    cases: List[Tuple[str, float, Optional[int], Optional[float]]] = [
        ("不截断，T=1.0", 1.0, None, None),
        ("不截断，T=0.5", 0.5, None, None),
        ("不截断，T=1.5", 1.5, None, None),
        ("top_k=5, T=1.0", 1.0, 5, None),
        ("top_k=40, T=1.0", 1.0, 40, None),
        ("top_p=0.9, T=1.0", 1.0, None, 0.9),
        ("top_p=0.95, T=1.0", 1.0, None, 0.95),
        ("top_k=40 + top_p=0.9", 1.0, 40, 0.9),
    ]

    print(f"{'策略':<24} {'候选数':>8} {'熵(nat)':>10} {'归一化熵':>10} {'top-1 概率':>11}")
    print("-" * 68)
    rows = []
    for name, temp, top_k, top_p in cases:
        scaled = logits / max(temp, 1e-6)
        cut = _apply_top_k_top_p(scaled.clone(), top_k, top_p)
        probs = F.softmax(cut, dim=-1)
        ent = entropy_of(cut)
        n_cand = cand_count(cut)
        max_prob = float(probs.max().item())
        rows.append((name, n_cand, ent, max_prob))
        print(f"{name:<24} {n_cand:>8} {ent:>10.3f} {ent / math.log(vocab_size):>10.3f} "
              f"{max_prob:>11.4f}")

    print()
    print(f"（词表大小 {vocab_size}，均匀分布的熵 = ln(V) = {math.log(vocab_size):.3f} nat，")
    print("  归一化熵 = 熵 / ln(V)，等于 1 表示完全均匀，接近 0 表示已经退化成确定性选择。）")
    print()
    print("从表里能读出三件事：")
    print(f"  1. 温度只改形状不改候选数：T=0.5 时候选仍是 {vocab_size} 个，")
    print(f"     但熵从 {rows[0][2]:.3f} 掉到 {rows[1][2]:.3f} —— 分布被压尖了。")
    print(f"  2. top-k 把候选数硬性钉在 k：k=5 只剩 {rows[3][1]} 个候选，")
    print("     有效杜绝了从长尾里抽出离谱 token 的可能。")
    print(f"  3. top-p 的候选数是浮动的：p=0.9 留下 {rows[5][1]} 个，")
    print(f"     p=0.95 留下 {rows[6][1]} 个 —— 这就是「自适应」的含义。")
    print()
    print("工程实践里的常见组合：temperature=0.8 + top_k=40（或 top_p=0.9~0.95）。")
    print("temperature=0 不能用除法实现（会除零），标准做法是取 argmax，本工程的")
    print("generate 里用 max(temperature, 1e-6) 兜底：logits 被放大到极尖，等价于贪心。")


# ======================================================================
# 3. 温度扫描
# ======================================================================
def demo_temperature_sweep(model: GPT, tokenizer: CharTokenizer, prompt_ids: List[int],
                           device: torch.device, n_tokens: int, top_k: Optional[int],
                           seed: int) -> None:
    print(banner("3. 温度扫描：从保守重复到随机发散"))
    print("固定随机种子（每次采样前都重新播种），只改温度。")
    print("判断标准有两个：眼睛看文本，数字看 n-gram 重复率与模型自身的 loss。")
    print("最左边两档温度很低，能看到分布被压尖之后的典型症状。")
    print()

    temperatures = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5]
    for temp in temperatures:
        ids = generate_ids(model, prompt_ids, device, n_tokens, temp, top_k, None, False, seed)
        new_ids = ids[len(prompt_ids):]
        gen_text = tokenizer.decode(new_ids)
        rep = repeat_ratio(new_ids, 3)
        nll = nll_of_continuation(model, ids, device)
        print(f"--- temperature = {temp} " + "-" * 40)
        print(f"    3-gram 重复率: {rep:.3f}    模型自身 loss: {nll:.4f}")
        print(f"    生成: {clip_text(escape_text(gen_text))}")
        print()

    print("怎么读这些结果：")
    print("  低温（0.2）: 分布尖锐，几乎总选最可能的 token -> 文本通顺但很快开始复读，")
    print("               loss 很低（模型对自己的输出很自信），信息量却在下降。")
    print("  中温（0.8）: 通顺与变化之间的常用折中，是大多数对话/创作场景的默认值。")
    print("  高温（1.5）: 长尾 token 被频繁抽中 -> 拼写崩坏、语法混乱，但偶尔有惊喜。")
    print()
    print("一个容易误判的点：loss 低不代表文本好。重复文本的 loss 极低，")
    print("所以评测生成质量不能只看 loss，要看多样性指标或人工评估。")


# ======================================================================
# 4. top-k / top-p 对比
# ======================================================================
def demo_top_k_top_p(model: GPT, tokenizer: CharTokenizer, prompt_ids: List[int],
                     device: torch.device, n_tokens: int, seed: int) -> None:
    print(banner("4. top-k / top-p 对比：温度固定 0.8"))
    print("温度固定成 0.8（保持分布形状），只改截断策略。")
    print("top_k=None 表示完全不截断，让长尾里那些「概率很小但很离谱」的 token 也有机会。")
    print()

    cases: List[Tuple[str, Optional[int], Optional[float]]] = [
        ("top_k=None（不截断）", None, None),
        ("top_k=5", 5, None),
        ("top_k=40", 40, None),
        ("top_k=200", 200, None),
        ("top_p=0.9", None, 0.9),
    ]

    print(f"{'策略':<22} {'3-gram 重复率':>14} {'字符种类':>10} {'模型 loss':>11}")
    print("-" * 62)
    for name, top_k, top_p in cases:
        ids = generate_ids(model, prompt_ids, device, n_tokens, 0.8, top_k, top_p, False, seed)
        new_ids = ids[len(prompt_ids):]
        text = tokenizer.decode(new_ids)
        rep = repeat_ratio(new_ids, 3)
        nll = nll_of_continuation(model, ids, device)
        print(f"{name:<22} {rep:>14.3f} {len(set(new_ids)):>10} {nll:>11.4f}")

    print()
    print("逐段文本（同一随机种子，只有截断策略不同）：")
    for name, top_k, top_p in cases:
        ids = generate_ids(model, prompt_ids, device, n_tokens, 0.8, top_k, top_p, False, seed)
        text = tokenizer.decode(ids[len(prompt_ids):])
        print(f"  [{name}]")
        print(f"    {clip_text(escape_text(text), 200)}")
    print()
    print("结论：")
    print("  top_k 太小（5）-> 候选池被限死，文本单调甚至循环；")
    print("  top_k 太大（200，本例词表只有几十个就等于不截断）-> 长尾噪声进来了，偶尔崩坏；")
    print("  top_k=40 是 GPT-2 论文里的默认值，在词表几万的大模型上对应「砍掉 99.9% 的尾巴」。")
    print("  top_p=0.9 效果类似，但候选数会随模型当下的确定性浮动，所以更常被现代模型采用。")


# ======================================================================
# 5. 条件生成
# ======================================================================
def demo_conditional(model: GPT, tokenizer: CharTokenizer, device: torch.device,
                     n_tokens: int, prompts: List[str], temperature: float,
                     top_k: Optional[int], top_p: Optional[float], seed: int) -> None:
    print(banner("5. 条件生成：prompt 决定往哪个方向续写"))
    print("同一个模型、同一套采样参数，只换开头那几个字符，续写的内容就完全不同。")
    print("这就是「条件生成」：prompt 把模型推到一个特定的上下文分布里。")
    print()

    for raw in prompts:
        shown = raw if raw else "（空 prompt：完全由模型自由发挥）"
        print(f"prompt = {shown!r}")
        try:
            ids = tokenizer.encode(raw)
        except KeyError as e:
            # 注意：报错发生在 encode 阶段（还没轮到模型），所以 try 必须包住 encode
            print(f"    跳过：prompt 里的字符不在词表里 -> {e}")
            print("    原因：字符级 tokenizer 的词表是从训练语料统计出来的，")
            print("          英文语料（tinyshakespeare）里当然没有汉字。")
            print("    想生成中文，请先准备中文语料并重新训练：")
            print("        & '.\\.venv\\Scripts\\python.exe' 'src\\07_training.py' "
                  "--dataset zh_poetry --quick")
            print()
            continue

        if not ids:
            ids = [tokenizer.stoi.get("\n", 0)]
            print("    （prompt 为空，改用换行符作为起始 token）")
        out = generate_ids(model, ids, device, n_tokens, temperature, top_k, top_p,
                           False, seed)

        full = tokenizer.decode(out)
        cont = tokenizer.decode(out[len(ids):])
        print(f"    续写: {clip_text(escape_text(cont), 260)}")
        print(f"    全文: {clip_text(escape_text(full), 260)}")
        print()

    print("观察点：")
    print("  1. prompt 里的格式（例如大写人名加冒号）会被模型延续下去 —— 它学到的是")
    print("     语言里的模式，而不仅仅是「下一个字符」。")
    print("  2. 训练语料决定了模型的知识边界：tinyshakespeare 只有莎士比亚剧本，")
    print("     问它中文或现代知识是徒劳的。模型再大也变不出训练数据里没有的东西。")
    print("  3. 字符级模型没有 [UNK] 这个逃生舱：词表外字符直接报错。")
    print("     这正是真实系统用 BPE 子词切分的原因之一 —— 任何文本都能被切成已知子词。")


# ======================================================================
# 6. 采样速度
# ======================================================================
def demo_speed(model: GPT, tokenizer: CharTokenizer, prompt_ids: List[int],
               device: torch.device, n_tokens: int, seed: int) -> None:
    print(banner("6. 采样速度：没有 KV Cache 的代价"))
    print("generate(use_cache=False) 每一步都把整段历史重新前向一遍；")
    print("generate(use_cache=True)  只前向新生成的那一个 token，历史 K/V 直接复用。")
    print(f"下面各生成 {n_tokens} 个 token，比较耗时（同一随机种子，输出应当完全一致）。")
    print()

    results: Dict[bool, Tuple[float, List[int]]] = {}
    for use_cache in (False, True):
        with Timer() as t:
            ids = generate_ids(model, prompt_ids, device, n_tokens, 0.8, 40, None,
                               use_cache, seed)
        results[use_cache] = (t.elapsed, ids)
        tag = "use_cache=True " if use_cache else "use_cache=False"
        print(f"  {tag}: {t.elapsed:7.3f}s  "
              f"（{n_tokens / max(t.elapsed, 1e-9):6.1f} token/s）")

    t_no, ids_no = results[False]
    t_yes, ids_yes = results[True]
    speedup = t_no / max(t_yes, 1e-9)

    print()
    print(f"加速倍数: {t_no:.3f}s / {t_yes:.3f}s = {speedup:.2f}x")
    print("两种方式的输出是否一致（贪心解码下应当逐 token 相同）："
          f"{'是' if ids_no == ids_yes else '否'}")
    if ids_no != ids_yes:
        print("  小心：采样模式下随机数消耗顺序不同，输出本来就可能不同；")
        print("  09 讲会用 temperature=0（贪心）严格验证两种路径的等价性。")
    print()
    print("为什么只是「缓存」就能快这么多？")
    print("  没有 cache 时，生成第 t 个 token 要对前 t 个 token 做一次完整前向，")
    print(f"  到第 {n_tokens} 步为止累计前向的 token 数是 O(T²) 量级；")
    print("  有了 cache，每步只算 1 个新 token，累计是 O(T) 量级。")
    print("  序列越长差距越大 —— 这正是 09 讲的主题。")


# ======================================================================
def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="08 讲：推理与采样策略")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="07 讲产出的 checkpoint 路径")
    parser.add_argument("--prompt", default="\n", help="起始文本，默认换行符")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="0 表示不限制")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--compare", action="store_true",
                        help="运行完整的采样策略对比（第 3、4 节展开多组对照）")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：每段只生成 40 个 token，CPU 上几十秒跑完")
    args = parser.parse_args()

    print(banner("08 讲：推理与采样策略"))
    device = pick_device(args.device)
    set_seed(args.seed)
    print(f"使用设备: {describe_device(device)}")
    print(f"随机种子: {args.seed}")

    ckpt_path = resolve_checkpoint(args.checkpoint)
    loaded = load_artifacts(ckpt_path, device)
    if loaded is None:
        # 关键：没有模型时返回 0，让教学流程能继续往下走，而不是崩在这里
        print()
        print("本次没有执行生成演示。完成 07 讲后重新运行本脚本即可。")
        return 0

    model, cfg, tokenizer, payload = loaded
    demo_load(model, cfg, tokenizer, payload, ckpt_path, device)

    # 采样长度：--quick 时缩短，保证 CPU 上也能很快看到完整流程
    n_tokens = 40 if args.quick else args.max_new_tokens
    top_k = args.top_k or None
    if args.quick:
        print(f"\n（--quick 模式：每段只生成 {n_tokens} 个 token）")

    demo_sampling_math(cfg.vocab_size)

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.stoi.get("\n", 0)]
    print()
    print(f"后续演示统一使用 prompt = {args.prompt!r} "
          f"（编码为 {prompt_ids}），共 {len(prompt_ids)} 个 token。")

    print(banner("默认采样参数下的生成结果"))
    ids = generate_ids(model, prompt_ids, device, n_tokens, args.temperature, top_k,
                       args.top_p, False, args.seed)
    print(f"temperature={args.temperature}, top_k={top_k}, top_p={args.top_p}")
    print(escape_text(tokenizer.decode(ids)))
    print()
    print("这个结果符合你的预期吗？换参数多试几次，或者用 --compare 看完整对照。")

    if args.compare:
        demo_temperature_sweep(model, tokenizer, prompt_ids, device, n_tokens,
                               top_k, args.seed)
        demo_top_k_top_p(model, tokenizer, prompt_ids, device, n_tokens, args.seed)

    # 条件生成：英文 prompt 保证在 tinyshakespeare 词表内；中文用于演示词表外字符
    prompts = [
        "ROMEO:",
        "To be, or not to be",
        "The king ",
        "你好，世界",       # 中文：英文语料的词表里没有汉字，会走友好的报错分支
    ]
    if args.quick:
        prompts = prompts[:2] + prompts[3:4]
    demo_conditional(model, tokenizer, device, n_tokens, prompts,
                     args.temperature, top_k, args.top_p, args.seed)

    speed_tokens = 32 if args.quick else min(n_tokens, 120)
    demo_speed(model, tokenizer, prompt_ids, device, speed_tokens, args.seed)

    print(banner("小结与下一步"))
    print("1. temperature 管「分布多尖」，top-k / top-p 管「留多少候选」：")
    print("   前者调性格，后者防跑偏，两者组合使用。")
    print("   工程默认值：temperature=0.8，top_k=40 或 top_p=0.9~0.95。")
    print("2. 采样策略只改变「从分布里挑谁」，不会提升模型的知识 —— ")
    print("   模型没见过的东西，任何参数都变不出来。")
    print("3. 温度扫描要固定种子，否则你看到的差异里混着随机性，无法归因。")
    print("4. 第 6 节已经看到：光是打开 KV Cache 就快了好几倍，")
    print("   而它不改变模型输出（数学上完全等价）。这是最划算的推理优化。")
    print()
    print("下一讲把这件事讲透：")
    print("    & '.\\.venv\\Scripts\\python.exe' 'src\\09_kv_cache.py'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
