r"""把 data/raw 下的纯文本编码成训练用的 id 序列。

用法：
    .\.venv\Scripts\python.exe scripts\prepare_data.py                     # 处理 data/raw 下全部语料
    .\.venv\Scripts\python.exe scripts\prepare_data.py --dataset zh_poetry # 只处理一个
    .\.venv\Scripts\python.exe scripts\prepare_data.py --force             # 已处理过也重新编码
    .\.venv\Scripts\python.exe scripts\prepare_data.py --val-ratio 0.05    # 改验证集比例

产物（data/processed/<name>/）：
    train.bin        训练集 id 序列（二进制）
    val.bin          验证集 id 序列
    tokenizer.json   词表：id <-> 字符
    meta.json        词表大小、token 数等元信息

为什么单独走这一步？
    编码上百万字符要花几秒，而每次训练都重新编码纯属浪费；落盘之后训练脚本只读 .bin，
    快得多，也保证每次实验用的都是同一份数据（可复现）。
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import (  # noqa: E402
    available_datasets,
    load_dataset,
    prepare,
    processed_dir,
    raw_text_path,
)
from common.utils import Timer, banner, human_time, setup_console  # noqa: E402


def indent(text: str, prefix: str = "  ") -> str:
    """给多行文本统一加前缀，让输出层次清楚。"""
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def disp_width(text: str) -> int:
    """按终端显示宽度数：全角（W/F）算 2 列，其余算 1 列。中英混排对齐要靠它。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int, align: str = "l") -> str:
    fill = " " * max(0, width - disp_width(text))
    if align == "r":
        return fill + text
    return text + fill


def process_one(name: str, val_ratio: float, force: bool) -> dict:
    """处理单个语料：编码 -> 读回来 -> 自检。出错就往上抛，由 main 记录并继续下一个。"""
    src = raw_text_path(name)
    if not src.exists():
        raise FileNotFoundError(
            f"找不到原始语料 {src}（先运行：powershell -File scripts\\download_data.ps1）"
        )

    out_dir = processed_dir(name)
    existed = (out_dir / "meta.json").exists()

    with Timer() as timer:
        prepare(name, val_ratio=val_ratio, force=force)
        ds = load_dataset(name)

    if existed and not force:
        action = "已有处理结果，跳过编码（要重做就加 --force）"
    elif existed:
        action = "已按 --force 重新编码"
    else:
        action = "首次编码完成"
    print(f"  {action}，耗时 {timer.elapsed:.2f}s")
    print()
    print(indent(ds.summary()))

    # ---- 自检 1：meta 里记的 token 总数必须等于两个 .bin 的实际长度 ----
    train_n, val_n = len(ds.train), len(ds.val)
    meta_n = int(ds.meta.get("total_tokens", -1))
    if meta_n != train_n + val_n:
        raise AssertionError(
            f"meta.total_tokens={meta_n} 与 train+val={train_n + val_n} 不一致"
        )
    print(f"  自检: token 数对得上（train {train_n:,} + val {val_n:,} = {meta_n:,}）")

    # ---- 自检 2：文本 -> id -> 文本 必须完全可逆 ----
    # 从正文正中间抽一段，而不是只测开头：开头往往是版权声明之类的短行。
    text = src.read_text(encoding="utf-8")
    mid = max(0, len(text) // 2 - 40)
    sample = text[mid:mid + 80]
    ids = ds.tokenizer.encode(sample)
    back = ds.tokenizer.decode(ids)
    if back != sample:
        raise AssertionError("文本 -> id -> 文本 不可逆，tokenizer 有问题")

    # 再反过来验一次：训练数据的 id -> 文本 -> id 也要一模一样
    head_ids = [int(i) for i in ds.train[:80]]
    if ds.tokenizer.encode(ds.tokenizer.decode(head_ids)) != head_ids:
        raise AssertionError("id -> 文本 -> id 不可逆，tokenizer 有问题")

    print(f"  抽样原文: {sample!r}")
    print(f"  编码    : {ids[:16]}{' ...' if len(ids) > 16 else ''}（共 {len(ids)} 个 id）")
    print(f"  解码回来: {back!r}")
    print("  自检: 文本 -> id -> 文本 完全一致；id -> 文本 -> id 也完全一致")

    return {
        "name": name,
        "vocab_size": ds.vocab_size,
        "train_tokens": train_n,
        "val_tokens": val_n,
        "total_tokens": meta_n,
        "dtype": str(ds.train.dtype),
        "skipped": existed and not force,
        "elapsed": timer.elapsed,
    }


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="把 data/raw 下的语料编码成训练用的 id 序列")
    parser.add_argument("--dataset", default=None,
                        help="只处理这个语料，默认处理 data/raw 下全部 .txt")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例，默认 0.1")
    parser.add_argument("--force", action="store_true",
                        help="已经有处理结果也重新编码")
    args = parser.parse_args()

    print(banner("准备数据：把 data/raw 下的文本编码成 id 序列"))

    names = [args.dataset] if args.dataset else available_datasets()
    if not names:
        print("data/raw 下没有找到任何 .txt 语料。")
        print("请先下载：powershell -File scripts\\download_data.ps1")
        return 1
    print(f"待处理语料（{len(names)} 个）：{', '.join(names)}")
    print(f"验证集比例 {args.val_ratio:.0%}，force={args.force}")

    results, failed = [], []
    with Timer() as total_timer:
        for name in names:
            print(banner(f"处理语料：{name}"))
            try:
                results.append(process_one(name, args.val_ratio, args.force))
            except Exception as exc:
                # 单个语料失败（缺文件、为空、太短、编码不对……）不影响其它语料
                print(f"  失败：{type(exc).__name__}: {exc}")
                failed.append((name, f"{type(exc).__name__}: {exc}"))

    # ---- 汇总 ----
    print(banner("汇总"))
    skipped = [r for r in results if r["skipped"]]
    fresh = [r for r in results if not r["skipped"]]
    print(f"成功 {len(results)}/{len(names)}：新编码 {len(fresh)} 个，"
          f"跳过（已有产物）{len(skipped)} 个，失败 {len(failed)} 个")
    if results:
        print()
        print(pad("语料", 22) + pad("词表", 8, "r") + pad("train", 13, "r")
              + pad("val", 11, "r") + pad("合计", 13, "r") + "  存储")
        print("-" * 78)
        for r in results:
            print(pad(r["name"], 22) + pad(f"{r['vocab_size']:,}", 8, "r")
                  + pad(f"{r['train_tokens']:,}", 13, "r")
                  + pad(f"{r['val_tokens']:,}", 11, "r")
                  + pad(f"{r['total_tokens']:,}", 13, "r") + "  " + r["dtype"])
        print("-" * 78)
        print(pad("合计", 22) + pad("", 8, "r")
              + pad(f"{sum(r['train_tokens'] for r in results):,}", 13, "r")
              + pad(f"{sum(r['val_tokens'] for r in results):,}", 11, "r")
              + pad(f"{sum(r['total_tokens'] for r in results):,}", 13, "r"))
    if failed:
        print()
        print(f"失败 {len(failed)} 个：")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
    print()
    print(f"总耗时 {human_time(total_timer.elapsed)}")
    print()
    print("下一步：")
    print("  .\\.venv\\Scripts\\python.exe src\\01_tensor_basics.py    # 从第一讲开始上课")
    print("  .\\.venv\\Scripts\\python.exe src\\07_training.py --quick # 直接训练一个 GPT（200 步）")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
