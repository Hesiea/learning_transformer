r"""课程脚本冒烟测试：把 src\01 ~ src\09 各用 --quick 跑一遍，检查返回码。

和 tests\test_common.py 的分工：
    test_common.py  验证「公共库的数值对不对」（手推公式 vs torch 实现）
    test_lessons.py 验证「每个课程脚本能不能从头到尾跑通」

教学工程最常见的翻车方式不是公式错，而是某个脚本在别人机器上
ImportError、参数改名、路径写死、中文引号写坏导致 SyntaxError。
这个测试就是那道最后的防线：任何一个脚本非 0 退出都算失败。

跑法：
    .\.venv\Scripts\python.exe tests\test_lessons.py                  # 全部（01~09）
    .\.venv\Scripts\python.exe tests\test_lessons.py --only 01,03     # 只跑指定讲次
    .\.venv\Scripts\python.exe tests\test_lessons.py --timeout 1800   # 放宽单脚本超时

注意：--quick 是工程约定（AGENTS.md 第 4 条），每个课程脚本都必须支持它。
      08、09 是推理脚本，没有 checkpoint 时会走「随机初始化模型」的降级路径，
      所以即使 out\ 里还没有训练好的模型，这两讲也能跑通。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.utils import human_time, setup_console  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
LESSONS = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
TAIL_LINES = 30                      # 失败时打印多少行输出，够定位问题又不会刷屏


# ----------------------------------------------------------------------
def find_script(number: str) -> Path | None:
    """01 -> src/01_xxx.py。找不到返回 None（说明这一讲还没写好，跳过即可）。"""
    matches = sorted(SRC_DIR.glob(f"{number}_*.py"))
    return matches[0] if matches else None


def parse_only(text: str) -> list:
    """把 --only 的值解析成讲次号列表：01,03 / 1,3 / 01，03 都认。"""
    numbers = []
    for token in text.replace("，", ",").split(","):
        digits = "".join(ch for ch in token.strip() if ch.isdigit())
        if digits:
            numbers.append(digits.zfill(2))
    return numbers


def run_script(script: Path, timeout: float):
    """用 --quick 跑一个课程脚本，返回 (是否成功, 耗时秒, 合并后的输出)。

    三个细节都是为了在中文 Windows 上稳定拿到可读输出：
        cwd          = 工程根目录，脚本里的相对路径（data/ out/）才对得上
        PYTHONPATH   = 工程根目录，即使脚本忘了 sys.path.insert 也能 import common
        编码固定 utf-8，脚本里的中文与特殊符号不会变成乱码或抛 UnicodeDecodeError
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    timer = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(script), "--quick"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - timer
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, elapsed, output


def print_tail(output: str, lines: int = TAIL_LINES) -> None:
    """失败时只打最后若干行：调试信息通常在末尾，前面全是正常日志。"""
    stripped = (output or "").strip()
    if not stripped:
        print("  |（没有任何输出，可能是解释器都没起来）")
        return
    for line in stripped.splitlines()[-lines:]:
        print(f"  | {line}")


# ----------------------------------------------------------------------
def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(
        description="课程脚本冒烟测试：逐个用 --quick 跑 src/01~src/09 并检查返回码")
    parser.add_argument("--only", default=None,
                        help="只跑指定讲次，逗号分隔，例如 --only 01,03；默认全部")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="单个脚本的超时秒数，默认 900")
    args = parser.parse_args()

    wanted = LESSONS if not args.only else parse_only(args.only)
    if not wanted:
        print(f"--only {args.only!r} 没有解析出任何讲次号，退出。")
        return 1
    unknown = [n for n in wanted if n not in LESSONS]
    if unknown:
        print(f"提示：{', '.join(unknown)} 不在本测试的范围内（只覆盖 01~07），会被跳过。")

    print("=" * 72)
    print("课程脚本冒烟测试（每个脚本都用 --quick 跑一遍）")
    print("=" * 72)
    print(f"解释器    : {sys.executable}")
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"单脚本超时: {args.timeout:.0f} 秒")
    print(f"待测讲次  : {', '.join(wanted)}")

    results = []          # (脚本相对路径, 是否通过, 耗时)
    skipped = []          # 文件还不存在的讲次
    for number in wanted:
        script = find_script(number)
        if script is None:
            print(f"\n[跳过] {number}：src/{number}_*.py 还不存在（可能还没写好），跳过。")
            skipped.append(number)
            continue

        rel = script.relative_to(PROJECT_ROOT)
        print(f"\n--- {rel} --quick ---")
        try:
            ok, elapsed, output = run_script(script, args.timeout)
        except subprocess.TimeoutExpired:
            ok, elapsed = False, args.timeout
            output = f"超过 {args.timeout:.0f} 秒还没结束，已强制终止。"
        except OSError as exc:
            ok, elapsed = False, 0.0
            output = f"无法启动子进程：{type(exc).__name__}: {exc}"

        results.append((str(rel), ok, elapsed))
        print(f"[{'PASS' if ok else 'FAIL'}] {rel}  用时 {elapsed:.1f}s")
        if not ok:
            print(f"  ---- 最后 {TAIL_LINES} 行输出 ----")
            print_tail(output)

    # ---- 汇总表 ----
    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    if not results:
        print("没有任何脚本被运行（全都跳过了），无法判定通过。")
        print(f"跳过：{', '.join(skipped) if skipped else '（无）'}")
        return 1

    width = max([len(name) for name, _, _ in results] + [len("脚本")])
    print(f"{'脚本'.ljust(width)}  {'状态':<6}{'耗时':>9}")
    print("-" * (width + 18))
    for name, ok, elapsed in results:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL':<6}{elapsed:>8.1f}s")
    print("-" * (width + 18))

    passed = sum(1 for _, ok, _ in results if ok)
    total_elapsed = sum(elapsed for _, _, elapsed in results)
    print(f"通过 {passed}/{len(results)}，总耗时 {human_time(total_elapsed)}")
    if skipped:
        print(f"跳过 {len(skipped)} 个（脚本还不存在）：{', '.join(skipped)}")
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print("失败项:")
        for name in failed:
            print(f"  - {name}")
        print("排查建议：先单独跑一遍失败的脚本，看最后几行报错；")
        print("          再确认它是否遵守 AGENTS.md 的约定（--quick、setup_console、sys.path）。")
    else:
        print("全部通过：src/01~src/09 都能用 --quick 跑通。")
    print("=" * 72)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
