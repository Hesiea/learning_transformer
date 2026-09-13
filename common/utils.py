"""通用工具：计时、随机种子、参数统计、日志、控制台编码。

写作约定（本工程全局遵守）：
    * 中文里的强调引号一律用「」（外层）和『』（内层）
    * 代码里的 " 和 ' 只用于真正的字符串定界
    这样混排时不会出现歧义，也不会误伤字符串。
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ======================================================================
# 可复现性
# ======================================================================
def set_seed(seed: int = 1337) -> None:
    """固定所有随机源。做对照实验时务必先调用它。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(prefer: str = "cpu") -> torch.device:
    """选设备。本工程按 CPU 设计，留出 cuda / mps 分支以便换机器时直接用。"""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return f"cpu ({torch.get_num_threads()} threads)"


# ======================================================================
# 计时
# ======================================================================
class Timer:
    """极简计时器：with Timer() as t: ... 之后读 t.elapsed。"""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start

    def reset(self) -> None:
        self.start = time.perf_counter()

    @property
    def minutes(self) -> float:
        return self.elapsed / 60.0


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


# ======================================================================
# 模型信息
# ======================================================================
def num_params(module: Any, trainable_only: bool = True) -> int:
    """统计参数量。module 可以是 nn.Module，也可以是含 parameters 的任意对象。"""
    params = getattr(module, "parameters", None)
    if params is None:
        return 0
    it = params() if callable(params) else params
    total = 0
    for p in it:
        if trainable_only and hasattr(p, "requires_grad") and not p.requires_grad:
            continue
        total += p.numel()
    return total


def human_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def module_table(module: torch.nn.Module, top: Optional[int] = None) -> str:
    """按参数量打印子模块表，用来快速看清参数都花在哪儿了。"""
    rows = []
    for name, child in module.named_children():
        rows.append((name, type(child).__name__, num_params(child)))
    rows.sort(key=lambda r: -r[2])
    if top:
        rows = rows[:top]
    width = max((len(r[0]) for r in rows), default=4)
    lines = [f"{'子模块'.ljust(width)}  {'类型':<22} {'参数量':>10}"]
    lines.append("-" * (width + 36))
    for name, cls, n in rows:
        lines.append(f"{name.ljust(width)}  {cls:<22} {human_params(n):>10}")
    lines.append("-" * (width + 36))
    lines.append(f"{'合计'.ljust(width)}  {'':<22} {human_params(num_params(module)):>10}")
    return "\n".join(lines)


# ======================================================================
# 日志 / 结果落盘
# ======================================================================
class History:
    """把训练过程中的指标记下来，最后存成 json 方便画图。"""

    def __init__(self) -> None:
        self.records: list[Dict[str, Any]] = []

    def log(self, **kwargs: Any) -> None:
        self.records.append(dict(kwargs))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: str) -> list:
        return [r.get(key) for r in self.records]


@contextmanager
def section(title: str):
    """打印一个带标题的分节，方便在长输出里定位。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    yield


def banner(text: str, width: int = 72, char: str = "=") -> str:
    return f"\n{char * width}\n{text}\n{char * width}"


def fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_root() -> Path:
    """工程根目录。"""
    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """把相对路径解析成相对工程根目录的绝对路径（而不是当前工作目录）。"""
    p = Path(path)
    return p if p.is_absolute() else package_root() / p


# ======================================================================
# 环境信息
# ======================================================================
def env_report() -> str:
    """打印一份环境快照，用于记录实验条件。"""
    import platform
    import sys

    lines = [
        f"python   : {sys.version.split()[0]} ({platform.machine()})",
        f"torch    : {torch.__version__}",
        f"numpy    : {np.__version__}",
        f"threads  : {torch.get_num_threads()}",
        f"device   : {describe_device(pick_device())}",
    ]
    try:
        import matplotlib

        lines.append(f"matplotlib: {matplotlib.__version__}")
    except Exception:
        lines.append("matplotlib: 未安装（画图功能会自动跳过）")
    return "\n".join(lines)


def cpu_name() -> str:
    return os.environ.get("PROCESSOR_IDENTIFIER", "unknown cpu")


def setup_console() -> None:
    """修正 Windows 控制台的编码，避免中文输出变乱码。

    Windows 上 Python 默认用系统代码页（简中系统是 GBK）写 stdout，
    而源码是 UTF-8，遇到生僻字可能报 UnicodeEncodeError 或显示乱码。
    这里统一改成 UTF-8；失败（例如输出被重定向到不支持的环境）就静默跳过。
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
