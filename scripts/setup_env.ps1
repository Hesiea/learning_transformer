# =====================================================================
#  Transformer 学习工程 - 环境初始化 (Windows / PowerShell)
# ---------------------------------------------------------------------
#  做两件事：
#     1) 准备一份本地 Python 解释器（优先复用，没有则从 python.org 下载安装到 tools\）
#     2) 用它在工程内建 .venv 并安装 CPU 版 PyTorch 等依赖
#
#  为什么不用系统自带的 Python？
#     本机只有 MSYS2 的 Python，它的平台标签是 mingw_x86_64_ucrt_gnu，
#     pip 不认 Windows 的 win_amd64 轮子，numpy/torch 只能从源码编译（几乎必然失败）。
#     所以这里在本工程目录里装一份官方 CPython，装完即用，不影响系统。
#
#  用法：pwsh -File scripts\setup_env.ps1
# =====================================================================
param(
    [string]$PyVersion = '3.12.10',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root 'tools'
$PyDir = Join-Path $Tools 'python312'
$PyExe = Join-Path $PyDir 'python.exe'
$VenvPy = Join-Path $Root '.venv\Scripts\python.exe'

Write-Host "工程根目录: $Root"

# ---------- 1. 准备解释器 ----------
if ((Test-Path $PyExe) -and -not $Force) {
    Write-Host "复用已有解释器: $PyExe"
} else {
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null
    $Installer = Join-Path $Tools "python-$PyVersion-amd64.exe"
    if (-not (Test-Path $Installer)) {
        $Url = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-amd64.exe"
        Write-Host "下载官方 CPython: $Url"
        Invoke-WebRequest -Uri $Url -OutFile $Installer -UseBasicParsing -TimeoutSec 600
    }
    Write-Host "静默安装到工程目录: $PyDir"
    $InstallArgs = @(
        '/quiet',
        'InstallAllUsers=0',
        "TargetDir=$PyDir",
        'Include_launcher=0',
        'Include_test=0',
        'Include_doc=0',
        'Include_tcltk=1',
        'Include_pip=1',
        'PrependPath=0',
        'Shortcuts=0',
        'AssociateFiles=0'
    )
    $p = Start-Process -FilePath $Installer -ArgumentList $InstallArgs -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "安装失败，退出码 $($p.ExitCode)" }
}

& $PyExe -V
$tag = & $PyExe -c "import sysconfig;print(sysconfig.get_platform())"
Write-Host "平台标签: $tag（必须是 win-amd64，否则 pip 装不了轮子）"
if ($tag.Trim() -ne 'win-amd64') { throw "解释器平台标签异常: $tag" }

# ---------- 2. 建虚拟环境 ----------
if ((Test-Path $VenvPy) -and -not $Force) {
    Write-Host "复用已有虚拟环境: $VenvPy"
} else {
    Write-Host "创建虚拟环境: $Root\.venv"
    & $PyExe -m venv (Join-Path $Root '.venv')
    if ($LASTEXITCODE -ne 0) { throw "venv 创建失败" }
}
& $VenvPy -V

# ---------- 3. 安装依赖 ----------
Write-Host "升级 pip ..."
& $VenvPy -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "pip 升级失败" }

$Req = Join-Path $Root 'requirements.txt'
Write-Host "安装依赖（首次约 250MB，请耐心等待）..."
& $VenvPy -m pip install -r $Req
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

# ---------- 4. 自检 ----------
Write-Host "`n===== 环境自检 ====="
$Check = @'
import platform, sys
print('python  :', sys.version.split()[0], '|', platform.machine())
import numpy; print('numpy   :', numpy.__version__)
import torch; print('torch   :', torch.__version__)
print('cuda    :', torch.cuda.is_available(), '(本工程按 CPU 设计)')
print('threads :', torch.get_num_threads())
try:
    import matplotlib; print('mpl     :', matplotlib.__version__)
except Exception:
    print('mpl     : 未安装（画图会自动跳过）')

# 一次极小的真实计算：确认前向 + 反向 + 优化器都可用
torch.manual_seed(0)
x = torch.randn(4, 3)
w = torch.randn(3, 2, requires_grad=True)
loss = ((x @ w) ** 2).mean()
loss.backward()
assert w.grad is not None and w.grad.shape == (3, 2)
print('autograd:', 'ok')
print('SELFTEST: PASS')
'@
$TmpPy = Join-Path $env:TEMP 'dsh_env_selftest.py'
Set-Content -LiteralPath $TmpPy -Value $Check -Encoding UTF8
& $VenvPy $TmpPy
if ($LASTEXITCODE -ne 0) { throw "环境自检失败" }
Remove-Item -LiteralPath $TmpPy -Force -ErrorAction SilentlyContinue

Write-Host "`n环境就绪。接下来："
Write-Host "  pwsh -File scripts\download_data.ps1"
Write-Host "  .\.venv\Scripts\python.exe scripts\prepare_data.py"
Write-Host "  .\.venv\Scripts\python.exe src\01_tensor_basics.py"
