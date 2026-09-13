# =====================================================================
#  下载实验语料
# ---------------------------------------------------------------------
#  用法：
#     powershell -File scripts\download_data.ps1              # 下载全部
#     powershell -File scripts\download_data.ps1 -Only tiny   # 只下 TinyShakespeare
#     powershell -File scripts\download_data.ps1 -Only zh     # 只下中文语料
#     powershell -File scripts\download_data.ps1 -Force       # 重新下载
#
#  下载完成后记得跑一次编码：
#     .\.venv\Scripts\python.exe scripts\prepare_data.py
#
#  中文语料说明（chinese-poetry 仓库，已实测）：
#     元曲/yuanqu.json        单文件 4.2MB，约 11000 条，解析出约 3.5MB 纯文本 —— 首选
#     全唐诗/poet.tang.N.json 按序号分片，每片约 400KB；标题在 title 字段
#     宋词/ci.song.N.json     注意词牌名在 rhythmic 字段，这个仓库里宋词没有 title
#     诗经/楚辞               内容在 content[] 而不是 paragraphs[]
#  若 raw.githubusercontent.com 不通，脚本会自动回退到 jsdelivr 镜像。
# =====================================================================
param(
    [ValidateSet('all', 'tiny', 'zh')]
    [string]$Only = 'all',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$RawDir = Join-Path $Root 'data\raw'
New-Item -ItemType Directory -Force -Path $RawDir | Out-Null

# 两个镜像的路径结构完全一致，raw 失败就换 cdn
$BaseRaw = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/'
$BaseCdn = 'https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/'

$Shakespeare = @{
    Name  = 'tinyshakespeare'
    Urls  = @(
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'https://raw.githubusercontent.com/karpathy/nanoGPT/master/data/shakespeare_char/input.txt'
    )
}

# 中文诗词源清单
#   kind = 'poem'    数组，元素 {title 或 rhythmic, author, paragraphs[]}
#   kind = 'content' 数组，元素 {title, content[]}
$ZhSources = @(
    @{ Path = '%E5%85%83%E6%9B%B2/yuanqu.json';                  Kind = 'poem'; TitleField = 'title'    },
    @{ Path = '%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.0.json';    Kind = 'poem'; TitleField = 'title'    },
    @{ Path = '%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.1000.json'; Kind = 'poem'; TitleField = 'title'    },
    @{ Path = '%E5%AE%8B%E8%AF%8D/ci.song.0.json';               Kind = 'poem'; TitleField = 'rhythmic' },
    @{ Path = '%E8%AF%97%E7%BB%8F/shijing.json';                 Kind = 'content'                        },
    @{ Path = '%E6%A5%9A%E8%BE%9E/chuci.json';                   Kind = 'content'                        }
)

function Get-RemoteFile {
    param([string]$Url, [string]$OutFile)
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -TimeoutSec 180
        return $true
    } catch {
        Write-Host "    失败: $($_.Exception.Message)"
        return $false
    }
}

# 关键：不要用 Invoke-WebRequest 的 .Content 自动解码。
# 它会根据响应头猜编码，中文语料经常被猜错。一律按 UTF-8 自己解码。
function Read-JsonUtf8 {
    param([string]$Path)
    $text = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($Path))
    return $text | ConvertFrom-Json
}

# PowerShell 5.1 的 Set-Content -Encoding UTF8 会写入 BOM，这里显式写无 BOM 的 UTF-8
function Write-TextNoBom {
    param([string]$Path, [string]$Text)
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding($false)))
}

# ---------- 英文语料 ----------
function Get-TinyShakespeare {
    $target = Join-Path $RawDir 'tinyshakespeare.txt'
    if ((Test-Path $target) -and -not $Force) {
        $kb = [math]::Round((Get-Item $target).Length / 1KB, 0)
        Write-Host "[tinyshakespeare] 已存在 (${kb} KB)，跳过。加 -Force 可重新下载。"
        return
    }
    Write-Host "[tinyshakespeare] 开始下载 ..."
    $tmp = Join-Path $RawDir 'tinyshakespeare.download'
    foreach ($url in $Shakespeare.Urls) {
        Write-Host "  尝试: $url"
        if (Get-RemoteFile -Url $url -OutFile $tmp) {
            $text = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($tmp))
            Write-TextNoBom -Path $target -Text $text
            Remove-Item -LiteralPath $tmp -Force
            $kb = [math]::Round((Get-Item $target).Length / 1KB, 0)
            Write-Host "[tinyshakespeare] 完成：${kb} KB"
            return
        }
    }
    Write-Host "[tinyshakespeare] 所有镜像都失败，跳过。"
}

# ---------- 中文语料 ----------
function Get-ChineseCorpus {
    $target = Join-Path $RawDir 'zh_poetry.txt'
    if ((Test-Path $target) -and -not $Force) {
        $kb = [math]::Round((Get-Item $target).Length / 1KB, 0)
        Write-Host "[zh_poetry] 已存在 (${kb} KB)，跳过。加 -Force 可重新下载。"
        return
    }

    Write-Host "[zh_poetry] 开始下载并汇总 ..."
    $sb = New-Object Text.StringBuilder
    $okCount = 0
    $failCount = 0

    foreach ($src in $ZhSources) {
        $name = [IO.Path]::GetFileName($src.Path)
        $tmp = Join-Path $env:TEMP "zhpoetry_$name"
        $downloaded = $false

        foreach ($base in @($BaseRaw, $BaseCdn)) {
            Write-Host "  尝试: $name"
            if (Get-RemoteFile -Url ($base + $src.Path) -OutFile $tmp) { $downloaded = $true; break }
        }
        if (-not $downloaded) {
            Write-Host "    $name 下载失败，跳过。"
            $failCount++
            continue
        }

        try {
            $data = Read-JsonUtf8 -Path $tmp
            $n = 0
            foreach ($item in $data) {
                if ($src.Kind -eq 'poem') {
                    $title = $item.$($src.TitleField)
                    [void]$sb.AppendLine("$title　$($item.author)")
                    foreach ($line in $item.paragraphs) { [void]$sb.AppendLine($line) }
                } else {
                    [void]$sb.AppendLine($item.title)
                    foreach ($line in $item.content) { [void]$sb.AppendLine($line) }
                }
                [void]$sb.AppendLine('')
                $n++
            }
            Write-Host "    解析 $n 条"
            $okCount++
        } catch {
            Write-Host "    解析失败: $($_.Exception.Message)"
            $failCount++
        } finally {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        }
    }

    if ($okCount -eq 0) {
        Write-Host "[zh_poetry] 没有任何一个源成功，跳过。"
        return
    }

    Write-TextNoBom -Path $target -Text $sb.ToString()
    $kb = [math]::Round((Get-Item $target).Length / 1KB, 0)
    Write-Host "[zh_poetry] 完成：${kb} KB（成功 $okCount 个源，失败 $failCount 个）"
}

# ---------- 主流程 ----------
if ($Only -eq 'all' -or $Only -eq 'tiny') { Get-TinyShakespeare }
if ($Only -eq 'all' -or $Only -eq 'zh') { Get-ChineseCorpus }

Write-Host "`n当前 data/raw 内容："
Get-ChildItem -LiteralPath $RawDir -Filter '*.txt' -ErrorAction SilentlyContinue |
    Select-Object Name, @{n = 'KB'; e = { [math]::Round($_.Length / 1KB, 1) } } |
    Format-Table -AutoSize

Write-Host "下一步：.\.venv\Scripts\python.exe scripts\prepare_data.py"
