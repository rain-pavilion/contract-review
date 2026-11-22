# 论文一键重建与验收（Windows PowerShell 5.1 可用）
#
# 用法：
#   & tools\rebuild_thesis.ps1                # 重建 + 全部静态验收（不调 Word）
#   & tools\rebuild_thesis.ps1 -WithWord      # 另外调用 Word 更新目录页码并导出 PDF
#   & tools\rebuild_thesis.ps1 -SkipBuild     # 只跑验收（已手工改过 docx 时用）
#
# 注意：本文件必须以 UTF-8 with BOM 保存。Windows PowerShell 5.1 若无 BOM 会按 GBK
# 解码，中文字符串会乱码并报"表达式或语句中包含意外的标记"。

param(
    [switch]$WithWord,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 优先用 DSH 运行时自带的 Python（依赖齐全），否则退回 PATH 里的 python
$py = 'C:\Users\ROG\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$env:PYTHONIOENCODING = 'utf-8'

function Step($name, $script, $args) {
    Write-Host ''
    Write-Host "=== $name ===" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $py $script @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[×] $name 失败（exit=$LASTEXITCODE）" -ForegroundColor Red
        return $false
    }
    Write-Host "[✓] $name" -ForegroundColor Green
    return $true
}

if (-not $SkipBuild) {
    if (-not (Step '重建 docx' 'tools\build_thesis_v2.py')) { exit 1 }
    if (-not (Step '结构自检' 'tools\verify_thesis_v2.py')) { exit 1 }
}

Step '排版审计' 'tools\audit_thesis_format.py' | Out-Null
Step 'AI 文风探针' 'tools\ai_style_probe.py' | Out-Null

if ($WithWord) {
    Step '修兼容模式与安全标记' 'tools\fix_thesis_toc.py' @('--no-word') | Out-Null
    Write-Host ''
    Write-Host '=== Word 更新目录页码 ===' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'update_toc.ps1')
    if ($LASTEXITCODE -ne 0) { Write-Host '[×] Word 更新目录失败（确认已装 Word 且无残留 WINWORD 进程）' -ForegroundColor Red }
}

Write-Host ''
Write-Host '完成。产物：docs\毕业论文-智律契灵合同智能审查系统v2.docx' -ForegroundColor Green
