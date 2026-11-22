# 用 Word COM 更新论文目录页码并导出 PDF 预览（分步打时间戳，便于定位卡点）
# 用法：pwsh -File tools/update_toc.ps1 [-Docx path] [-Pdf path]
param(
    [string]$Docx = 'docs\毕业论文-智律契灵合同智能审查系统v2.docx',
    [string]$Pdf  = "$env:TEMP\thesis_preview.pdf"
)
$ErrorActionPreference = 'Stop'
function Log($m) { Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $m) }

$docx = (Resolve-Path $Docx).Path
if ([IO.Path]::IsPathRooted($Pdf)) { $pdfPath = $Pdf } else { $pdfPath = Join-Path (Get-Location).Path $Pdf }
Log "目标 docx: $docx"
Log "目标 pdf : $pdfPath"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try { $word.ScreenUpdating = $false } catch {}
Log "Word 已启动（Version=$($word.Version)）"

try {
    $d = $word.Documents.Open($docx, $false, $false)   # ConfirmConversions=false, ReadOnly=false
    Log "已打开，兼容模式=$($d.CompatibilityMode) 节数=$($d.Sections.Count)"

    if ($d.TablesOfContents.Count -gt 0) {
        Log "开始更新目录（$($d.TablesOfContents.Count) 个）…"
        $d.TablesOfContents.Item(1).Update()
        Log "目录已更新"
    } else {
        Log "文档里没有 TOC 域，跳过"
    }

    $d.Repaginate()
    Log "重排完成：页数=$($d.ComputeStatistics(2)) 字数=$($d.ComputeStatistics(0)) 表=$($d.Tables.Count) 图=$($d.InlineShapes.Count)"
    $d.Save()
    Log "已保存"

    $d.ExportAsFixedFormat($pdfPath, 17)   # wdExportFormatPDF
    Log "PDF 已导出（$((Get-Item $pdfPath).Length) B）"
}
finally {
    if ($d) { $d.Close(0); Log "文档已关闭" }
    $word.Quit()
    [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    Log "Word 已退出"
}
