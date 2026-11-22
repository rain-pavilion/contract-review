param(
    [string]$Docx = 'docs\期刊论文-第一页-马宇轩.docx',
    [string]$Pdf  = "$env:TEMP\journal_page1.pdf"
)
$ErrorActionPreference = 'Stop'
function Log($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

if ([IO.Path]::IsPathRooted($Pdf)) { $pdfPath = $Pdf } else { $pdfPath = Join-Path (Get-Location).Path $Pdf }
$docxPath = if ([IO.Path]::IsPathRooted($Docx)) { $Docx } else { Join-Path (Get-Location).Path $Docx }

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.ScreenUpdating = $false
$doc = $null
try {
    Log "打开 $docxPath"
    $doc = $word.Documents.Open($docxPath, $false, $true)
    Log ("页数=" + $doc.ComputeStatistics(2) + " 字数=" + $doc.ComputeStatistics(0))
    $doc.Repaginate()
    if (Test-Path $pdfPath) { Remove-Item $pdfPath -Force }
    $doc.ExportAsFixedFormat($pdfPath, 17)
    Log ("PDF 已导出（" + (Get-Item $pdfPath).Length + " B）→ $pdfPath")
}
finally {
    if ($doc) { $doc.Close(0) }
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
