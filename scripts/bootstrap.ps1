# ============================================================
#  bootstrap.ps1 -- clone corpora, then run the first ingest
#
#  Usage:
#      powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
#
#  ------------------------------------------------------------------
#  IMPORTANT: keep this file ASCII-only.
#
#  This machine runs Windows PowerShell 5.1 (not pwsh 7). PS 5.1
#  decodes a BOM-less .ps1 using the system ANSI code page (936 on a
#  Chinese Windows). Non-ASCII bytes then swallow the following
#  newline, every line number shifts, and the script fails to parse
#  with confusing "Unexpected token" errors.
#
#  Python sources are unaffected: Python 3 always decodes source files
#  as UTF-8, regardless of the console code page. So all Chinese
#  user-facing output lives in the Python layer, not here.
#  ------------------------------------------------------------------

$ErrorActionPreference = 'Continue'
$env:GIT_TERMINAL_PROMPT = '0'   # never hang waiting for credentials

$Root = Split-Path -Parent $PSScriptRoot
$Raw  = Join-Path $Root 'data\raw'
$Out  = Join-Path $Root 'data\processed'

New-Item -ItemType Directory -Force -Path $Raw, $Out | Out-Null

# High-quality Chinese Markdown corpora -- clone, no scraping needed.
$repos = [ordered]@{
    'JavaGuide'       = 'https://github.com/Snailclimb/JavaGuide.git'
    'CS-Notes'        = 'https://github.com/CyC2018/CS-Notes.git'
    'happy-llm'       = 'https://github.com/datawhalechina/happy-llm.git'
    'self-llm'        = 'https://github.com/datawhalechina/self-llm.git'
    'leetcode-master' = 'https://github.com/youngyangyang04/leetcode-master.git'
}

Write-Host ''
Write-Host '=== 1/2  clone corpora ===' -ForegroundColor Cyan
foreach ($repoName in $repos.Keys) {
    $dest = Join-Path $Raw $repoName
    if (Test-Path $dest) {
        Write-Host ("  SKIP  {0}  (already exists)" -f $repoName) -ForegroundColor DarkGray
        continue
    }
    Write-Host ("  CLONE {0} ..." -f $repoName)
    git clone --depth 1 --quiet $repos[$repoName] $dest
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("    OK   {0}" -f $repoName) -ForegroundColor Green
    }
    else {
        Write-Host ("    FAIL {0}  (exit {1})" -f $repoName, $LASTEXITCODE) -ForegroundColor Red
    }
}

Write-Host ''
Write-Host '=== 2/2  ingest ===' -ForegroundColor Cyan
python (Join-Path $Root 'ingest.py') ingest --raw $Raw --out $Out
$code = $LASTEXITCODE

Write-Host ''
if ($code -eq 0) {
    Write-Host ("Done. Stats -> {0}" -f (Join-Path $Out 'stats.json')) -ForegroundColor Yellow
}
else {
    Write-Host ("Ingest failed with exit code {0}" -f $code) -ForegroundColor Red
}
exit $code
