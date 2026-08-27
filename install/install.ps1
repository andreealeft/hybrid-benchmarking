# Install the tool and put an icon on the Desktop.
#
#   irm https://raw.githubusercontent.com/andreealeft/hybrid-benchmarking/main/install/install.ps1 | iex
#
# Everything lands in one folder of its own, so nothing else on the computer is
# touched and uninstalling is deleting that folder and the icon.

$ErrorActionPreference = "Stop"

$HomeDir = Join-Path $env:LOCALAPPDATA "hybrid-benchmarking"
$Venv    = Join-Path $HomeDir "venv"
$Source  = "https://github.com/andreealeft/hybrid-benchmarking/archive/refs/heads/main.zip"

Write-Host ""
Write-Host "  Hybrid benchmarking"
Write-Host "  ==================="
Write-Host ""

# ---------------------------------------------------------------- Python
$py = $null
foreach ($candidate in @("py", "python")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) { $py = $candidate; break }
}
if (-not $py) {
    Write-Host "  This needs Python, which is not on this computer yet."
    Write-Host "  Opening the download page. Install it, ticking Add Python to PATH,"
    Write-Host "  then run this again."
    Start-Process "https://www.python.org/downloads/windows/"
    return
}

# ---------------------------------------------------------------- the tool
Write-Host "  Installing into a folder of its own. This takes a minute."
New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $py -m venv $Venv
}
$venvPython = Join-Path $Venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade --quiet pip
& $venvPython -m pip install --upgrade --quiet $Source

# ---------------------------------------------------------------- the icon
# pythonw runs without a console window, so the icon behaves like an app rather
# than opening a black box of text.
$shortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Hybrid benchmarking.lnk"
$link = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
$link.TargetPath       = Join-Path $Venv "Scripts\pythonw.exe"
$link.Arguments        = "-m hybrid_benchmarking.cli open"
$link.WorkingDirectory = $HomeDir
$link.Description      = "Resource estimates for quantum algorithms"
$link.Save()

Write-Host ""
Write-Host "  Done. There is now an icon on your Desktop called Hybrid benchmarking."
Write-Host "  Double-click it whenever you want the tool: it opens in your browser."
Write-Host "  It keeps itself up to date, so this is the last time you need a terminal."
Write-Host ""
Write-Host "  Opening it now."
& (Join-Path $Venv "Scripts\hybrid-benchmarking.exe") open
