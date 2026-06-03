$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".conda-v2s\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Python 3.10 environment was not found at .conda-v2s. Create it first with: conda create --solver classic -p .\.conda-v2s python=3.10 -y"
}

Set-Location $root
& $python app_gui.py
