[CmdletBinding()]
param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    & $venvPython -m pip install --cache-dir .runtime/pip-cache -c tools/dev-constraints.txt -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Development dependency installation failed.' }
    & $venvPython tools/dev.py doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }
    & $venvPython tools/dev.py qa smoke
    if ($LASTEXITCODE -ne 0) { throw 'Bootstrap QA failed.' }
} finally {
    Pop-Location
}
