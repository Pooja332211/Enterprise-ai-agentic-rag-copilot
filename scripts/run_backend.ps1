# Run FastAPI backend locally
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = "$ProjectRoot"

if (Test-Path "$ProjectRoot\.venv\Scripts\Activate.ps1") {
    & "$ProjectRoot\.venv\Scripts\Activate.ps1"
}

Write-Host "==> Starting FastAPI server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
