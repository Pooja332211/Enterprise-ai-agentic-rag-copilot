# Setup virtual environment and dependencies for Windows
$ErrorActionPreference = "Stop"

Write-Host "==> Initializing Enterprise AI Copilot environment..." -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing backend dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

Write-Host "==> Setup complete! Run .\scripts\run_backend.ps1 to start the backend." -ForegroundColor Green
