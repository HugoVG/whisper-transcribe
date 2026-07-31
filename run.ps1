# Whisper Transcribe — Windows quick-start
# 1. Copy .env.example to .env and set your HF_TOKEN
# 2. Run this script to build and start the API server

param(
    [switch]$Build,
    [switch]$Down,
    [switch]$Logs
)

$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "No .env found — copy .env.example to .env and set your HF_TOKEN" -ForegroundColor Yellow
    Copy-Item (Join-Path $PSScriptRoot ".env.example") $envFile
}

if ($Down) {
    docker compose down
    exit 0
}

if ($Build) {
    Write-Host "Building Whisper image (~20 min on first run)..." -ForegroundColor Cyan
    docker compose build
}

if ($Logs) {
    docker compose logs -f
    exit 0
}

Write-Host "Starting Whisper Transcribe API on http://localhost:8000" -ForegroundColor Green
Write-Host "Test: curl http://localhost:8000/health" -ForegroundColor Gray
docker compose up -d
