# Multi-Agent System Startup Script

Write-Host "🚀 Starting Multi-Agent System..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check if Ollama is running
Write-Host "🔍 Checking Ollama service..." -ForegroundColor Yellow
$ollamaRunning = $false
try {
    $ollamaResponse = Invoke-RestMethod -Uri "http://192.168.50.20:11434/api/version" -Method GET -TimeoutSec 5
    Write-Host "✓ Ollama is running" -ForegroundColor Green
    $ollamaRunning = $true
} catch {
    Write-Host "⚠️  Ollama service not detected at localhost:11434" -ForegroundColor Yellow
    Write-Host "   Attempting to start Ollama..." -ForegroundColor Yellow

    try {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3

        # Check again
        $ollamaResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 5
        Write-Host "✓ Ollama started successfully" -ForegroundColor Green
        $ollamaRunning = $true
    } catch {
        Write-Host "❌ Could not start Ollama. Please install Ollama from https://ollama.ai/" -ForegroundColor Red
        Write-Host "   After installation, run: ollama pull llama2" -ForegroundColor Yellow
        exit 1
    }
}

# Check if virtual environment exists
if (Test-Path "venv\Scripts\activate.ps1") {
    Write-Host "✓ Virtual environment found" -ForegroundColor Green
    & "venv\Scripts\activate.ps1"
} else {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    & "venv\Scripts\activate.ps1"
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Install dependencies
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
try {
    pip install --upgrade pip
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Write-Host "   Please check requirements.txt and try again" -ForegroundColor Yellow
    exit 1
}

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
$directories = @(
    "dashboard\static",
    "dashboard\templates",
    "logs",
    "data"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✓ Created directory: $dir" -ForegroundColor Green
    }
}

# Copy environment file if it doesn't exist
if (!(Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✓ Created .env file from .env.example" -ForegroundColor Green
        Write-Host "   Please review and update .env file with your settings" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  No .env.example file found" -ForegroundColor Yellow
    }
}

# Start the application
Write-Host "" 
Write-Host "🎯 Starting Multi-Agent System..." -ForegroundColor Cyan
Write-Host "   Dashboard will be available at: http://localhost:8000" -ForegroundColor White
Write-Host "   Press Ctrl+C to stop the system" -ForegroundColor White
Write-Host ""

try {
    python main.py
} catch {
    Write-Host "❌ Failed to start the application" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "👋 Multi-Agent System stopped." -ForegroundColor Yellow