# Project startup script
# Run this script in terminal to start the project
# Press Ctrl+C to exit and stop all services

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = $scriptDir

Write-Host "Project directory: $projectDir" -ForegroundColor Green

# Check if virtual environment exists
$venvPath = Join-Path $projectDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvPath)) {
    Write-Host "Error: Virtual environment does not exist. Please create it first." -ForegroundColor Red
    Write-Host "Recommended: python -m venv .venv" -ForegroundColor Yellow
    return
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
. $venvPath

# Start backend service
Write-Host "Starting backend service..." -ForegroundColor Green
$backendProcess = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" -WorkingDirectory $projectDir -PassThru

# Wait for backend service to start
Write-Host "Waiting for backend service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Start frontend service
Write-Host "Starting frontend service..." -ForegroundColor Green
$frontendProcess = Start-Process -FilePath "python" -ArgumentList "frontend_server.py" -WorkingDirectory $projectDir -PassThru

# Show service status
Write-Host "Project started successfully!" -ForegroundColor Cyan
Write-Host "Backend service: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend service: http://localhost:3000" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to exit and stop all services" -ForegroundColor Yellow

# Wait for user input or Ctrl+C
try {
    # Infinite loop until user presses Ctrl+C
    while ($true) {
        Start-Sleep -Seconds 1
    }
} Catch [System.Management.Automation.RuntimeException] {
    # Catch Ctrl+C exception
    Write-Host "" -ForegroundColor Cyan
    Write-Host "Stopping services..." -ForegroundColor Yellow
    
    # Stop frontend service
    if ($frontendProcess) {
        Write-Host "Stopping frontend service..." -ForegroundColor Green
        $frontendProcess.Kill()
        $frontendProcess.WaitForExit()
    }
    
    # Stop backend service
    if ($backendProcess) {
        Write-Host "Stopping backend service..." -ForegroundColor Green
        $backendProcess.Kill()
        $backendProcess.WaitForExit()
    }
    
    Write-Host "All services stopped" -ForegroundColor Cyan
}