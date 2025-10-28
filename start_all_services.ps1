Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "PX Score Platform Startup Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installing API dependencies..." -ForegroundColor Yellow
pip install -q -r requirements_api.txt
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Starting Services..." -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Starting API Server (Port 5000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python conversation_api.py"
Start-Sleep -Seconds 3
Write-Host "2. Starting Conversation Analyzer (Port 7860)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python src\models\app.py"
Start-Sleep -Seconds 3
Write-Host "3. Starting PX Score Dashboard (Port 7861)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python main_app.py"
Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "All Services Started!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access the applications at:" -ForegroundColor Cyan
Write-Host "  API Server:              http://localhost:5000" -ForegroundColor White
Write-Host "  Conversation Analyzer:   http://localhost:7860" -ForegroundColor White
Write-Host "  PX Score Dashboard:      http://localhost:7861" -ForegroundColor White
Write-Host ""
