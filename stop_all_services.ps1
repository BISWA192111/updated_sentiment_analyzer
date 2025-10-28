Write-Host '=====================================' -ForegroundColor Red
Write-Host 'Stopping PX Score Platform Services' -ForegroundColor Red
Write-Host '=====================================' -ForegroundColor Red
Write-Host ''
Write-Host 'Stopping all Python processes...' -ForegroundColor Yellow
Get-Process python,pythonw -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host ''
Write-Host '=====================================' -ForegroundColor Green
Write-Host 'All Services Stopped!' -ForegroundColor Green
Write-Host '=====================================' -ForegroundColor Green
Write-Host ''
Write-Host 'All services have been terminated.' -ForegroundColor White
Write-Host 'To restart: .\start_all_services.ps1' -ForegroundColor Cyan
Write-Host ''
