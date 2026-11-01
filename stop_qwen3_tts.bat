@echo off
chcp 936 >nul
powershell -NoProfile -Command " = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue; if () { Stop-Process -Id [0].OwningProcess -Force; Write-Host ('[ok] stopped PID ' + [0].OwningProcess) } else { Write-Host '[info] 8001 not listening' }"
pause