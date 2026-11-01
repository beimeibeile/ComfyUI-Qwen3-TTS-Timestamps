@echo off
chcp 936 >nul
set PY=D:\Ai\ComfyUI-aki-v3.2\python\python.exe
set DIR=%~dp0
cd /d %DIR%server
echo [check] port 8001 ...
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo [fail] 8001 already in use, service may be running.
  pause
  exit /b 1
)
start "Qwen3TTS-Server" /min "%PY%" "%DIR%server\tts_server.py"
echo [ok] started: http://127.0.0.1:8001/health
pause