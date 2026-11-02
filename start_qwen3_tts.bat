@echo off
chcp 936 >nul
rem 指定 python.exe(需含 fastapi/uvicorn/soundfile/requests 等依赖)。
rem 优先级: 环境变量 QWEN3_TTS_PY > PATH 中的 python。
rem 示例(写入系统环境变量后即生效):
rem   set QWEN3_TTS_PY=C:\path\to\python.exe
set "PY=python"
if defined QWEN3_TTS_PY set "PY=%QWEN3_TTS_PY%"
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