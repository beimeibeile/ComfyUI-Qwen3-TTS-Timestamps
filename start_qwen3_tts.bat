@echo off
chcp 936 >nul
rem Python resolver priority:
rem   1) env var QWEN3_TTS_PY (e.g. set QWEN3_TTS_PY=C:\path\to\python.exe)
rem   2) ComfyUI-embedded python relative to this repo (custom_nodes install: ..\..\..\python\python.exe)
rem   3) python on PATH
set "PY="
if defined QWEN3_TTS_PY set "PY=%QWEN3_TTS_PY%"
if not defined PY if exist "%~dp0..\..\..\python\python.exe" set "PY=%~dp0..\..\..\python\python.exe"
if not defined PY set "PY=python"
set "DIR=%~dp0"
cd /d "%DIR%server"
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