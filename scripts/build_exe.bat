@echo off
setlocal

set "REPO_ROOT=%~dp0.."

powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\scripts\build_exe.ps1"
if errorlevel 1 (
  echo [build] failed.
  exit /b 1
)

echo [build] done.
exit /b 0
