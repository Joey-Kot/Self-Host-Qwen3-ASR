@echo off
cd /d "%~dp0"
set PROVIDER=%~1
if "%PROVIDER%"=="" set PROVIDER=ModelScope
set DOWNLOAD_FAILED=0

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_model.ps1" -Provider "%PROVIDER%"
if errorlevel 1 set DOWNLOAD_FAILED=1

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_model.ps1" -Provider "%PROVIDER%" -ModelId "Qwen/Qwen3-ForcedAligner-0.6B" -ModelDir "%~dp0models\Qwen3-ForcedAligner-0.6B"
if errorlevel 1 set DOWNLOAD_FAILED=1

if "%DOWNLOAD_FAILED%"=="1" echo One or more checkpoint downloads did not complete. Run this script again to retry.
pause
exit /b %DOWNLOAD_FAILED%
