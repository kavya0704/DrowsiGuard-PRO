@echo off
title AI Driver Drowsiness Detection System
color 0B
echo ============================================================
echo    AI Driver Drowsiness Detection System - DrowsiGuard
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install.
    echo Trying with --user flag...
    pip install -r requirements.txt --quiet --user
)

echo.
echo [2/3] Generating alarm sound...
python generate_alarm.py

echo.
echo [3/3] Starting DrowsiGuard server...
echo.
echo ============================================================
echo   Dashboard:  http://localhost:5000
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.

:: Open browser after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Start the Flask app
python app.py

pause
