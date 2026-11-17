@echo off
REM Predictd NBA - daily accuracy + monitor
cd /d "%~dp0"
call .venv\Scripts\activate.bat

python src\track_accuracy.py
if errorlevel 1 (
    echo [%date% %time%] FAILED (track_accuracy) with code %errorlevel% >> logs\accuracy_check.log
    exit /b 1
)

python src\adaptive_monitor.py
if errorlevel 1 (
    echo [%date% %time%] FAILED (adaptive_monitor) with code %errorlevel% >> logs\accuracy_check.log
    exit /b 1
)

echo [%date% %time%] accuracy check OK >> logs\accuracy_check.log
exit /b 0