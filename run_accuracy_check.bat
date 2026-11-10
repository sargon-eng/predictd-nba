@echo off
REM Predictd NBA — daily accuracy + monitor
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python src\track_accuracy.py
python src\adaptive_monitor.py
if errorlevel 1 goto :error
echo [%date% %time%] accuracy check OK >> logs\accuracy_check.log
exit /b 0
:error
echo [%date% %time%] FAILED with code %errorlevel% >> logs\accuracy_check.log
exit /b 1