@echo off
REM Predictd NBA — daily odds fetch + prediction
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python src\fetch_odds.py
if errorlevel 1 goto :error
python src\predict_live.py
if errorlevel 1 goto :error
echo [%date% %time%] daily predictions OK >> logs\daily_predictions.log
exit /b 0
:error
echo [%date% %time%] FAILED with code %errorlevel% >> logs\daily_predictions.log
exit /b 1