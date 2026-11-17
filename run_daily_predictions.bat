@echo off
REM Predictd NBA - daily odds fetch + prediction
cd /d "%~dp0"
call .venv\Scripts\activate.bat

python src\fetch_odds.py
if errorlevel 1 (
    echo [%date% %time%] FAILED (fetch_odds) with code %errorlevel% >> logs\daily_predictions.log
    exit /b 1
)

python src\predict_live.py
if errorlevel 1 (
    echo [%date% %time%] FAILED (predict_live) with code %errorlevel% >> logs\daily_predictions.log
    exit /b 1
)

python src\bet_slip_builder.py
if errorlevel 1 (
    echo [%date% %time%] FAILED (bet_slip_builder) with code %errorlevel% >> logs\daily_predictions.log
    exit /b 1
)

echo [%date% %time%] daily predictions OK >> logs\daily_predictions.log
exit /b 0