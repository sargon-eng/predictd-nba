@echo off
REM Predictd NBA — weekly feature rebuild + model retrain
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python src\build_features.py
if errorlevel 1 goto :error
python src\train_models.py
if errorlevel 1 goto :error
echo [%date% %time%] weekly retrain OK >> logs\weekly_retrain.log
exit /b 0
:error
echo [%date% %time%] FAILED with code %errorlevel% >> logs\weekly_retrain.log
exit /b 1