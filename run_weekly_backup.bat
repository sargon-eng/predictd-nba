@echo off
REM Predictd NBA — weekly project backup (skips .venv and data/raw)
cd /d "%~dp0"
set BACKUP_DIR=%USERPROFILE%\Desktop
set STAMP=%date:~-4%-%date:~-7,2%-%date:~-10,2%
set BACKUP_NAME=PredictdNBA_backup_%STAMP%.zip
powershell -NoProfile -Command "Compress-Archive -Path '*.py','*.bat','*.txt','*.md','src','models','predictions','data\processed' -DestinationPath '%BACKUP_DIR%\%BACKUP_NAME%' -Force -ErrorAction SilentlyContinue"
if exist "%BACKUP_DIR%\%BACKUP_NAME%" (
    echo [%date% %time%] backup OK: %BACKUP_NAME% >> logs\weekly_backup.log
) else (
    echo [%date% %time%] backup FAILED >> logs\weekly_backup.log
    exit /b 1
)
exit /b 0