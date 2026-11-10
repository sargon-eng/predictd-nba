# Predictd NBA

Sports prediction system for NBA games. Companion to the football Predictd project.

## Model performance (2025-26 held-out)

- Moneyline: 66.9% (baseline 55.5%)
- Spread sign: 65.0% (MAE 11.95)
- Total O/U 220.5: 65.5% (MAE 15.72)

## Quick commands

Run predictions: run_daily_predictions.bat
Update results: run_accuracy_check.bat
Retrain models: run_weekly_retrain.bat
Backup: run_weekly_backup.bat

Dashboard: streamlit run dashboard.py

## Scheduled tasks

- PredictdNBA_Daily_Odds      -> 19:00 daily
- PredictdNBA_Accuracy_Check -> 20:00 daily
- PredictdNBA_Weekly_Retrain -> 18:00 Mondays
- PredictdNBA_Weekly_Backup  -> 05:00 Sundays

## Known limitations

- No injury data
- No trade/roster data
- Early-season predictions are AWAY-biased until ~10 games played per team

See full docs in src/*.py docstrings.
