"""
Step 5 — Train three models on historical NBA data.

Reads:  data/processed/features.csv
Writes: models/moneyline.pkl        (logistic regression -> P(home win))
        models/margin.pkl           (gradient boosting   -> home margin)
        models/total.pkl            (gradient boosting   -> combined points)
        models/feature_columns.json (list of columns the models expect)

Split strategy:
  Train: all games before 2025-10-01  (seasons 2021-22 .. 2024-25)
  Test:  games on/after  2025-10-01  (season 2025-26)

Run:
  python src\\train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, mean_absolute_error, brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "features.csv"
FEATURES_JSON = MODELS_DIR / "feature_columns.json"


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TEST_START_DATE = "2025-10-01"

FEATURE_COLUMNS = [
    # Elo
    "home_elo_pre", "away_elo_pre", "elo_diff_pre", "elo_home_win_prob_pre",
    # Rest
    "home_rest_days", "away_rest_days", "rest_diff", "home_b2b", "away_b2b",
    # Season-to-date
    "home_season_win_pct", "away_season_win_pct",
    "home_home_win_pct", "away_away_win_pct",
    # Rolling form
    "home_win_pct_last5", "away_win_pct_last5",
    "home_win_pct_last10", "away_win_pct_last10",
    "home_avg_margin_last5", "away_avg_margin_last5",
    "home_avg_pts_for_last5", "away_avg_pts_for_last5",
    "home_avg_pts_against_last5", "away_avg_pts_against_last5",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, cutoff: str):
    train = df[df["game_date"] < cutoff].copy()
    test = df[df["game_date"] >= cutoff].copy()
    return train, test


def moneyline_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def regressor_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg", GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Loading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["game_date"], dtype={"game_id": str})
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  rows: {len(df)}")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: missing feature columns in features.csv: {missing}")
        return 1

    train, test = time_split(df, TEST_START_DATE)
    print(f"\nTrain: {len(train)} games  "
          f"({train['game_date'].min().date()} -> {train['game_date'].max().date()})")
    print(f"Test:  {len(test)} games   "
          f"({test['game_date'].min().date()} -> {test['game_date'].max().date()})")

    if len(test) == 0:
        print("ERROR: test set is empty. Check TEST_START_DATE.")
        return 1

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    # ===================================================================
    # 1) MONEYLINE
    # ===================================================================
    print("\n" + "=" * 60)
    print("MONEYLINE (predict P(home win))")
    print("=" * 60)

    y_train = train["home_win"].astype(int).values
    y_test = test["home_win"].astype(int).values

    ml_model = moneyline_pipeline()
    ml_model.fit(X_train, y_train)

    proba = ml_model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, proba, labels=[0, 1])
    brier = brier_score_loss(y_test, proba)

    baseline_home = y_test.mean()
    baseline_acc = max(baseline_home, 1 - baseline_home)

    print(f"  test games:            {len(y_test)}")
    print(f"  home win rate (test):  {baseline_home:.3f}")
    print(f"  always-home baseline:  {baseline_acc:.3f}")
    print(f"  model accuracy:        {acc:.3f}")
    print(f"  log loss:              {ll:.4f}")
    print(f"  Brier score:           {brier:.4f}")

    # Calibration: how confident is the model when it's right?
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() > 20:
            w = (preds[mask] == y_test[mask]).mean()
            print(f"  when P(home) in [{lo:.1f},{hi:.1f}): "
                  f"n={mask.sum():4d}  actual acc={w:.3f}")

    # ===================================================================
    # 2) MARGIN (home point differential)
    # ===================================================================
    print("\n" + "=" * 60)
    print("MARGIN (predict home point differential)")
    print("=" * 60)

    margin_model = regressor_pipeline()
    margin_model.fit(X_train, train["margin"].values)

    margin_preds = margin_model.predict(X_test)
    margin_mae = mean_absolute_error(test["margin"].values, margin_preds)
    margin_bias = float(np.mean(margin_preds - test["margin"].values))

    # Sign accuracy: did we get the winner right?
    margin_sign_acc = float(
        ((margin_preds >= 0).astype(int) == y_test).mean()
    )
    print(f"  MAE:                {margin_mae:.2f} points")
    print(f"  bias (pred-actual): {margin_bias:+.2f}")
    print(f"  sign accuracy:      {margin_sign_acc:.3f}")

    # ===================================================================
    # 3) TOTAL (combined points)
    # ===================================================================
    print("\n" + "=" * 60)
    print("TOTAL (predict combined points)")
    print("=" * 60)

    total_model = regressor_pipeline()
    total_model.fit(X_train, train["total_pts"].values)

    total_preds = total_model.predict(X_test)
    total_mae = mean_absolute_error(test["total_pts"].values, total_preds)
    total_bias = float(np.mean(total_preds - test["total_pts"].values))

    # Accuracy if you just predicted "over" or "under" the market's typical 220.5
    typical_line = 220.5
    over_actual = (test["total_pts"].values > typical_line).astype(int)
    over_pred = (total_preds > typical_line).astype(int)
    over_acc = float((over_actual == over_pred).mean())

    print(f"  MAE:                {total_mae:.2f} points")
    print(f"  bias (pred-actual): {total_bias:+.2f}")
    print(f"  Over/Under 220.5 accuracy: {over_acc:.3f}")

    # ===================================================================
    # SAVE
    # ===================================================================
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    joblib.dump(ml_model,     MODELS_DIR / "moneyline.pkl")
    joblib.dump(margin_model, MODELS_DIR / "margin.pkl")
    joblib.dump(total_model,  MODELS_DIR / "total.pkl")

    with open(FEATURES_JSON, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"  wrote {MODELS_DIR / 'moneyline.pkl'}")
    print(f"  wrote {MODELS_DIR / 'margin.pkl'}")
    print(f"  wrote {MODELS_DIR / 'total.pkl'}")
    print(f"  wrote {FEATURES_JSON}")

    # Quick top-of-the-list feature importances for margin (tree model)
    print("\nTop-10 margin-model feature importances:")
    importances = margin_model.named_steps["reg"].feature_importances_
    order = np.argsort(importances)[::-1][:10]
    for i in order:
        print(f"  {FEATURE_COLUMNS[i]:32s} {importances[i]:.4f}")

    print("\n[done] Step 5 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())