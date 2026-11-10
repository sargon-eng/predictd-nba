"""
Step 9 — Adaptive monitor.

Reads:  predictions/tracked.csv
Writes: logs/monitor_YYYY-MM-DD.log

Checks:
  - Overall accuracy vs baselines
  - Rolling last-N ML accuracy (drift alert)
  - Calibration: does high-confidence == high accuracy?
  - Bookmaker-level win rates
  - Cold streaks

Run:
  python src\\adaptive_monitor.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "predictions"
TRACKED_PATH = PRED_DIR / "tracked.csv"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------------
EXPECTED_ML_ACC = 0.65
EXPECTED_SPREAD_ACC = 0.52
EXPECTED_TOTAL_ACC = 0.55

ML_ROLLING_WINDOW = 30
ML_DRIFT_ALERT = 0.60
COLD_STREAK_ALERT = 5
MIN_PICKS_FOR_ALERT = 20


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_tracked() -> pd.DataFrame:
    if not TRACKED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRACKED_PATH)
    if "resolved" in df.columns:
        df = df[df["resolved"] == True].copy()  # noqa: E712
    if "commence_time" in df.columns:
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
        df = df.sort_values("commence_time").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------
def rolling_accuracy(series: pd.Series, window: int) -> float:
    recent = series.dropna().tail(window)
    if len(recent) < 5:
        return np.nan
    return float(recent.mean())


def longest_loss_streak(series: pd.Series) -> int:
    s = series.dropna().tolist()
    streak = 0
    for v in reversed(s):
        if v == 0:
            streak += 1
        else:
            break
    return streak


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    if "model_home_win_prob" not in df.columns or "ml_correct" not in df.columns:
        return pd.DataFrame()

    d = df[df["ml_pick"].isin(["HOME", "AWAY"])].copy()
    if d.empty:
        return pd.DataFrame()

    def conf(row):
        if row["ml_pick"] == "HOME":
            return row["model_home_win_prob"]
        else:
            return 1 - row["model_home_win_prob"]
    d["confidence"] = d.apply(conf, axis=1)

    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]
    labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65",
              "0.65-0.70", "0.70-0.80", "0.80-1.00"]
    d["bucket"] = pd.cut(d["confidence"], bins=bins, labels=labels, right=False)

    rows = []
    for label, g in d.groupby("bucket", observed=True):
        if len(g) == 0:
            continue
        rows.append({
            "confidence_bucket": label,
            "n": len(g),
            "expected_acc": round(float(g["confidence"].mean()), 3),
            "actual_acc":   round(float(g["ml_correct"].mean()), 3),
            "gap":          round(float(g["ml_correct"].mean() - g["confidence"].mean()), 3),
        })
    return pd.DataFrame(rows)


def bookmaker_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "bookmaker" not in df.columns or "ml_correct" not in df.columns:
        return pd.DataFrame()
    d = df[df["ml_pick"].isin(["HOME", "AWAY"])]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for book, g in d.groupby("bookmaker"):
        rows.append({
            "bookmaker": book,
            "n": len(g),
            "ml_accuracy": round(float(g["ml_correct"].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame) -> tuple[str, list]:
    lines: list = []
    alerts: list = []

    def log(s=""):
        lines.append(s)
        print(s)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    log("=" * 72)
    log(f"Predictd NBA — Adaptive Monitor   {now}")
    log("=" * 72)

    if df.empty:
        log("")
        log("No resolved games yet. Tracked file empty or missing.")
        log("Run `python src\\track_accuracy.py` after some games have finished.")
        return "\n".join(lines), alerts

    log("")
    log(f"Resolved games: {len(df)}")
    if "commence_time" in df.columns:
        log(f"Date range: {df['commence_time'].min().date()} -> "
            f"{df['commence_time'].max().date()}")

    # ---------- ML ----------
    log("")
    log("-" * 72)
    log("MONEYLINE")
    log("-" * 72)

    ml = df[df["ml_pick"].isin(["HOME", "AWAY"])].copy()
    if ml.empty:
        log("No ML picks yet.")
    else:
        acc = ml["ml_correct"].mean()
        n = len(ml)
        log(f"  picks:        {n}")
        log(f"  accuracy:     {acc:.3f}   (baseline expectation {EXPECTED_ML_ACC:.2f})")
        log(f"  home picks:   {int((ml['ml_pick'] == 'HOME').sum())}")
        log(f"  away picks:   {int((ml['ml_pick'] == 'AWAY').sum())}")

        for side in ["HOME", "AWAY"]:
            s = ml[ml["ml_pick"] == side]
            if len(s) == 0:
                continue
            log(f"    {side:5s} {int(s['ml_correct'].sum()):3d}/{len(s):3d}  "
                f"({s['ml_correct'].mean():.3f})")

        roll = rolling_accuracy(ml["ml_correct"], ML_ROLLING_WINDOW)
        if not np.isnan(roll):
            log(f"  last {ML_ROLLING_WINDOW} accuracy: {roll:.3f}")
            if n >= MIN_PICKS_FOR_ALERT and roll < ML_DRIFT_ALERT:
                alerts.append(
                    f"ML drift: last {ML_ROLLING_WINDOW} accuracy {roll:.3f} "
                    f"is below {ML_DRIFT_ALERT:.2f}"
                )

        streak = longest_loss_streak(ml["ml_correct"])
        if streak >= COLD_STREAK_ALERT:
            alerts.append(f"ML cold streak: {streak} consecutive losses")

        cal = calibration_table(df)
        if not cal.empty:
            log("")
            log("  Calibration:")
            log(f"  {'bucket':14s} {'n':>5s} {'expected':>10s} {'actual':>8s} {'gap':>8s}")
            for _, r in cal.iterrows():
                gap_str = f"{r['gap']:+.3f}"
                flag = "  <-- overconfident" if r["gap"] < -0.10 else ""
                log(f"  {r['confidence_bucket']:14s} {r['n']:5d} "
                    f"{r['expected_acc']:>10.3f} {r['actual_acc']:>8.3f} "
                    f"{gap_str:>8s}{flag}")
                if r["n"] >= 15 and r["gap"] < -0.10:
                    alerts.append(
                        f"ML overconfidence in bin {r['confidence_bucket']}: "
                        f"expected {r['expected_acc']:.2f}, got {r['actual_acc']:.2f}"
                    )

    # ---------- Spread ----------
    log("")
    log("-" * 72)
    log("SPREAD")
    log("-" * 72)
    sp = df[df["spread_pick"].isin(["HOME", "AWAY"])]
    if sp.empty:
        log("No spread picks yet.")
    else:
        log(f"  picks:        {len(sp)}")
        log(f"  accuracy:     {sp['spread_correct'].mean():.3f} "
            f"  (baseline {EXPECTED_SPREAD_ACC:.2f})")
        roll = rolling_accuracy(sp["spread_correct"], ML_ROLLING_WINDOW)
        if not np.isnan(roll):
            log(f"  last {ML_ROLLING_WINDOW} accuracy: {roll:.3f}")

    # ---------- Total ----------
    log("")
    log("-" * 72)
    log("TOTAL")
    log("-" * 72)
    tot = df[df["total_pick"].isin(["OVER", "UNDER"])]
    if tot.empty:
        log("No total picks yet.")
    else:
        log(f"  picks:        {len(tot)}")
        log(f"  accuracy:     {tot['total_correct'].mean():.3f} "
            f"  (baseline {EXPECTED_TOTAL_ACC:.2f})")
        roll = rolling_accuracy(tot["total_correct"], ML_ROLLING_WINDOW)
        if not np.isnan(roll):
            log(f"  last {ML_ROLLING_WINDOW} accuracy: {roll:.3f}")

    # ---------- Bookmakers ----------
    books = bookmaker_stats(df)
    if not books.empty:
        log("")
        log("-" * 72)
        log("BY BOOKMAKER (moneyline)")
        log("-" * 72)
        for _, r in books.iterrows():
            log(f"  {r['bookmaker']:12s} n={int(r['n']):3d}  "
                f"acc={r['ml_accuracy']:.3f}")

    # ---------- Alerts ----------
    log("")
    log("=" * 72)
    if alerts:
        log(f"ALERTS ({len(alerts)})")
        log("=" * 72)
        for a in alerts:
            log(f"  [!] {a}")
    else:
        log("No alerts — everything within expected range.")
    log("=" * 72)

    return "\n".join(lines), alerts


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print("Loading tracked predictions ...")
    df = load_tracked()
    print(f"  {len(df)} resolved rows")

    report, alerts = build_report(df)

    out_path = LOGS_DIR / f"monitor_{datetime.now().strftime('%Y-%m-%d')}.log"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    print(f"[done] Step 9 complete. {len(alerts)} alert(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())