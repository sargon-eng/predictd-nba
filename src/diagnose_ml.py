"""
Diagnose the ML model's live performance.

Reads:  predictions/tracked.csv (resolved games with ML picks)
Writes: nothing (diagnostic only)

Run:
  python src\\diagnose_ml.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_PATH = PROJECT_ROOT / "predictions" / "tracked.csv"


def hr(title: str = "") -> None:
    if title:
        print()
        print("=" * 72)
        print(title)
        print("=" * 72)
    else:
        print("-" * 72)


def main() -> int:
    if not TRACKED_PATH.exists():
        print(f"ERROR: {TRACKED_PATH} not found")
        return 1

    df = pd.read_csv(TRACKED_PATH, dtype={"game_id": str})
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")

    if "resolved" in df.columns:
        df = df[df["resolved"] == True].copy()
    if "ml_pick" not in df.columns or "ml_correct" not in df.columns:
        print("ERROR: tracked.csv missing ml_pick / ml_correct columns")
        return 1

    ml = df[df["ml_pick"].isin(["HOME", "AWAY"])].copy()
    ml["ml_correct"] = ml["ml_correct"].fillna(0).astype(int)
    ml["actual_home_win"] = ml["actual_home_win"].fillna(-1).astype(int)

    # Model probability for the side it picked
    ml["model_pick_prob"] = np.where(
        ml["ml_pick"] == "HOME",
        ml["model_home_win_prob"],
        1.0 - ml["model_home_win_prob"],
    )
    # Market probability for the side the model picked
    ml["market_pick_prob"] = np.where(
        ml["ml_pick"] == "HOME",
        ml["market_home_prob"],
        1.0 - ml["market_home_prob"],
    )
    # Which side did the market favor?
    ml["market_fav"] = np.where(ml["market_home_prob"] > 0.5, "HOME", "AWAY")
    ml["bets_with_market"] = (ml["ml_pick"] == ml["market_fav"])

    n = len(ml)
    correct = int(ml["ml_correct"].sum())
    acc = correct / n if n else 0
    flip_acc = 1.0 - acc  # if you flipped every pick

    # Baseline comparisons
    home_wins = int(ml["actual_home_win"].sum())
    away_wins = n - home_wins
    always_home_acc = home_wins / n if n else 0
    always_away_acc = away_wins / n if n else 0

    # Market favorite accuracy (per-row baseline)
    mkt_fav_correct = int(
        ((ml["market_fav"] == "HOME") & (ml["actual_home_win"] == 1)).sum() +
        ((ml["market_fav"] == "AWAY") & (ml["actual_home_win"] == 0)).sum()
    )
    mkt_fav_acc = mkt_fav_correct / n if n else 0

    # =========================================================
    hr("OVERALL")
    print(f"  total ML picks:      {n}")
    print(f"  model correct:       {correct}  ({acc:.3f})")
    print()
    print("  Baselines (same games):")
    print(f"    flip the model:    {n - correct}  ({flip_acc:.3f})")
    print(f"    always HOME:       {home_wins}  ({always_home_acc:.3f})")
    print(f"    always AWAY:       {away_wins}  ({always_away_acc:.3f})")
    print(f"    always MKT FAV:    {mkt_fav_correct}  ({mkt_fav_acc:.3f})")
    print()
    print(f"  Edge over always-home:  {(acc - always_home_acc)*100:+.1f} pp")
    print(f"  Edge over flip:         {(acc - flip_acc)*100:+.1f} pp")

    # =========================================================
    hr("PICK DISTRIBUTION (HOME vs AWAY)")
    for side in ["HOME", "AWAY"]:
        s = ml[ml["ml_pick"] == side]
        if len(s) == 0:
            print(f"  {side}: 0 picks")
            continue
        w = int(s["ml_correct"].sum())
        print(f"  {side}: {len(s):3d} picks, {w:3d} correct ({w/len(s):.3f})")

    # =========================================================
    hr("MODEL vs MARKET (is the model always picking underdogs?)")
    print("  Market-favorite side vs model-picked side:")
    ct = pd.crosstab(ml["market_fav"], ml["ml_pick"], margins=True, margins_name="TOTAL")
    print(ct.to_string())

    print()
    with_m = ml[ml["bets_with_market"]]
    against_m = ml[~ml["bets_with_market"]]
    if len(with_m) > 0:
        w = int(with_m["ml_correct"].sum())
        print(f"  Model bets WITH market fav:    {w}/{len(with_m)} ({w/len(with_m):.3f})")
    if len(against_m) > 0:
        w = int(against_m["ml_correct"].sum())
        print(f"  Model bets AGAINST market fav: {w}/{len(against_m)} ({w/len(against_m):.3f})")

    # =========================================================
    hr("ACCURACY BY MODEL CONFIDENCE")
    print("  (probability the model assigned to its pick)")
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]
    labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", "0.80-1.00"]
    ml["conf_bucket"] = pd.cut(ml["model_pick_prob"], bins=bins, labels=labels, right=False)
    print(f"  {'bucket':12s} {'picks':>6s} {'wins':>6s} {'acc':>8s}")
    for bucket, grp in ml.groupby("conf_bucket", observed=True):
        if len(grp) == 0:
            continue
        w = int(grp["ml_correct"].sum())
        print(f"  {str(bucket):12s} {len(grp):6d} {w:6d} {w/len(grp):8.3f}")

    # =========================================================
    hr("ACCURACY BY EDGE (model_prob - market_prob on picked side)")
    # ml_edge is already model - market; sign depends on which side
    # Actually ml_edge = home_edge or away_edge in predict_live; for HOME picks it's model_home - mkt_home; for AWAY picks it's model_away - mkt_away
    edge_bins = [0.0, 0.03, 0.05, 0.10, 0.15, 0.20, 1.0]
    edge_labels = ["0-3%", "3-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    ml["edge_bucket"] = pd.cut(ml["ml_edge"].abs(), bins=edge_bins, labels=edge_labels, right=False)
    print(f"  {'edge':8s} {'picks':>6s} {'wins':>6s} {'acc':>8s}")
    for bucket, grp in ml.groupby("edge_bucket", observed=True):
        if len(grp) == 0:
            continue
        w = int(grp["ml_correct"].sum())
        print(f"  {str(bucket):8s} {len(grp):6d} {w:6d} {w/len(grp):8.3f}")

    # =========================================================
    hr("ACCURACY BY WEEK (is the model improving or degrading?)")
    ml["week"] = ml["commence_time"].dt.to_period("W").astype(str)
    print(f"  {'week':14s} {'picks':>6s} {'wins':>6s} {'acc':>8s}")
    for week, grp in ml.sort_values("commence_time").groupby("week", sort=True):
        w = int(grp["ml_correct"].sum())
        print(f"  {week:14s} {len(grp):6d} {w:6d} {w/len(grp):8.3f}")

    # =========================================================
    hr("REGRESSION CHECK (would simple rules beat the model?)")
    print("  Rule: 'only bet when model confidence > X'")
    print(f"  {'threshold':10s} {'picks':>6s} {'wins':>6s} {'acc':>8s} {'coverage':>10s}")
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sub = ml[ml["model_pick_prob"] >= thr]
        if len(sub) == 0:
            print(f"  >= {thr:.2f}   {0:6d}   -      -         0.0%")
            continue
        w = int(sub["ml_correct"].sum())
        cov = len(sub) / n
        print(f"  >= {thr:.2f}   {len(sub):6d} {w:6d} {w/len(sub):8.3f} {cov*100:9.1f}%")

    print()
    print("  Rule: 'flip the model when edge > X%'")
    print(f"  {'threshold':10s} {'picks':>6s} {'wins':>6s} {'acc':>8s} {'coverage':>10s}")
    for thr in [0.03, 0.05, 0.10, 0.15, 0.20]:
        sub = ml[ml["ml_edge"].abs() >= thr]
        if len(sub) == 0:
            continue
        # Flip: pick the OPPOSITE side, evaluate correctness
        w = len(sub) - int(sub["ml_correct"].sum())
        cov = len(sub) / n
        print(f"  >= {thr:.2f}   {len(sub):6d} {w:6d} {w/len(sub):8.3f} {cov*100:9.1f}%")

    # =========================================================
    hr("VERDICT")
    print(f"  Model accuracy:      {acc:.3f}")
    print(f"  Flip-model accuracy: {flip_acc:.3f}")
    print(f"  Always-home:         {always_home_acc:.3f}")
    print(f"  Always-market-fav:   {mkt_fav_acc:.3f}")
    print()
    best_baseline = max(flip_acc, always_home_acc, mkt_fav_acc)
    if acc >= best_baseline:
        print("  → Model BEATS all naive baselines. Problem is miscalibration, not signal.")
    else:
        gap = (best_baseline - acc) * 100
        print(f"  → Model is {gap:.1f} pp WORSE than the best baseline.")
        print(f"  → The best baseline is: " +
              ("flip model" if flip_acc == best_baseline else
               "always-home" if always_home_acc == best_baseline else
               "always-market-favorite"))

    print()
    print("[done] Diagnostic complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())