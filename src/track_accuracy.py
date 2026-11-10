"""
Track prediction accuracy against actual NBA results.

Reads:  predictions/preds_*.csv        (all archived predictions)
        predictions/bet_slips_*.csv    (bet slip CSVs, new)
Fetches: nba_api LeagueGameFinder for recent dates only
Writes: predictions/tracked.csv        (one row per resolved game)
        logs/bet_slip_accuracy.csv     (one row per fully-resolved slip, new)

Run:
  python src\\track_accuracy.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from nba_api.stats.endpoints import leaguegamefinder
except ImportError:
    print("ERROR: nba_api not installed. Run: pip install nba_api")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRED_DIR = PROJECT_ROOT / "predictions"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRACKED_PATH = PRED_DIR / "tracked.csv"
BET_SLIP_ACC_PATH = LOGS_DIR / "bet_slip_accuracy.csv"


# ---------------------------------------------------------------------------
# TEAM NAME CANONICALIZATION
# ---------------------------------------------------------------------------
NAME_MAP = {
    "LA Clippers": "Los Angeles Clippers",
    "Los Angeles Clippers": "Los Angeles Clippers",
    "Portland Trail Blazers": "Portland Trail Blazers",
}


def canon(name: str) -> str:
    return NAME_MAP.get(str(name).strip(), str(name).strip())


# ---------------------------------------------------------------------------
# LOAD PREDICTIONS
# ---------------------------------------------------------------------------
def load_all_predictions() -> pd.DataFrame:
    files = sorted(glob.glob(str(PRED_DIR / "preds_*.csv")))
    if not files:
        print(f"No prediction files found in {PRED_DIR}")
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
        df["_source"] = Path(f).name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("_source")
    combined = combined.drop_duplicates(
        subset=["commence_time", "home_team", "away_team"], keep="last"
    ).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# FETCH ACTUAL RESULTS
# ---------------------------------------------------------------------------
def fetch_recent_results(dates: list) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()

    min_d = min(dates).date()
    max_d = max(dates).date()
    date_from = min_d.strftime("%m/%d/%Y")
    date_to = max_d.strftime("%m/%d/%Y")

    print(f"  querying nba_api for {date_from} -> {date_to} ...")
    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=date_from,
        date_to_nullable=date_to,
        league_id_nullable="00",
    )
    raw = finder.get_data_frames()[0]
    if raw.empty:
        return pd.DataFrame()

    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw["is_home"] = raw["MATCHUP"].str.contains("vs.", regex=False)
    raw["_key_team"] = raw["TEAM_NAME"].map(canon)

    home = raw[raw["is_home"]].copy()
    away = raw[~raw["is_home"]].copy()

    merged = home.merge(away, on="GAME_ID", suffixes=("_h", "_a"))
    out = pd.DataFrame({
        "actual_date":     merged["GAME_DATE_h"].dt.date,
        "actual_home":     merged["_key_team_h"],
        "actual_away":     merged["_key_team_a"],
        "actual_home_pts": merged["PTS_h"],
        "actual_away_pts": merged["PTS_a"],
        "actual_home_win": (merged["WL_h"] == "W").astype(int),
        "actual_margin":   merged["PTS_h"] - merged["PTS_a"],
        "actual_total":    merged["PTS_h"] + merged["PTS_a"],
    })
    return out


# ---------------------------------------------------------------------------
# MATCH + SCORE
# ---------------------------------------------------------------------------
def score_predictions(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    preds["home_canon"] = preds["home_team"].map(canon)
    preds["away_canon"] = preds["away_team"].map(canon)

    preds["pred_date_utc"] = preds["commence_time"].dt.date
    preds["pred_date_us"] = preds.apply(
        lambda r: (r["commence_time"] - pd.Timedelta(hours=10)).date()
        if r["commence_time"].hour < 10 else r["commence_time"].date(),
        axis=1,
    )

    merged = preds.merge(
        results,
        left_on=["pred_date_us", "home_canon", "away_canon"],
        right_on=["actual_date", "actual_home", "actual_away"],
        how="left",
    )

    still_missing = merged["actual_home_win"].isna()
    if still_missing.any():
        fallback = preds[still_missing].merge(
            results,
            left_on=["pred_date_utc", "home_canon", "away_canon"],
            right_on=["actual_date", "actual_home", "actual_away"],
            how="left",
            suffixes=("", "_fb"),
        )
        for col in ["actual_date", "actual_home", "actual_away",
                    "actual_home_pts", "actual_away_pts",
                    "actual_home_win", "actual_margin", "actual_total"]:
            merged.loc[still_missing, col] = fallback[col].values

    return merged


def add_hit_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    resolved = df["actual_home_win"].notna()
    df["resolved"] = resolved

    # ML
    df["ml_correct"] = np.nan
    mask = resolved & df["ml_pick"].isin(["HOME", "AWAY"])
    home_won = df["actual_home_win"] == 1
    df.loc[mask & (df["ml_pick"] == "HOME") & home_won, "ml_correct"] = 1
    df.loc[mask & (df["ml_pick"] == "HOME") & ~home_won, "ml_correct"] = 0
    df.loc[mask & (df["ml_pick"] == "AWAY") & ~home_won, "ml_correct"] = 1
    df.loc[mask & (df["ml_pick"] == "AWAY") & home_won, "ml_correct"] = 0

    # Spread (existing formula preserved — NOT modified per spec)
    df["spread_correct"] = np.nan
    mask = resolved & df["spread_pick"].isin(["HOME", "AWAY"]) & df["market_spread"].notna()
    df.loc[mask & (df["spread_pick"] == "HOME") &
           (df["actual_margin"] > df["market_spread"]), "spread_correct"] = 1
    df.loc[mask & (df["spread_pick"] == "HOME") &
           (df["actual_margin"] < df["market_spread"]), "spread_correct"] = 0
    df.loc[mask & (df["spread_pick"] == "AWAY") &
           (df["actual_margin"] < df["market_spread"]), "spread_correct"] = 1
    df.loc[mask & (df["spread_pick"] == "AWAY") &
           (df["actual_margin"] > df["market_spread"]), "spread_correct"] = 0

    # Total
    df["total_correct"] = np.nan
    mask = resolved & df["total_pick"].isin(["OVER", "UNDER"]) & df["market_total"].notna()
    df.loc[mask & (df["total_pick"] == "OVER") &
           (df["actual_total"] > df["market_total"]), "total_correct"] = 1
    df.loc[mask & (df["total_pick"] == "OVER") &
           (df["actual_total"] < df["market_total"]), "total_correct"] = 0
    df.loc[mask & (df["total_pick"] == "UNDER") &
           (df["actual_total"] < df["market_total"]), "total_correct"] = 1
    df.loc[mask & (df["total_pick"] == "UNDER") &
           (df["actual_total"] > df["market_total"]), "total_correct"] = 0

    return df


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> None:
    resolved = df[df["resolved"]].copy()
    print(f"\n{'='*70}")
    print(f"TRACKING SUMMARY — {len(resolved)} resolved games "
          f"of {len(df)} total predictions")
    print(f"{'='*70}")

    if resolved.empty:
        print("No resolved games yet. Run again after some games finish.")
        return

    def block(label: str, pick_col: str, hit_col: str):
        r = resolved[resolved[pick_col].isin(
            ["HOME", "AWAY", "OVER", "UNDER"]
        )]
        if r.empty:
            print(f"\n  {label}: no actionable picks yet")
            return
        n = len(r)
        correct = int(r[hit_col].sum())
        pct = correct / n
        print(f"\n  {label}:")
        print(f"    picks:     {n}")
        print(f"    correct:   {correct}")
        print(f"    accuracy:  {pct:.3f}")

        for side in ["HOME", "AWAY", "OVER", "UNDER"]:
            side_df = r[r[pick_col] == side]
            if len(side_df) == 0:
                continue
            acc = side_df[hit_col].mean()
            print(f"      {side:6s}: {int(side_df[hit_col].sum()):3d}/{len(side_df):3d}"
                  f"  ({acc:.3f})")

    block("MONEYLINE", "ml_pick", "ml_correct")
    block("SPREAD", "spread_pick", "spread_correct")
    block("TOTAL", "total_pick", "total_correct")

    if "season" in resolved.columns:
        print("\n  By season (moneyline only):")
        for season, grp in resolved.groupby("season"):
            r = grp[grp["ml_pick"].isin(["HOME", "AWAY"])]
            if len(r) == 0:
                continue
            print(f"    {season}: "
                  f"{int(r['ml_correct'].sum())}/{len(r)} "
                  f"({r['ml_correct'].mean():.3f})")


# ===========================================================================
# BET SLIP ACCURACY (NEW SECTION)
# ===========================================================================
def _slip_pick_correct(pick_row, tracked_df: pd.DataFrame):
    """Return True/False if pick is correct, or None if not resolvable.

    Uses CORRECT spread/total formulas (unlike the existing track flow above).
    """
    pick_market = pick_row["Market_Type"]
    team = pick_row["Team"]
    home = pick_row["HomeTeam"]
    away = pick_row["AwayTeam"]
    game_date = pick_row["Date"]

    match = tracked_df[
        (tracked_df["home_team"] == home) &
        (tracked_df["away_team"] == away)
    ].copy()
    if match.empty:
        return None

    # Filter to matching UTC date
    match["_utc_date"] = match["commence_time"].dt.strftime("%Y-%m-%d")
    date_match = match[match["_utc_date"] == game_date]
    if date_match.empty:
        date_match = match

    row = date_match.iloc[0]
    if not bool(row.get("resolved", False)):
        return None

    if pick_market == "ML":
        if pd.isna(row.get("actual_home_win")):
            return None
        actual = int(row["actual_home_win"])
        return bool(actual == 1) if team == home else bool(actual == 0)

    if pick_market == "Spread":
        if pd.isna(row.get("actual_margin")) or pd.isna(row.get("market_spread")):
            return None
        margin = float(row["actual_margin"])
        line = float(row["market_spread"])  # home-side line
        # home covers iff margin + line > 0  (line = -4.5 -> covers if margin > 4.5)
        home_covers = (margin + line) > 0
        return home_covers if team == home else (not home_covers)

    if pick_market == "Total":
        if pd.isna(row.get("actual_total")) or pd.isna(row.get("market_total")):
            return None
        total = float(row["actual_total"])
        line = float(row["market_total"])
        return bool(total > line) if team == "Over" else bool(total < line)

    return None


def track_bet_slips(tracked_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every bet_slips_*.csv. A slip is scored only if ALL picks
    have resolved. Returns DataFrame of resolved slips."""
    bet_files = sorted(glob.glob(str(PRED_DIR / "bet_slips_*.csv")))
    if not bet_files:
        return pd.DataFrame()

    results = []
    for f in bet_files:
        fname = Path(f).name  # bet_slips_YYYYMMDD.csv
        try:
            stamp = fname.replace("bet_slips_", "").replace(".csv", "")
            slip_date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        except Exception:
            continue

        df = pd.read_csv(f)
        if df.empty:
            continue

        for slip_id, slip_df in df.groupby("slip_id"):
            strategy = slip_df["strategy"].iloc[0]
            market = slip_df["market"].iloc[0]
            size = int(slip_df["size"].iloc[0])
            combined_prob = float(slip_df["combined_prob"].iloc[0])

            all_correct = []
            missing = False
            for _, pick in slip_df.iterrows():
                c = _slip_pick_correct(pick, tracked_df)
                if c is None:
                    missing = True
                    break
                all_correct.append(c)

            if missing:
                continue

            outcome = "WIN" if all(all_correct) else "LOSS"
            results.append({
                "slip_date": slip_date,
                "strategy": strategy,
                "market": market,
                "size": size,
                "slip_id": int(slip_id),
                "combined_prob": round(combined_prob, 4),
                "outcome": outcome,
            })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values(
        ["slip_date", "slip_id"]
    ).reset_index(drop=True)
    out.to_csv(BET_SLIP_ACC_PATH, index=False)
    return out


def print_bet_slip_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("BET SLIP ACCURACY")
    print("=" * 70)

    if df.empty:
        print("No fully-played slips yet. Results appear once all picks in "
              "a slip resolve.")
        return

    n = len(df)
    wins = int((df["outcome"] == "WIN").sum())
    win_rate = wins / n if n else 0
    avg_pred = float(df["combined_prob"].mean())

    print(f"  Slips evaluated: {n}")
    print(f"  Wins:            {wins}")
    print(f"  Win rate:        {win_rate:.3f}")
    print(f"  Avg predicted:   {avg_pred:.3f}")

    gap = avg_pred - win_rate
    if gap > 0.05:
        print(f"  ⚠  Overconfident by {gap*100:.1f} pp")
    elif gap < -0.05:
        print(f"  ✓  Under-confident by {-gap*100:.1f} pp (good)")
    else:
        print(f"  ✓  Well-calibrated (gap {gap*100:+.1f} pp)")

    print("\n  By strategy:")
    for strat, grp in df.groupby("strategy"):
        w = int((grp["outcome"] == "WIN").sum())
        print(f"    {strat:12s} {w}/{len(grp)} ({w/len(grp):.3f})")

    print("\n  By size:")
    for size_val, grp in df.groupby("size"):
        w = int((grp["outcome"] == "WIN").sum())
        print(f"    {int(size_val)}-fold       {w}/{len(grp)} ({w/len(grp):.3f})")

    print("\n  By market:")
    for mkt, grp in df.groupby("market"):
        w = int((grp["outcome"] == "WIN").sum())
        print(f"    {mkt:12s} {w}/{len(grp)} ({w/len(grp):.3f})")


def append_bet_slip_accuracy() -> None:
    """Called from main() AFTER the existing accuracy flow."""
    try:
        if not TRACKED_PATH.exists():
            print("\n(skipping bet slip accuracy — no tracked.csv yet)")
            return
        tracked = pd.read_csv(TRACKED_PATH)
        tracked["commence_time"] = pd.to_datetime(
            tracked["commence_time"], utc=True, errors="coerce"
        )
        slip_results = track_bet_slips(tracked)
        print_bet_slip_summary(slip_results)
    except Exception as e:
        print(f"\n(bet slip accuracy failed: {e!r})")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print("Loading archived predictions ...")
    preds = load_all_predictions()
    if preds.empty:
        print("Nothing to track yet.")
        return 1
    print(f"  loaded {len(preds)} unique predictions "
          f"across {preds['_source'].nunique()} files")

    now_utc = pd.Timestamp.now(tz="UTC")
    eligible = preds[preds["commence_time"] < now_utc - pd.Timedelta(hours=4)]
    print(f"  games ready to check: {len(eligible)}")

    if eligible.empty:
        print("\nNo games have finished yet. Run again later.")
        # Even with no new results, we can still re-evaluate existing bet slips
        # against tracked.csv, so don't return early.
        append_bet_slip_accuracy()
        return 0

    dates = eligible["commence_time"].dt.tz_convert("UTC").dt.tz_localize(None).tolist()
    results = fetch_recent_results(dates)
    if results.empty:
        print("  no results returned from nba_api (games may not be in the DB yet)")
        append_bet_slip_accuracy()
        return 0
    print(f"  fetched {len(results)} actual results")

    merged = score_predictions(eligible, results)
    merged = add_hit_flags(merged)

    keep = [
        "commence_time", "season", "home_team", "away_team", "bookmaker",
        "model_home_win_prob", "market_home_prob", "ml_edge", "ml_pick", "ml_correct",
        "model_margin", "market_spread", "spread_edge", "spread_pick", "spread_correct",
        "model_total", "market_total", "total_edge", "total_pick", "total_correct",
        "actual_home_pts", "actual_away_pts", "actual_home_win",
        "actual_margin", "actual_total", "resolved",
    ]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].sort_values("commence_time")

    TRACKED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TRACKED_PATH, index=False)
    print(f"\nWrote {TRACKED_PATH.relative_to(PROJECT_ROOT)}")

    summarize(out)

    # ------- NEW: bet slip accuracy section -------
    append_bet_slip_accuracy()

    print(f"\n[done] track_accuracy complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())