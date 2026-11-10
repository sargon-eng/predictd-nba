"""
Bet slip builder for Predictd NBA.

Generates up to 20 combinatorial bet slips per run (4 strategies × 5 sizes)
with diversity rules, jittered scoring, and learning weights derived from
historical accuracy in predictions/tracked.csv.

Reads:  predictions/latest.csv         (upcoming games)
        predictions/tracked.csv        (resolved history for learning)
Writes: predictions/bet_slips_YYYYMMDD.csv

Run:
  python src\\bet_slip_builder.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "predictions"
LATEST_CSV = PRED_DIR / "latest.csv"
TRACKED_CSV = PRED_DIR / "tracked.csv"


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
WEEK_WINDOW_DAYS = 5
MIN_MATCHES_FOR_LEARNING = 10

# Spread/total probability models (std devs from Step 5 training MAE)
SIGMA_MARGIN = 15.0
SIGMA_TOTAL = 19.7

STRATEGIES = {
    "top_mixed": {
        "seed": 100,
        "markets": ["ml", "spread", "total"],
        "desc": "🏆 Best overall picks across ML/Spread/Total",
    },
    "ml_best": {
        "seed": 200,
        "markets": ["ml"],
        "desc": "💰 Best Moneyline picks only",
    },
    "spread_best": {
        "seed": 300,
        "markets": ["spread"],
        "desc": "📊 Best Spread picks only",
    },
    "total_best": {
        "seed": 400,
        "markets": ["total"],
        "desc": "📈 Best Over/Under picks only",
    },
}

SIZES = [3, 4, 5, 6, 7]

# Spec's INTENDED max-per-day, used when enough days are available.
# Adjusted adaptively in build_slip() when games cluster on fewer days.
MAX_PER_DAY_INTENDED = {3: 1, 4: 2, 5: 2, 6: 3, 7: 3}

CONF_BUCKETS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def acc_weight(acc: float) -> float:
    return clamp(0.7 + (acc - 0.5) * 3.0, 0.7, 1.3)


def bucket_for(prob: float) -> tuple:
    for lo, hi in CONF_BUCKETS:
        if lo <= prob < hi:
            return (lo, hi)
    return CONF_BUCKETS[-1]


def prob_spread(model_margin: float, market_spread: float, side: str) -> float:
    """P(home covers) under Normal(model_margin, SIGMA_MARGIN).

    Convention: home covers iff actual_margin > -market_spread.
    """
    threshold = -float(market_spread)
    z = (threshold - float(model_margin)) / SIGMA_MARGIN
    return float(norm.sf(z)) if side == "HOME" else float(norm.cdf(z))


def prob_total(model_total: float, market_total: float, side: str) -> float:
    z = (float(market_total) - float(model_total)) / SIGMA_TOTAL
    return float(norm.sf(z)) if side == "OVER" else float(norm.cdf(z))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_latest() -> pd.DataFrame:
    if not LATEST_CSV.exists():
        print(f"ERROR: {LATEST_CSV} not found")
        return pd.DataFrame()
    df = pd.read_csv(LATEST_CSV)
    df["commence_time"] = pd.to_datetime(
        df["commence_time"], utc=True, errors="coerce"
    )
    return df


def load_tracked_resolved() -> pd.DataFrame:
    if not TRACKED_CSV.exists():
        print(f"WARN: {TRACKED_CSV} not found — learning weights all 1.0")
        return pd.DataFrame()
    df = pd.read_csv(TRACKED_CSV)
    df["commence_time"] = pd.to_datetime(
        df["commence_time"], utc=True, errors="coerce"
    )
    if "resolved" in df.columns:
        df = df[df["resolved"] == True].copy()
    return df


# ---------------------------------------------------------------------------
# LEARNING WEIGHTS
# ---------------------------------------------------------------------------
def correct_for_pick(row, market: str, side: str):
    if pd.isna(row.get("actual_margin")):
        return None
    if market == "ML":
        if pd.isna(row.get("actual_home_win")):
            return None
        actual = int(row["actual_home_win"])
        return int(actual == 1) if side == "HOME" else int(actual == 0)
    if market == "Spread":
        if pd.isna(row.get("market_spread")):
            return None
        diff = float(row["actual_margin"]) + float(row["market_spread"])
        if abs(diff) < 1e-9:
            return None
        return int(diff > 0) if side == "HOME" else int(diff < 0)
    if market == "Total":
        if pd.isna(row.get("market_total")) or pd.isna(row.get("actual_total")):
            return None
        diff = float(row["actual_total"]) - float(row["market_total"])
        if abs(diff) < 1e-9:
            return None
        return int(diff > 0) if side == "OVER" else int(diff < 0)
    return None


def compute_learning_weights(tracked: pd.DataFrame) -> dict:
    team_w, team_n = {}, {}
    market_w, market_n = {}, {}
    bucket_w, bucket_n = {}, {}

    for _, row in tracked.iterrows():
        for market, pick_col in [
            ("ML", "ml_pick"),
            ("Spread", "spread_pick"),
            ("Total", "total_pick"),
        ]:
            pick = row.get(pick_col)
            if pick not in ("HOME", "AWAY", "OVER", "UNDER"):
                continue
            correct = correct_for_pick(row, market, pick)
            if correct is None:
                continue

            if market == "ML":
                mhw = row.get("model_home_win_prob")
                if pd.isna(mhw):
                    continue
                conf = float(mhw) if pick == "HOME" else 1.0 - float(mhw)
            elif market == "Spread":
                mm, ms = row.get("model_margin"), row.get("market_spread")
                if pd.isna(mm) or pd.isna(ms):
                    continue
                conf = prob_spread(float(mm), float(ms), pick)
            else:
                mt, mkt = row.get("model_total"), row.get("market_total")
                if pd.isna(mt) or pd.isna(mkt):
                    continue
                conf = prob_total(float(mt), float(mkt), pick)

            market_n[market] = market_n.get(market, 0) + 1
            market_w[market] = market_w.get(market, 0) + correct

            if market != "Total":
                team = row["home_team"] if pick == "HOME" else row["away_team"]
                team_n[team] = team_n.get(team, 0) + 1
                team_w[team] = team_w.get(team, 0) + correct

            b = bucket_for(conf)
            bucket_n[b] = bucket_n.get(b, 0) + 1
            bucket_w[b] = bucket_w.get(b, 0) + correct

    def finalize(wins, totals):
        return {
            k: acc_weight(wins.get(k, 0) / n) if n >= MIN_MATCHES_FOR_LEARNING else 1.0
            for k, n in totals.items()
        }

    return {
        "team": finalize(team_w, team_n),
        "market": finalize(market_w, market_n),
        "bucket": finalize(bucket_w, bucket_n),
        "raw": {
            "team": {k: (team_w.get(k, 0), team_n[k]) for k in team_n},
            "market": {k: (market_w.get(k, 0), market_n[k]) for k in market_n},
            "bucket": {k: (bucket_w.get(k, 0), bucket_n[k]) for k in bucket_n},
        },
    }


# ---------------------------------------------------------------------------
# CANDIDATE PICKS
# ---------------------------------------------------------------------------
def build_candidates(latest: pd.DataFrame, w_start, w_end) -> list:
    candidates = []
    for _, row in latest.iterrows():
        ct = row.get("commence_time")
        if pd.isna(ct) or not (w_start <= ct <= w_end):
            continue
        game_key = f"{ct.isoformat()}|{row['home_team']}|{row['away_team']}"
        game_date = ct.date()

        if row.get("ml_pick") in ("HOME", "AWAY"):
            side = row["ml_pick"]
            team = row["home_team"] if side == "HOME" else row["away_team"]
            prob = float(row["model_home_win_prob"]) if side == "HOME" \
                else float(row["model_away_win_prob"])
            candidates.append({
                "game_key": game_key, "commence_time": ct, "game_date": game_date,
                "home_team": row["home_team"], "away_team": row["away_team"],
                "market": "ML", "market_lower": "ml", "side": side, "team": team,
                "diversity_team": team, "probability": prob,
                "pick_value": f"{team} ML",
            })

        if row.get("spread_pick") in ("HOME", "AWAY") and pd.notna(row.get("market_spread")):
            side = row["spread_pick"]
            team = row["home_team"] if side == "HOME" else row["away_team"]
            line = float(row["market_spread"])
            prob = prob_spread(float(row["model_margin"]), line, side)
            candidates.append({
                "game_key": game_key, "commence_time": ct, "game_date": game_date,
                "home_team": row["home_team"], "away_team": row["away_team"],
                "market": "Spread", "market_lower": "spread", "side": side, "team": team,
                "diversity_team": team, "probability": prob,
                "pick_value": f"{team} {line:+.1f}",
            })

        if row.get("total_pick") in ("OVER", "UNDER") and pd.notna(row.get("market_total")):
            side = row["total_pick"]
            line = float(row["market_total"])
            prob = prob_total(float(row["model_total"]), line, side)
            candidates.append({
                "game_key": game_key, "commence_time": ct, "game_date": game_date,
                "home_team": row["home_team"], "away_team": row["away_team"],
                "market": "Total", "market_lower": "total", "side": side,
                "team": side.capitalize(), "diversity_team": f"TOTAL_{game_key}_{side}",
                "probability": prob,
                "pick_value": f"{side.capitalize()} {line:.1f}",
            })

    return candidates


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------
def score_pick(pick: dict, weights: dict) -> float:
    team_w = 1.0 if pick["market"] == "Total" else weights["team"].get(pick["team"], 1.0)
    market_w = weights["market"].get(pick["market"], 1.0)
    b = bucket_for(pick["probability"])
    bucket_w = weights["bucket"].get(b, 1.0)
    return pick["probability"] * team_w * market_w * bucket_w


# ---------------------------------------------------------------------------
# SLIP BUILDER
# ---------------------------------------------------------------------------
def build_slip(candidates: list, strategy: str, size: int) -> tuple:
    """Returns (slip_dict or None, failure_reason or None)."""
    cfg = STRATEGIES[strategy]
    pool = [p for p in candidates if p["market_lower"] in cfg["markets"]]

    if len(pool) < size:
        return None, f"only {len(pool)} candidate picks (need {size})"

    # Hard diversity constraint: one pick per game
    unique_games = len(set(p["game_key"] for p in pool))
    if unique_games < size:
        return None, f"only {unique_games} unique games (need {size})"

    # Adaptive max-per-day: spec's limit OR enough to reach `size` given available days
    n_days = len(set(p["game_date"] for p in pool))
    intended = MAX_PER_DAY_INTENDED[size]
    min_needed = ceil_div(size, n_days)
    effective_max_per_day = max(intended, min_needed)

    rng = np.random.default_rng(cfg["seed"] * 10 + size)
    jittered = [(p, p["score"] + rng.normal(0, 0.01)) for p in pool]
    jittered.sort(key=lambda x: -x[1])

    selected = []
    used_games, used_teams, used_days = set(), set(), {}

    for pick, _ in jittered:
        if len(selected) >= size:
            break
        if pick["game_key"] in used_games:
            continue
        if pick["diversity_team"] in used_teams:
            continue
        if used_days.get(pick["game_date"], 0) >= effective_max_per_day:
            continue
        selected.append(pick)
        used_games.add(pick["game_key"])
        used_teams.add(pick["diversity_team"])
        used_days[pick["game_date"]] = used_days.get(pick["game_date"], 0) + 1

    if len(selected) < size:
        return None, f"selected only {len(selected)}/{size} (diversity limits)"

    combined = 1.0
    for p in selected:
        combined *= p["probability"]

    return {
        "strategy": strategy,
        "market": "mixed" if len(cfg["markets"]) > 1 else cfg["markets"][0],
        "size": size,
        "picks": selected,
        "combined_prob": combined,
    }, None


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def write_output(slips: list, now) -> Path | None:
    if not slips:
        return None
    out_path = PRED_DIR / f"bet_slips_{now.strftime('%Y%m%d')}.csv"
    rows = []
    for slip in slips:
        for i, p in enumerate(slip["picks"], 1):
            rows.append({
                "slip_id": slip["slip_id"],
                "strategy": slip["strategy"],
                "market": slip["market"],
                "size": slip["size"],
                "combined_prob": round(slip["combined_prob"], 4),
                "pick_num": i,
                "Date": p["game_date"].isoformat(),
                "HomeTeam": p["home_team"],
                "AwayTeam": p["away_team"],
                "Team": p["team"],
                "Market_Type": p["market"],
                "pick_value": p["pick_value"],
                "probability": round(p["probability"], 4),
                "score": round(p["score"], 4),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def print_weights(weights: dict) -> None:
    print()
    print("=" * 70)
    print("LEARNING WEIGHTS (from historical accuracy)")
    print("=" * 70)

    print("\nBy market:")
    for m in ["ML", "Spread", "Total"]:
        w, n = weights["raw"]["market"].get(m, (0, 0))
        wt = weights["market"].get(m, 1.0)
        if n > 0:
            print(f"  {m:8s} {w}/{n} ({w/n:.3f})  -> weight {wt:.3f}")
        else:
            print(f"  {m:8s} no data  -> weight {wt:.3f}")

    print("\nBy confidence bucket:")
    for b in CONF_BUCKETS:
        w, n = weights["raw"]["bucket"].get(b, (0, 0))
        wt = weights["bucket"].get(b, 1.0)
        label = f"{b[0]:.1f}-{b[1]:.1f}"
        if n > 0:
            print(f"  {label:8s} {w}/{n} ({w/n:.3f})  -> weight {wt:.3f}")
        else:
            print(f"  {label:8s} no data  -> weight {wt:.3f}")

    eligible = [(k, v) for k, v in weights["raw"]["team"].items()
                if v[1] >= MIN_MATCHES_FOR_LEARNING]
    if eligible:
        eligible.sort(key=lambda kv: -(kv[1][0] / kv[1][1]))
        print(f"\nBy team ({len(eligible)} eligible):")
        for team, (w, n) in eligible:
            print(f"  {team:25s} {w}/{n} ({w/n:.3f})  -> weight {weights['team'][team]:.3f}")
    else:
        print(f"\nNo teams have >= {MIN_MATCHES_FOR_LEARNING} picks yet")


def print_report(slips: list, latest: pd.DataFrame, w_start, w_end,
                 candidates: list) -> None:
    print()
    print("=" * 70)
    print(f"BET SLIP REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print(f"Window: {w_start.date()} -> {w_end.date()} ({WEEK_WINDOW_DAYS} days)")

    day_counts = {}
    for _, row in latest.iterrows():
        ct = row.get("commence_time")
        if pd.isna(ct) or not (w_start <= ct <= w_end):
            continue
        d = ct.date()
        day_counts[d] = day_counts.get(d, 0) + 1

    print("\nGames in window by day:")
    if not day_counts:
        print("  (none)")
    for d in sorted(day_counts):
        print(f"  {d}  {day_counts[d]} games")

    if candidates:
        unique_games = len(set(c["game_key"] for c in candidates))
        by_market = {}
        for c in candidates:
            by_market[c["market_lower"]] = by_market.get(c["market_lower"], 0) + 1
        print(f"\nCandidates: {len(candidates)} picks from {unique_games} unique games")
        print(f"  by market: {by_market}")

    print(f"\n{'─'*70}")
    print(f"Generated {len(slips)} of 20 slips")
    print(f"{'─'*70}")

    current_strat = None
    for slip in slips:
        if slip["strategy"] != current_strat:
            current_strat = slip["strategy"]
            print(f"\n{STRATEGIES[current_strat]['desc']}")
        dot = "🟢" if slip["combined_prob"] >= 0.5 else (
            "🟡" if slip["combined_prob"] >= 0.25 else "🔴")
        print(f"  {dot} #{slip['slip_id']:2d}  {slip['size']}-fold  "
              f"combined={slip['combined_prob']*100:5.1f}%")
        for i, p in enumerate(slip["picks"], 1):
            print(f"          {i}. {p['pick_value']:28s} "
                  f"({p['home_team']} vs {p['away_team']}, {p['game_date']}) "
                  f"p={p['probability']:.3f}")

    print(f"\n{'─'*70}")
    print("Coverage:")
    for strat in STRATEGIES:
        strat_slips = [s for s in slips if s["strategy"] == strat]
        if not strat_slips:
            print(f"  {strat:12s} (none)")
            continue
        teams, markets = set(), set()
        for s in strat_slips:
            for p in s["picks"]:
                markets.add(p["market"])
                if p["market"] != "Total":
                    teams.add(p["team"])
        print(f"  {strat:12s} {len(strat_slips)} slips | "
              f"{len(teams)} teams | markets: {sorted(markets)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print("Predictd NBA — Bet Slip Builder")
    print()
    latest = load_latest()
    print(f"Loaded {len(latest)} games from latest.csv")
    tracked = load_tracked_resolved()
    print(f"Loaded {len(tracked)} resolved games from tracked.csv")

    weights = compute_learning_weights(tracked)
    print_weights(weights)

    now = pd.Timestamp.now(tz="UTC")
    w_start = now
    w_end = now + pd.Timedelta(days=WEEK_WINDOW_DAYS)

    candidates = build_candidates(latest, w_start, w_end)

    if len(candidates) == 0:
        print()
        print(f"ERROR: No games found in the {WEEK_WINDOW_DAYS}-day window.")
        print(f"  Window: {w_start} -> {w_end}")
        print(f"  latest.csv has {len(latest)} games, none in window.")
        print()
        print("To fix:")
        print("  1. python src\\fetch_odds.py    (fetch fresh odds)")
        print("  2. python src\\predict_live.py   (regenerate predictions)")
        print("  3. python src\\bet_slip_builder.py")
        return 1

    for c in candidates:
        c["score"] = score_pick(c, weights)

    slips = []
    failures = []
    for strat in STRATEGIES:
        for size in SIZES:
            slip, reason = build_slip(candidates, strat, size)
            if slip is None:
                failures.append((strat, size, reason))
                continue
            slips.append(slip)

    for i, s in enumerate(slips, 1):
        s["slip_id"] = i

    if not slips:
        print()
        print("ERROR: 0 slips built. Reasons:")
        for strat, size, reason in failures:
            print(f"  {strat} {size}-fold: {reason}")
        return 1

    out = write_output(slips, now)
    if out:
        print(f"\nWrote {out.relative_to(PROJECT_ROOT)}")

    print_report(slips, latest, w_start, w_end, candidates)

    if failures:
        print(f"\n{'─'*70}")
        print(f"Skipped {len(failures)} slip slots (low game count in window):")
        for strat, size, reason in failures:
            print(f"  {strat:12s} {size}-fold: {reason}")
        print(f"\n  This is expected on days with few games. The system will")
        print(f"  generate closer to 20 slips once more games are on the schedule.")

    print(f"\n[done] {len(slips)} slips written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())