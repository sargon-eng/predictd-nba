"""
Step 7 — Generate live predictions for upcoming NBA games.

Reads:  models/moneyline.pkl, margin.pkl, total.pkl, feature_columns.json
        data/processed/features.csv   (to replay team state up to today)
        data/processed/odds_latest.csv

Writes: predictions/latest.csv                 (overwrites)
        predictions/preds_YYYY-MM-DD.csv       (archive for accuracy tracker)

Run:
  python src\\predict_live.py
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRED_DIR = PROJECT_ROOT / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = PROCESSED_DIR / "features.csv"
ODDS_CSV = PROCESSED_DIR / "odds_latest.csv"
FEATURE_JSON = MODELS_DIR / "feature_columns.json"


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ELO_START = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE_ELO = 100.0
ROLL_WINDOW_LONG = 10
SEASON_OPENER_REST_DAYS = 7
HORIZON_DAYS = 14                # only predict games within next 14 days
EDGE_THRESHOLD = 0.03            # 3% edge required to flag a pick

# At the season boundary, pull each team's Elo 25% back toward the mean.
# Reflects offseason roster turnover that Elo otherwise misses.
SEASON_REGRESSION = 0.25

# Team-name aliases: keys come from The Odds API, values match nba_api names
TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "LA Clippers": "LA Clippers",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Golden State Warriors": "Golden State Warriors",
    "Brooklyn Nets": "Brooklyn Nets",
    "Washington Wizards": "Washington Wizards",
    "Charlotte Hornets": "Charlotte Hornets",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "Utah Jazz": "Utah Jazz",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def expected_home_win(home_elo: float, away_elo: float) -> float:
    diff = (home_elo + HOME_ADVANTAGE_ELO) - away_elo
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def win_pct(num: int, den: int) -> float:
    return (num / den) if den > 0 else np.nan


def mean_of_last(d, team_id, n):
    r = d.get(team_id)
    if not r:
        return np.nan
    last_n = list(r)[-n:]
    return float(np.mean(last_n)) if last_n else np.nan


def american_to_implied(ml: float) -> float:
    if ml is None or pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        dec = 1 + ml / 100.0
    else:
        dec = 1 + 100.0 / abs(ml)
    return 1.0 / dec


def devig_two_way(p_home_raw: float, p_away_raw: float):
    if pd.isna(p_home_raw) or pd.isna(p_away_raw):
        return np.nan, np.nan
    tot = p_home_raw + p_away_raw
    if tot <= 0:
        return np.nan, np.nan
    return p_home_raw / tot, p_away_raw / tot


def season_label(d: datetime) -> str:
    y, m = d.year, d.month
    if m >= 10:
        return f"{y}-{str(y + 1)[-2:]}"
    if m <= 6:
        return f"{y - 1}-{str(y)[-2:]}"
    return f"{y}-{str(y + 1)[-2:]}"


def to_naive_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


# ---------------------------------------------------------------------------
# STATE REPLAY
# ---------------------------------------------------------------------------
def replay_team_state(features: pd.DataFrame) -> dict:
    features = features.sort_values("game_date").reset_index(drop=True)

    elo: dict[int, float] = {}
    last_game_date: dict[int, pd.Timestamp] = {}
    last_season: dict[int, str] = {}       # most recent season a team played in

    season_wins: dict[tuple, int] = {}
    season_games: dict[tuple, int] = {}
    home_wins: dict[tuple, int] = {}
    home_games: dict[tuple, int] = {}
    away_wins: dict[tuple, int] = {}
    away_games: dict[tuple, int] = {}

    recent_results: dict[int, deque] = {}
    recent_margins: dict[int, deque] = {}
    recent_pts_for: dict[int, deque] = {}
    recent_pts_against: dict[int, deque] = {}

    for g in features.itertuples(index=False):
        hid = int(g.home_team_id)
        aid = int(g.away_team_id)
        season = g.season
        gdate = g.game_date
        actual_home = int(g.home_win)
        margin = int(g.margin)
        total_pts = int(g.total_pts)
        home_pts = (total_pts + margin) // 2
        away_pts = total_pts - home_pts

        home_elo = elo.get(hid, ELO_START)
        away_elo = elo.get(aid, ELO_START)

        # Season-boundary Elo regression to mean
        if hid in last_season and last_season[hid] != season:
            home_elo = home_elo * (1 - SEASON_REGRESSION) + ELO_START * SEASON_REGRESSION
        if aid in last_season and last_season[aid] != season:
            away_elo = away_elo * (1 - SEASON_REGRESSION) + ELO_START * SEASON_REGRESSION

        exp_home = expected_home_win(home_elo, away_elo)

        elo[hid] = home_elo + ELO_K * (actual_home - exp_home)
        elo[aid] = away_elo + ELO_K * ((1 - actual_home) - (1 - exp_home))
        last_game_date[hid] = gdate
        last_game_date[aid] = gdate
        last_season[hid] = season
        last_season[aid] = season

        season_games[(hid, season)] = season_games.get((hid, season), 0) + 1
        season_games[(aid, season)] = season_games.get((aid, season), 0) + 1
        season_wins[(hid, season)] = season_wins.get((hid, season), 0) + actual_home
        season_wins[(aid, season)] = season_wins.get((aid, season), 0) + (1 - actual_home)

        home_games[(hid, season)] = home_games.get((hid, season), 0) + 1
        home_wins[(hid, season)] = home_wins.get((hid, season), 0) + actual_home
        away_games[(aid, season)] = away_games.get((aid, season), 0) + 1
        away_wins[(aid, season)] = away_wins.get((aid, season), 0) + (1 - actual_home)

        recent_results.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(actual_home)
        recent_results.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(1 - actual_home)
        recent_margins.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(margin)
        recent_margins.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(-margin)
        recent_pts_for.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(home_pts)
        recent_pts_for.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(away_pts)
        recent_pts_against.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(away_pts)
        recent_pts_against.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(home_pts)

    # ---- Build "last completed season" stats per team ----
    # For each team, find their most recent season and record final stats.
    last_season_stats: dict[int, dict] = {}
    for (tid, season), games_played in season_games.items():
        if last_season.get(tid) != season:
            continue
        wins = season_wins.get((tid, season), 0)
        last_season_stats[tid] = {
            "win_pct": win_pct(wins, games_played),
            "home_win_pct": win_pct(home_wins.get((tid, season), 0),
                                    home_games.get((tid, season), 0)),
            "away_win_pct": win_pct(away_wins.get((tid, season), 0),
                                    away_games.get((tid, season), 0)),
        }

    return {
        "elo": elo,
        "last_game_date": last_game_date,
        "last_season": last_season,
        "last_season_stats": last_season_stats,
        "season_wins": season_wins,
        "season_games": season_games,
        "home_wins": home_wins,
        "home_games": home_games,
        "away_wins": away_wins,
        "away_games": away_games,
        "recent_results": recent_results,
        "recent_margins": recent_margins,
        "recent_pts_for": recent_pts_for,
        "recent_pts_against": recent_pts_against,
    }


def build_upcoming_features(state: dict, home_id: int, away_id: int,
                            commence_utc: pd.Timestamp,
                            current_season: str) -> dict:
    commence_utc = to_naive_utc(commence_utc)

    elo = state["elo"]
    lgd = state["last_game_date"]

    home_elo = elo.get(home_id, ELO_START)
    away_elo = elo.get(away_id, ELO_START)

    def rest(team_id):
        if team_id not in lgd:
            return SEASON_OPENER_REST_DAYS
        return (commence_utc - lgd[team_id]).days

    home_rest = rest(home_id)
    away_rest = rest(away_id)

    def season_stat(team_id, stat_key):
        """Return stat from current season if any games; else fall back to
        team's most recent completed season; else NaN."""
        games = state["season_games"].get((team_id, current_season), 0)
        if games > 0:
            if stat_key == "win_pct":
                return win_pct(state["season_wins"].get((team_id, current_season), 0),
                               games)
            if stat_key == "home_win_pct":
                return win_pct(state["home_wins"].get((team_id, current_season), 0),
                               state["home_games"].get((team_id, current_season), 0))
            if stat_key == "away_win_pct":
                return win_pct(state["away_wins"].get((team_id, current_season), 0),
                               state["away_games"].get((team_id, current_season), 0))
        fallback = state["last_season_stats"].get(team_id)
        if fallback:
            return fallback.get(stat_key, np.nan)
        return np.nan

    return {
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff_pre": home_elo - away_elo,
        "elo_home_win_prob_pre": expected_home_win(home_elo, away_elo),

        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_diff": home_rest - away_rest,
        "home_b2b": int(home_rest <= 1),
        "away_b2b": int(away_rest <= 1),

        "home_season_win_pct": season_stat(home_id, "win_pct"),
        "away_season_win_pct": season_stat(away_id, "win_pct"),
        "home_home_win_pct":   season_stat(home_id, "home_win_pct"),
        "away_away_win_pct":   season_stat(away_id, "away_win_pct"),

        "home_win_pct_last5":  mean_of_last(state["recent_results"], home_id, 5),
        "away_win_pct_last5":  mean_of_last(state["recent_results"], away_id, 5),
        "home_win_pct_last10": mean_of_last(state["recent_results"], home_id, 10),
        "away_win_pct_last10": mean_of_last(state["recent_results"], away_id, 10),
        "home_avg_margin_last5":      mean_of_last(state["recent_margins"], home_id, 5),
        "away_avg_margin_last5":      mean_of_last(state["recent_margins"], away_id, 5),
        "home_avg_pts_for_last5":     mean_of_last(state["recent_pts_for"], home_id, 5),
        "away_avg_pts_for_last5":     mean_of_last(state["recent_pts_for"], away_id, 5),
        "home_avg_pts_against_last5": mean_of_last(state["recent_pts_against"], home_id, 5),
        "away_avg_pts_against_last5": mean_of_last(state["recent_pts_against"], away_id, 5),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    for p in [MODELS_DIR / "moneyline.pkl",
              MODELS_DIR / "margin.pkl",
              MODELS_DIR / "total.pkl",
              FEATURE_JSON]:
        if not p.exists():
            print(f"ERROR: missing {p}")
            print("       Run: python src\\train_models.py")
            return 1

    ml_model = joblib.load(MODELS_DIR / "moneyline.pkl")
    margin_model = joblib.load(MODELS_DIR / "margin.pkl")
    total_model = joblib.load(MODELS_DIR / "total.pkl")
    with open(FEATURE_JSON) as f:
        feature_cols = json.load(f)
    print(f"Loaded models. {len(feature_cols)} feature columns.")

    if not FEATURES_CSV.exists():
        print(f"ERROR: missing {FEATURES_CSV}")
        return 1

    print(f"Loading {FEATURES_CSV.name} ...")
    features = pd.read_csv(FEATURES_CSV, parse_dates=["game_date"],
                           dtype={"game_id": str})
    state = replay_team_state(features)
    print(f"  replayed {len(features)} historical games")
    print(f"  last completed season stats known for "
          f"{len(state['last_season_stats'])} teams")

    name_to_id = {}
    for df, name_col, id_col in [(features, "home_team", "home_team_id"),
                                 (features, "away_team", "away_team_id")]:
        for name, tid in zip(df[name_col], df[id_col]):
            name_to_id[str(name)] = int(tid)
    print(f"  known teams: {len(name_to_id)}")

    if not ODDS_CSV.exists():
        print(f"ERROR: missing {ODDS_CSV}")
        return 1

    odds = pd.read_csv(ODDS_CSV)
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True)
    print(f"Loaded {len(odds)} upcoming games from odds feed")

    now_utc = pd.Timestamp.now(tz="UTC")
    horizon = now_utc + pd.Timedelta(days=HORIZON_DAYS)
    odds = odds[(odds["commence_time"] >= now_utc) &
                (odds["commence_time"] <= horizon)].copy()
    print(f"  games within next {HORIZON_DAYS} days: {len(odds)}")

    if odds.empty:
        print("\nNo games in the near-term horizon. Nothing to predict.")
        return 0

    rows = []
    unmatched = set()
    for g in odds.itertuples(index=False):
        home_name = TEAM_ALIASES.get(g.home_team, g.home_team)
        away_name = TEAM_ALIASES.get(g.away_team, g.away_team)
        home_id = name_to_id.get(home_name)
        away_id = name_to_id.get(away_name)
        if home_id is None:
            unmatched.add(g.home_team); continue
        if away_id is None:
            unmatched.add(g.away_team); continue

        season = season_label(g.commence_time.to_pydatetime())
        feats = build_upcoming_features(state, home_id, away_id,
                                        g.commence_time, season)

        X = pd.DataFrame([feats])[feature_cols]
        p_home = float(ml_model.predict_proba(X)[:, 1][0])
        pred_margin = float(margin_model.predict(X)[0])
        pred_total = float(total_model.predict(X)[0])

        imp_home_raw = american_to_implied(g.home_ml)
        imp_away_raw = american_to_implied(g.away_ml)
        mkt_home, mkt_away = devig_two_way(imp_home_raw, imp_away_raw)

        home_edge = p_home - mkt_home if not pd.isna(mkt_home) else np.nan
        away_edge = (1 - p_home) - mkt_away if not pd.isna(mkt_away) else np.nan

        ml_pick, ml_edge = "PASS", np.nan
        if not pd.isna(home_edge) and not pd.isna(away_edge):
            if home_edge >= EDGE_THRESHOLD and home_edge >= away_edge:
                ml_pick, ml_edge = "HOME", home_edge
            elif away_edge >= EDGE_THRESHOLD:
                ml_pick, ml_edge = "AWAY", away_edge

        spread_pick, spread_edge = "PASS", np.nan
        if not pd.isna(g.home_spread):
            diff = pred_margin - float(g.home_spread)
            if diff >= 2.0:
                spread_pick, spread_edge = "HOME", diff
            elif diff <= -2.0:
                spread_pick, spread_edge = "AWAY", diff

        total_pick, total_edge = "PASS", np.nan
        if not pd.isna(g.total):
            diff = pred_total - float(g.total)
            if diff >= 3.0:
                total_pick, total_edge = "OVER", diff
            elif diff <= -3.0:
                total_pick, total_edge = "UNDER", diff

        rows.append({
            "commence_time": g.commence_time,
            "season": season,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "bookmaker": g.bookmaker,

            "model_home_win_prob": round(p_home, 4),
            "model_away_win_prob": round(1 - p_home, 4),
            "market_home_ml": g.home_ml,
            "market_away_ml": g.away_ml,
            "market_home_prob": round(mkt_home, 4) if not pd.isna(mkt_home) else np.nan,
            "market_away_prob": round(mkt_away, 4) if not pd.isna(mkt_away) else np.nan,
            "ml_edge": round(float(ml_edge), 4) if not pd.isna(ml_edge) else np.nan,
            "ml_pick": ml_pick,

            "model_margin": round(pred_margin, 2),
            "market_spread": g.home_spread,
            "spread_edge": round(float(spread_edge), 2) if not pd.isna(spread_edge) else np.nan,
            "spread_pick": spread_pick,

            "model_total": round(pred_total, 2),
            "market_total": g.total,
            "total_edge": round(float(total_edge), 2) if not pd.isna(total_edge) else np.nan,
            "total_pick": total_pick,
        })

    if unmatched:
        print(f"\nWARNING: could not match {len(unmatched)} team names:")
        for t in sorted(unmatched):
            print(f"  - {t}")

    if not rows:
        print("\nNo predictions generated (all games unmatched).")
        return 1

    preds = pd.DataFrame(rows).sort_values("commence_time").reset_index(drop=True)

    latest_path = PRED_DIR / "latest.csv"
    dated_path = PRED_DIR / f"preds_{now_utc.strftime('%Y-%m-%d')}.csv"
    preds.to_csv(latest_path, index=False)
    preds.to_csv(dated_path, index=False)
    print(f"\nWrote {latest_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {dated_path.relative_to(PROJECT_ROOT)}")

    print(f"\n{'='*100}")
    print(f"PREDICTIONS — {len(preds)} games")
    print(f"{'='*100}")
    for row in preds.itertuples(index=False):
        ct = row.commence_time.strftime("%a %b %d %H:%M UTC")
        print(f"\n{ct}")
        print(f"  {row.away_team} @ {row.home_team}   [{row.bookmaker}]")

        ph = row.model_home_win_prob * 100
        print(f"  ML     model: home {ph:.1f}% / away {100 - ph:.1f}%"
              f"   market: home {row.market_home_prob*100:.1f}% / away {row.market_away_prob*100:.1f}%"
              f"   -> pick: {row.ml_pick}"
              + (f"  edge +{row.ml_edge*100:.1f}%" if row.ml_pick != "PASS" else ""))

        print(f"  SPREAD model: home {row.model_margin:+.1f}"
              f"   market: home {row.market_spread:+.1f}"
              f"   -> pick: {row.spread_pick}"
              + (f"  edge {row.spread_edge:+.1f}" if row.spread_pick != "PASS" else ""))

        print(f"  TOTAL  model: {row.model_total:.1f}"
              f"   market: {row.market_total:.1f}"
              f"   -> pick: {row.total_pick}"
              + (f"  edge {row.total_edge:+.1f}" if row.total_pick != "PASS" else ""))

    print(f"\n{'='*100}")
    print("Pick summary:")
    print(f"  ML:     {preds['ml_pick'].value_counts().to_dict()}")
    print(f"  Spread: {preds['spread_pick'].value_counts().to_dict()}")
    print(f"  Total:  {preds['total_pick'].value_counts().to_dict()}")

    print(f"\n[done] Step 7 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())