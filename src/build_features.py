"""
Step 4 — Feature engineering.

Reads:  data/processed/games.csv
Writes: data/processed/features.csv

What it does:
  1. Filters out preseason and all-star games.
  2. Dedupes by game_id.
  3. Iterates chronologically over every game and, for each, records
     PRE-GAME features for home and away teams with NO data leakage:
        - Elo rating (with home advantage)
        - Rest days & back-to-back flag
        - Season-to-date win % (overall, plus home-only / away-only)
        - Rolling last-5 / last-10 win % and scoring averages

Targets saved alongside: home_win, margin, total_pts.

Run:
  python src\\build_features.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "games.csv"
OUTPUT_PATH = PROCESSED_DIR / "features.csv"


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ELO_START = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE_ELO = 100.0       # added to home team's Elo for expected calc
ROLL_WINDOW_SHORT = 5
ROLL_WINDOW_LONG = 10
SEASON_OPENER_REST_DAYS = 7      # assume a week off before opener

# NBA game_id prefixes
KEEP_PREFIXES = {"002", "004", "005"}
PREFIX_TO_TYPE = {"002": "regular", "004": "playoffs", "005": "playin"}


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


def normalize_game_id(raw) -> str:
    """Game IDs are 10-digit strings with a 3-digit prefix (e.g. '0022100001').
    CSV round-trips sometimes strip the leading zeros — zfill them back."""
    s = str(raw).split(".")[0]          # drop any '.0' from float coercion
    return s.zfill(10)


# ---------------------------------------------------------------------------
# FEATURE BUILDER
# ---------------------------------------------------------------------------
def build_features(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values("game_date").reset_index(drop=True)

    elo: dict[int, float] = {}
    last_game_date: dict[int, pd.Timestamp] = {}

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

    rows = []

    for g in games.itertuples(index=False):
        hid = int(g.home_team_id)
        aid = int(g.away_team_id)
        season = g.season
        gdate = g.game_date
        actual_home = int(g.home_win)
        margin = int(g.margin)
        home_pts = int(g.home_pts)
        away_pts = int(g.away_pts)

        home_elo = elo.get(hid, ELO_START)
        away_elo = elo.get(aid, ELO_START)

        def rest_days(team_id):
            if team_id not in last_game_date:
                return SEASON_OPENER_REST_DAYS
            return (gdate - last_game_date[team_id]).days

        home_rest = rest_days(hid)
        away_rest = rest_days(aid)

        h_sw = season_wins.get((hid, season), 0)
        h_sg = season_games.get((hid, season), 0)
        a_sw = season_wins.get((aid, season), 0)
        a_sg = season_games.get((aid, season), 0)

        h_hw = home_wins.get((hid, season), 0)
        h_hg = home_games.get((hid, season), 0)
        a_aw = away_wins.get((aid, season), 0)
        a_ag = away_games.get((aid, season), 0)

        prefix = str(g.game_id)[:3]
        season_type = PREFIX_TO_TYPE.get(prefix, "other")

        rows.append({
            "game_id": g.game_id,
            "game_date": gdate,
            "season": season,
            "season_type": season_type,
            "home_team_id": hid,
            "away_team_id": aid,
            "home_team": g.home_team,
            "away_team": g.away_team,

            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "elo_diff_pre": home_elo - away_elo,
            "elo_home_win_prob_pre": expected_home_win(home_elo, away_elo),

            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": home_rest - away_rest,
            "home_b2b": int(home_rest <= 1),
            "away_b2b": int(away_rest <= 1),

            "home_season_win_pct": win_pct(h_sw, h_sg),
            "away_season_win_pct": win_pct(a_sw, a_sg),
            "home_home_win_pct":   win_pct(h_hw, h_hg),
            "away_away_win_pct":   win_pct(a_aw, a_ag),

            "home_win_pct_last5":  mean_of_last(recent_results, hid, 5),
            "away_win_pct_last5":  mean_of_last(recent_results, aid, 5),
            "home_win_pct_last10": mean_of_last(recent_results, hid, 10),
            "away_win_pct_last10": mean_of_last(recent_results, aid, 10),
            "home_avg_margin_last5":      mean_of_last(recent_margins, hid, 5),
            "away_avg_margin_last5":      mean_of_last(recent_margins, aid, 5),
            "home_avg_pts_for_last5":     mean_of_last(recent_pts_for, hid, 5),
            "away_avg_pts_for_last5":     mean_of_last(recent_pts_for, aid, 5),
            "home_avg_pts_against_last5": mean_of_last(recent_pts_against, hid, 5),
            "away_avg_pts_against_last5": mean_of_last(recent_pts_against, aid, 5),

            "home_win": actual_home,
            "margin": margin,
            "total_pts": home_pts + away_pts,
        })

        # ---------- POST-GAME UPDATES ----------
        exp_home = expected_home_win(home_elo, away_elo)
        elo[hid] = home_elo + ELO_K * (actual_home - exp_home)
        elo[aid] = away_elo + ELO_K * ((1 - actual_home) - (1 - exp_home))

        last_game_date[hid] = gdate
        last_game_date[aid] = gdate

        season_games[(hid, season)] = h_sg + 1
        season_games[(aid, season)] = a_sg + 1
        season_wins[(hid, season)] = h_sw + actual_home
        season_wins[(aid, season)] = a_sw + (1 - actual_home)

        home_games[(hid, season)] = h_hg + 1
        home_wins[(hid, season)] = h_hw + actual_home
        away_games[(aid, season)] = a_ag + 1
        away_wins[(aid, season)] = a_aw + (1 - actual_home)

        recent_results.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(actual_home)
        recent_results.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(1 - actual_home)

        recent_margins.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(margin)
        recent_margins.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(-margin)

        recent_pts_for.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(home_pts)
        recent_pts_for.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(away_pts)

        recent_pts_against.setdefault(hid, deque(maxlen=ROLL_WINDOW_LONG)).append(away_pts)
        recent_pts_against.setdefault(aid, deque(maxlen=ROLL_WINDOW_LONG)).append(home_pts)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Loading {INPUT_PATH}")
    games = pd.read_csv(
        INPUT_PATH,
        parse_dates=["game_date"],
        dtype={"game_id": str},
    )
    print(f"  raw rows: {len(games)}")

    # Normalize game_id (add back any leading zeros stripped by CSV round-trip)
    games["game_id"] = games["game_id"].map(normalize_game_id)

    # Show breakdown by game_id prefix BEFORE filtering
    prefix = games["game_id"].str[:3]
    for p in ["001", "002", "003", "004", "005"]:
        print(f"  prefix {p}: {int((prefix == p).sum())}")

    games = games[prefix.isin(KEEP_PREFIXES)].copy()
    print(f"  after removing preseason/all-star: {len(games)}")

    dupes = int(games["game_id"].duplicated().sum())
    if dupes:
        print(f"  removing {dupes} duplicate game_ids")
        games = games.drop_duplicates(subset="game_id", keep="first")
    print(f"  after dedupe: {len(games)}")

    if games.empty:
        print("No games left after filtering. Aborting.")
        return 1

    print("\nBuilding features (this is fast, ~5 seconds) ...")
    features = build_features(games)
    print(f"  features shape: {features.shape}")

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[done ] wrote {OUTPUT_PATH}")

    print("\nSanity check:")
    print(f"  games:            {len(features)}")
    print(f"  date range:       {features['game_date'].min().date()} -> "
          f"{features['game_date'].max().date()}")
    print(f"  home win rate:    {features['home_win'].mean():.3f}")
    print(f"  avg total pts:    {features['total_pts'].mean():.1f}")
    print(f"  avg margin:       {features['margin'].mean():+.2f}")
    print(f"  Elo range (pre):  {features['home_elo_pre'].min():.0f} - "
          f"{features['home_elo_pre'].max():.0f}")
    print(f"  rows with rolling last-5 data: "
          f"{int(features['home_win_pct_last5'].notna().sum())}")

    print("\nBreakdown by season_type:")
    print(features["season_type"].value_counts().to_string())

    print("\nFirst 5 rows (key columns):")
    cols = ["game_date", "home_team", "away_team",
            "home_elo_pre", "away_elo_pre",
            "home_rest_days", "away_rest_days",
            "home_win_pct_last5", "away_win_pct_last5",
            "home_win"]
    print(features[cols].head().to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())