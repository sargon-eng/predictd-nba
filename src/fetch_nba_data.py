"""
Step 3 — Download historical NBA game data via nba_api.

What this does:
  1. For each season in SEASONS, calls LeagueGameFinder to get one row
     per team per game.
  2. Saves the raw response to:  data/raw/nba/team_games_<season>.csv
  3. Builds a clean one-row-per-game DataFrame (home/away joined)
     and saves it to:           data/processed/games.csv

Resumable:
  If data/raw/nba/team_games_<season>.csv already exists, that season is
  skipped. Delete the file to re-fetch it.

Run:
  python src\\fetch_nba_data.py
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import pandas as pd

try:
    from nba_api.stats.endpoints import leaguegamefinder
except ImportError:
    print("ERROR: nba_api is not installed. Run: pip install nba_api")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

REQUEST_DELAY_SECONDS = 0.6  # be polite to the NBA API


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nba"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------
def fetch_season_raw(season: str) -> pd.DataFrame:
    """Fetch one season of team-game rows from LeagueGameFinder."""
    print(f"[fetch] {season}: calling LeagueGameFinder ...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # "00" = NBA
    )
    df = finder.get_data_frames()[0]
    print(f"[fetch] {season}: got {len(df)} team-game rows")
    return df


def load_or_fetch_season(season: str) -> pd.DataFrame:
    """Return raw team-game rows for one season, using cache if present."""
    cache_path = RAW_DIR / f"team_games_{season}.csv"
    if cache_path.exists():
        print(f"[cache] {season}: using {cache_path.name}")
        df = pd.read_csv(cache_path)
    else:
        df = fetch_season_raw(season)
        df.to_csv(cache_path, index=False)
        print(f"[save ] {season}: wrote {cache_path.name}")
        time.sleep(REQUEST_DELAY_SECONDS)
    df["SEASON_LABEL"] = season
    return df


# ---------------------------------------------------------------------------
# BUILD GAMES TABLE
# ---------------------------------------------------------------------------
def col(df: pd.DataFrame, name: str):
    """Return a column if it exists, otherwise a NaN series of the right length."""
    if name in df.columns:
        return df[name]
    return pd.Series([pd.NA] * len(df), index=df.index)


def build_games_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert team-game rows into one row per game (home + away joined)."""
    df = raw.copy()
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)

    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    # Sanity check — each GAME_ID should have exactly one home and one away row
    dupes = home["GAME_ID"].duplicated().sum() + away["GAME_ID"].duplicated().sum()
    if dupes:
        print(f"[warn ] {dupes} duplicated GAME_IDs across home/away rows")

    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))

    dates = pd.to_datetime(games["GAME_DATE_home"], errors="coerce")
    if dates.isna().any():
        print(f"[warn ] {int(dates.isna().sum())} rows have unparseable date")

    out = pd.DataFrame({
        "game_id":         games["GAME_ID"],
        "game_date":       dates,
        "season":          games["SEASON_LABEL_home"],
        "home_team_id":    games["TEAM_ID_home"],
        "home_team":       games["TEAM_NAME_home"],
        "home_abbr":       games["TEAM_ABBREVIATION_home"],
        "away_team_id":    games["TEAM_ID_away"],
        "away_team":       games["TEAM_NAME_away"],
        "away_abbr":       games["TEAM_ABBREVIATION_away"],
        "home_pts":        games["PTS_home"],
        "away_pts":        games["PTS_away"],
        "home_win":        (games["WL_home"] == "W").astype(int),
        # home box score
        "home_fga":        col(games, "FGA_home"),
        "home_fg3a":       col(games, "FG3A_home"),
        "home_fta":        col(games, "FTA_home"),
        "home_tov":        col(games, "TOV_home"),
        "home_reb":        col(games, "REB_home"),
        # away box score
        "away_fga":        col(games, "FGA_away"),
        "away_fg3a":       col(games, "FG3A_away"),
        "away_fta":        col(games, "FTA_away"),
        "away_tov":        col(games, "TOV_away"),
        "away_reb":        col(games, "REB_away"),
    })

    out = out.dropna(subset=["home_pts", "away_pts", "game_date"]).copy()
    out["home_pts"]  = out["home_pts"].astype(int)
    out["away_pts"]  = out["away_pts"].astype(int)
    out["total_pts"] = out["home_pts"] + out["away_pts"]
    out["margin"]    = out["home_pts"] - out["away_pts"]
    out = out.sort_values("game_date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw dir:      {RAW_DIR}")
    print(f"Seasons:      {SEASONS}")
    print("-" * 60)

    all_raw = []
    for season in SEASONS:
        try:
            all_raw.append(load_or_fetch_season(season))
        except Exception as e:
            print(f"[error] {season}: {e!r}")
            print("        Skipping. Re-run the script to retry this season.")

    if not all_raw:
        print("No data fetched. Aborting.")
        return 1

    raw = pd.concat(all_raw, ignore_index=True)
    print(f"\n[build] total team-game rows: {len(raw)}")

    games = build_games_table(raw)
    print(f"[build] games: {len(games)}")
    print(f"[build] date range: {games['game_date'].min().date()} -> "
          f"{games['game_date'].max().date()}")
    print(f"[build] seasons: {sorted(games['season'].dropna().unique().tolist())}")

    out_path = PROCESSED_DIR / "games.csv"
    games.to_csv(out_path, index=False)
    print(f"\n[done ] wrote {out_path}")
    print(f"[done ] shape: {games.shape}")

    # Peek at the first few rows so you can eyeball them
    print("\nFirst 5 rows:")
    print(games.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())