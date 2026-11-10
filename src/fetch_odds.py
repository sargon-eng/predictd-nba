"""
Step 6 — Fetch live NBA odds from The Odds API.

Reads:  .env (ODDS_API_KEY)
Writes: data/raw/odds/odds_YYYYMMDD_HHMMSS.json   (raw API response)
        data/processed/odds_latest.csv            (parsed one-row-per-game)

Bookmaker preference: Pinnacle first, then FanDuel, then DraftKings,
then whatever is available.

Run:
  python src\\fetch_odds.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# PATHS + CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "odds"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("ODDS_API_KEY", "").strip()
SPORT_KEY = "basketball_nba"
REGIONS = "us,eu"               # us for US books, eu for Pinnacle
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "american"
PREFERRED_BOOKS = ["pinnacle", "fanduel", "draftkings"]
API_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------
def fetch_odds() -> requests.Response:
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    return requests.get(API_URL, params=params, timeout=30)


# ---------------------------------------------------------------------------
# PARSE
# ---------------------------------------------------------------------------
def pick_bookmaker(game: dict) -> dict | None:
    books = {b["key"]: b for b in game.get("bookmakers", [])}
    for key in PREFERRED_BOOKS:
        if key in books:
            return books[key]
    return next(iter(books.values())) if books else None


def parse_game(game: dict) -> dict:
    row = {
        "game_id_api": game["id"],
        "commence_time": game["commence_time"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "bookmaker": None,
        "home_ml": None, "away_ml": None,
        "home_spread": None, "home_spread_price": None,
        "away_spread": None, "away_spread_price": None,
        "total": None, "over_price": None, "under_price": None,
    }

    book = pick_bookmaker(game)
    if not book:
        return row

    row["bookmaker"] = book["key"]

    for market in book.get("markets", []):
        key = market["key"]
        for o in market.get("outcomes", []):
            name = o["name"]
            if key == "h2h":
                if name == game["home_team"]:
                    row["home_ml"] = o["price"]
                elif name == game["away_team"]:
                    row["away_ml"] = o["price"]
            elif key == "spreads":
                if name == game["home_team"]:
                    row["home_spread"] = o["point"]
                    row["home_spread_price"] = o["price"]
                elif name == game["away_team"]:
                    row["away_spread"] = o["point"]
                    row["away_spread_price"] = o["price"]
            elif key == "totals":
                if name == "Over":
                    row["total"] = o["point"]
                    row["over_price"] = o["price"]
                elif name == "Under":
                    row["under_price"] = o["price"]

    return row


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    if not API_KEY or API_KEY == "put_your_key_here_later":
        print("ERROR: ODDS_API_KEY is not set in .env")
        print(f"       Looked at: {PROJECT_ROOT / '.env'}")
        print("       Open .env with Notepad and paste your real key.")
        return 1

    print("Fetching NBA odds from The Odds API ...")
    print(f"  sport:    {SPORT_KEY}")
    print(f"  markets:  {MARKETS}")
    print(f"  regions:  {REGIONS}")

    try:
        r = fetch_odds()
    except requests.RequestException as e:
        print(f"Request failed: {e!r}")
        return 1

    remaining = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")

    if r.status_code == 401:
        print("HTTP 401 — API key rejected.")
        print(f"  response: {r.text[:300]}")
        return 1
    if r.status_code == 422:
        print("HTTP 422 — invalid parameters (check sport key / markets).")
        print(f"  response: {r.text[:300]}")
        return 1
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"  response: {r.text[:300]}")
        return 1

    print(f"  requests remaining: {remaining}")
    print(f"  requests used:      {used}")

    games = r.json()
    print(f"  games returned:     {len(games)}")

    if not games:
        print("\nNo upcoming NBA games available right now.")
        print("(This is normal during the offseason or when no games are scheduled.)")
        return 0

    # Save raw
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = RAW_DIR / f"odds_{ts}.json"
    raw_path.write_text(json.dumps(games, indent=2))
    print(f"\n  wrote raw: {raw_path.relative_to(PROJECT_ROOT)}")

    # Parse to CSV
    rows = [parse_game(g) for g in games]
    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "odds_latest.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote parsed: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  parsed shape: {df.shape}")

    # Pretty print
    print("\nUpcoming games:")
    show_cols = [
        "commence_time", "away_team", "home_team",
        "bookmaker", "away_ml", "home_ml",
        "home_spread", "total",
    ]
    printable = df[show_cols].copy()
    printable["commence_time"] = pd.to_datetime(
        printable["commence_time"]
    ).dt.strftime("%a %b %d %H:%M UTC")
    print(printable.to_string(index=False))

    print("\nBookmaker used per game:")
    print(df["bookmaker"].value_counts(dropna=False).to_string())

    print(f"\n[done] Step 6 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())