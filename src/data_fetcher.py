import requests
import pandas as pd
import json
import time
from pathlib import Path
from src.config import (
    API_KEY, BASE_URL, COMPETITIONS, DATA_DIR, CACHE_DIR,
    HIST_SEASONS, FDC_BASE_URL, normalize_team
)


def _get_headers():
    return {"X-Auth-Token": API_KEY}


def fetch_api(endpoint: str, params: dict = None) -> dict:
    """Fetch from football-data.org with 1-hour cache."""
    cache_key = endpoint.replace("/", "_") + (str(sorted(params.items())) if params else "")
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 3600:
            with open(cache_file) as f:
                return json.load(f)

    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=_get_headers(), params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data


def load_historical_data(competition: str = "PL") -> pd.DataFrame:
    """Download and parse historical data from football-data.co.uk."""
    cfg = COMPETITIONS.get(competition, {})
    fdc_file = cfg.get("fdc_file")
    seasons = cfg.get("seasons", [])

    if not fdc_file or not seasons:
        return pd.DataFrame()

    all_matches = []
    for season in seasons:
        code = HIST_SEASONS.get(season)
        if not code:
            continue

        local_path = DATA_DIR / f"{competition}_{season}.csv"

        if not local_path.exists():
            url = f"{FDC_BASE_URL}/{code}/{fdc_file}.csv"
            try:
                df = pd.read_csv(url)
                df.to_csv(local_path, index=False)
                print(f"Downloaded {competition} {season}")
            except Exception as e:
                print(f"Failed to download {competition} {season}: {e}")
                continue
        else:
            df = pd.read_csv(local_path)

        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        if not all(c in df.columns for c in required):
            continue

        df = df[required].copy()
        df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        df["HomeTeam"] = df["HomeTeam"].apply(lambda x: normalize_team(str(x), competition, "fdc"))
        df["AwayTeam"] = df["AwayTeam"].apply(lambda x: normalize_team(str(x), competition, "fdc"))
        df["Season"] = season
        df["Competition"] = competition

        parsed = False
        for fmt in ["%d/%m/%Y", "%d/%m/%y"]:
            try:
                df["Date"] = pd.to_datetime(df["Date"], format=fmt)
                parsed = True
                break
            except Exception:
                continue
        if not parsed:
            try:
                df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
            except Exception:
                print(f"Could not parse dates for {competition} {season}")
                continue

        all_matches.append(df)

    if not all_matches:
        return pd.DataFrame()

    result = pd.concat(all_matches, ignore_index=True)
    return result.sort_values("Date").reset_index(drop=True)


def fetch_current_season_results(competition: str = "PL") -> pd.DataFrame:
    """Fetch finished matches from football-data.org API."""
    try:
        data = fetch_api(f"competitions/{competition}/matches", {"status": "FINISHED"})
        matches = []
        for m in data.get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            if score.get("home") is None:
                continue
            hg, ag = score["home"], score["away"]
            matches.append({
                "Date": pd.to_datetime(m["utcDate"]).tz_localize(None),
                "HomeTeam": normalize_team(m["homeTeam"]["name"], competition, "fdo"),
                "AwayTeam": normalize_team(m["awayTeam"]["name"], competition, "fdo"),
                "FTHG": int(hg),
                "FTAG": int(ag),
                "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
                "Season": "2024-25",
                "Competition": competition,
            })
        df = pd.DataFrame(matches)
        if not df.empty:
            df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Failed to fetch {competition} results: {e}")
        return pd.DataFrame()


def fetch_upcoming_fixtures(competition: str = "PL") -> pd.DataFrame:
    """Fetch upcoming fixtures — includes SCHEDULED and TIMED (today's matches)."""
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    future = (pd.Timestamp.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    matches = []
    try:
        # Use date range so we catch TIMED (today) + SCHEDULED (future) statuses
        data = fetch_api(
            f"competitions/{competition}/matches",
            {"dateFrom": today, "dateTo": future}
        )
        for m in data.get("matches", []):
            status = m.get("status", "")
            # Skip already finished or live matches
            if status in ("FINISHED", "IN_PLAY", "PAUSED", "CANCELLED", "POSTPONED", "SUSPENDED"):
                continue
            matches.append({
                "match_id": m["id"],
                "date": pd.to_datetime(m["utcDate"]).tz_localize(None),
                "matchday": m.get("matchday"),
                "home_team": normalize_team(m["homeTeam"]["name"], competition, "fdo"),
                "away_team": normalize_team(m["awayTeam"]["name"], competition, "fdo"),
                "status": status,
            })
    except Exception as e:
        print(f"Failed to fetch {competition} fixtures: {e}")

    df = pd.DataFrame(matches)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def get_standings(competition: str = "PL") -> pd.DataFrame:
    """Fetch current standings."""
    try:
        data = fetch_api(f"competitions/{competition}/standings")
        table = []
        for standing in data.get("standings", []):
            if standing.get("type") == "TOTAL":
                for row in standing.get("table", []):
                    table.append({
                        "Pos": row["position"],
                        "Team": normalize_team(row["team"]["name"], competition, "fdo"),
                        "P": row["playedGames"],
                        "W": row["won"],
                        "D": row["draw"],
                        "L": row["lost"],
                        "GF": row["goalsFor"],
                        "GA": row["goalsAgainst"],
                        "GD": row["goalDifference"],
                        "Pts": row["points"],
                    })
        return pd.DataFrame(table)
    except Exception as e:
        print(f"Failed to fetch {competition} standings: {e}")
        return pd.DataFrame()


def get_all_results(competition: str = "PL") -> pd.DataFrame:
    """Combine historical CSVs + current season API data."""
    hist = load_historical_data(competition)
    current = fetch_current_season_results(competition)

    if hist.empty and current.empty:
        return pd.DataFrame()
    elif hist.empty:
        return current
    elif current.empty:
        return hist

    combined = pd.concat([hist, current], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return combined.sort_values("Date").reset_index(drop=True)


def get_todays_results(competition: str = "PL") -> pd.DataFrame:
    """Get matches already played today for same-day momentum."""
    try:
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        data = fetch_api(f"competitions/{competition}/matches", {"dateFrom": today, "dateTo": today})
        matches = []
        for m in data.get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            status = m.get("status", "")
            matches.append({
                "home": normalize_team(m["homeTeam"]["name"], competition, "fdo"),
                "away": normalize_team(m["awayTeam"]["name"], competition, "fdo"),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "status": status,
                "finished": status == "FINISHED",
            })
        return pd.DataFrame(matches)
    except Exception:
        return pd.DataFrame()
