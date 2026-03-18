import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

API_KEY = _get_secret("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = Path("models")

for d in [DATA_DIR, CACHE_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

# Competition configs
COMPETITIONS = {
    "PL": {
        "name": "Premier League",
        "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "fdc_file": "E0",
        "seasons": ["2021-22", "2022-23", "2023-24"],
    },
    "PD": {
        "name": "La Liga",
        "flag": "🇪🇸",
        "fdc_file": "SP1",
        "seasons": ["2021-22", "2022-23", "2023-24"],
    },
    "CL": {
        "name": "Champions League",
        "flag": "🏆",
        "fdc_file": None,
        "seasons": [],
    },
}

HIST_SEASONS = {
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
}

FDC_BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Premier League — football-data.co.uk short names → canonical
PL_FDC_MAP = {
    "Man United": "Manchester United", "Man City": "Manchester City",
    "Wolves": "Wolverhampton Wanderers", "Spurs": "Tottenham Hotspur",
    "Brighton": "Brighton & Hove Albion", "Leicester": "Leicester City",
    "West Ham": "West Ham United", "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest", "Crystal Palace": "Crystal Palace",
    "Brentford": "Brentford", "Everton": "Everton", "Chelsea": "Chelsea",
    "Aston Villa": "Aston Villa", "Arsenal": "Arsenal", "Liverpool": "Liverpool",
    "Fulham": "Fulham", "Bournemouth": "Bournemouth", "Southampton": "Southampton",
    "Ipswich": "Ipswich Town", "Leeds": "Leeds United", "Norwich": "Norwich City",
    "Watford": "Watford", "Burnley": "Burnley", "Sheffield United": "Sheffield United",
    "Luton": "Luton Town",
}

# La Liga — football-data.co.uk names → canonical
PD_FDC_MAP = {
    "Celta": "Celta Vigo", "Alaves": "Alaves", "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid", "Barcelona": "Barcelona",
    "Betis": "Real Betis", "Cadiz": "Cadiz", "Espanol": "Espanyol",
    "Getafe": "Getafe", "Girona": "Girona", "Granada": "Granada",
    "Las Palmas": "Las Palmas", "Leganes": "Leganes", "Mallorca": "Mallorca",
    "Osasuna": "Osasuna", "Rayo Vallecano": "Rayo Vallecano",
    "Real Madrid": "Real Madrid", "Real Sociedad": "Real Sociedad",
    "Sevilla": "Sevilla", "Valencia": "Valencia", "Valladolid": "Valladolid",
    "Villarreal": "Villarreal", "Almeria": "Almeria",
}

# Premier League — football-data.org full names → canonical
PL_FDO_MAP = {
    "Manchester United FC": "Manchester United", "Manchester City FC": "Manchester City",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Leicester City FC": "Leicester City", "West Ham United FC": "West Ham United",
    "Newcastle United FC": "Newcastle United", "Nottingham Forest FC": "Nottingham Forest",
    "Crystal Palace FC": "Crystal Palace", "Brentford FC": "Brentford",
    "Everton FC": "Everton", "Chelsea FC": "Chelsea", "Aston Villa FC": "Aston Villa",
    "Arsenal FC": "Arsenal", "Liverpool FC": "Liverpool", "Fulham FC": "Fulham",
    "AFC Bournemouth": "Bournemouth", "Southampton FC": "Southampton",
    "Ipswich Town FC": "Ipswich Town", "Leeds United FC": "Leeds United",
    "Luton Town FC": "Luton Town", "Sheffield United FC": "Sheffield United",
    "Burnley FC": "Burnley", "Watford FC": "Watford", "Norwich City FC": "Norwich City",
}

# La Liga — football-data.org full names → canonical
PD_FDO_MAP = {
    "FC Barcelona": "Barcelona", "Real Madrid CF": "Real Madrid",
    "Club Atlético de Madrid": "Atletico Madrid", "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia", "Villarreal CF": "Villarreal",
    "Athletic Club": "Athletic Club", "Real Sociedad de Fútbol": "Real Sociedad",
    "Real Betis Balompié": "Real Betis", "RC Celta de Vigo": "Celta Vigo",
    "Girona FC": "Girona", "Getafe CF": "Getafe", "RCD Mallorca": "Mallorca",
    "UD Las Palmas": "Las Palmas", "CA Osasuna": "Osasuna",
    "Deportivo Alavés": "Alaves", "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "CD Leganés": "Leganes", "RCD Espanyol de Barcelona": "Espanyol",
    "Real Valladolid CF": "Valladolid", "Cádiz CF": "Cadiz",
    "UD Almería": "Almeria", "Granada CF": "Granada",
}

# Champions League — football-data.org → canonical (common teams)
CL_FDO_MAP = {
    "FC Bayern München": "Bayern Munich", "Borussia Dortmund": "Borussia Dortmund",
    "Paris Saint-Germain FC": "PSG", "Juventus FC": "Juventus",
    "FC Internazionale Milano": "Inter Milan", "AC Milan": "AC Milan",
    "SSC Napoli": "Napoli", "AS Roma": "Roma", "SS Lazio": "Lazio",
    "SL Benfica": "Benfica", "FC Porto": "Porto", "Sporting CP": "Sporting CP",
    "Ajax Amsterdam": "Ajax", "PSV Eindhoven": "PSV", "Feyenoord Rotterdam": "Feyenoord",
    "Celtic FC": "Celtic", "Rangers FC": "Rangers",
    "Shakhtar Donetsk": "Shakhtar Donetsk", "Dynamo Kyiv": "Dynamo Kyiv",
    "GNK Dinamo Zagreb": "Dinamo Zagreb", "HNK Rijeka": "Rijeka",
    "Club Brugge KV": "Club Brugge", "RSC Anderlecht": "Anderlecht",
    "Bayer 04 Leverkusen": "Bayer Leverkusen", "RB Leipzig": "RB Leipzig",
    "FC Salzburg": "Red Bull Salzburg", "FK Austria Wien": "Austria Wien",
    "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco", "Stade Rennais FC 1901": "Rennes",
    "LOSC Lille": "Lille", "RC Lens": "Lens",
    "Eintracht Frankfurt": "Eintracht Frankfurt", "VfB Stuttgart": "Stuttgart",
    "FC Augsburg": "Augsburg", "TSG 1899 Hoffenheim": "Hoffenheim",
    "Sport Lisboa e Benfica": "Benfica",
    "Manchester United FC": "Manchester United", "Manchester City FC": "Manchester City",
    "Arsenal FC": "Arsenal", "Liverpool FC": "Liverpool", "Chelsea FC": "Chelsea",
    "Tottenham Hotspur FC": "Tottenham Hotspur", "Aston Villa FC": "Aston Villa",
    "FC Barcelona": "Barcelona", "Real Madrid CF": "Real Madrid",
    "Club Atlético de Madrid": "Atletico Madrid", "Sevilla FC": "Sevilla",
    "Villarreal CF": "Villarreal", "Athletic Club": "Athletic Club",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "FC Internazionale Milano": "Inter Milan",
}


def normalize_team(name: str, competition: str = "PL", source: str = "fdc") -> str:
    """Normalize team name to canonical form."""
    if source == "fdc":
        if competition == "PL":
            return PL_FDC_MAP.get(name, name)
        elif competition == "PD":
            return PD_FDC_MAP.get(name, name)
        return name
    elif source == "fdo":
        # Try all maps in order
        for m in [PL_FDO_MAP, PD_FDO_MAP, CL_FDO_MAP]:
            if name in m:
                return m[name]
        # Fallback: strip common suffixes
        clean = name
        for suffix in [" FC", " CF", " AC", " SC", " SK", " BV", " SV", " AF"]:
            clean = clean.replace(suffix, "")
        return clean.strip()
    return name
