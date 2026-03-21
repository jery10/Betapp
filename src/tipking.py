"""
Fetches crowd prediction signals from TipKing for use in BetPredict.
"""
import os
import requests

TIPKING_URL = os.getenv("TIPKING_URL", "https://web-production-2b6bf.up.railway.app")


def get_crowd_signals(home_team: str, away_team: str, timeout: int = 6) -> dict | None:
    """
    Fetch crowd consensus for a match from TipKing.
    Returns None if TipKing is unreachable or no data exists.
    """
    try:
        url = f"{TIPKING_URL}/api/match/{requests.utils.quote(home_team)}/vs/{requests.utils.quote(away_team)}"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("total_predictions", 0) == 0:
            return None
        return data
    except Exception:
        return None
