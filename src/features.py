import pandas as pd
import numpy as np
from typing import Optional


def _team_matches(matches: pd.DataFrame, team: str, before_date, venue: str = "all"):
    """Get a team's matches before a date, optionally filtered by venue."""
    if venue == "home":
        mask = (matches["HomeTeam"] == team) & (matches["Date"] < before_date)
        df = matches[mask].copy()
        df["GF"] = df["FTHG"]
        df["GA"] = df["FTAG"]
        df["Points"] = df["FTR"].map({"H": 3, "D": 1, "A": 0})
    elif venue == "away":
        mask = (matches["AwayTeam"] == team) & (matches["Date"] < before_date)
        df = matches[mask].copy()
        df["GF"] = df["FTAG"]
        df["GA"] = df["FTHG"]
        df["Points"] = df["FTR"].map({"A": 3, "D": 1, "H": 0})
    else:
        home = matches[(matches["HomeTeam"] == team) & (matches["Date"] < before_date)].copy()
        away = matches[(matches["AwayTeam"] == team) & (matches["Date"] < before_date)].copy()
        home["GF"] = home["FTHG"]
        home["GA"] = home["FTAG"]
        home["Points"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})
        away["GF"] = away["FTAG"]
        away["GA"] = away["FTHG"]
        away["Points"] = away["FTR"].map({"A": 3, "D": 1, "H": 0})
        df = pd.concat([home, away]).sort_values("Date")

    return df.sort_values("Date")


def calculate_form(matches: pd.DataFrame, team: str, before_date, n: int = 5, venue: str = "all") -> dict:
    """Calculate rolling form stats for a team before a given date."""
    df = _team_matches(matches, team, before_date, venue).tail(n)

    if df.empty:
        return {"pts": 0.0, "gf": 0.0, "ga": 0.0, "gd": 0.0, "wins": 0, "games": 0}

    pts = df["Points"].sum()
    gf = df["GF"].mean()
    ga = df["GA"].mean()
    wins = (df["Points"] == 3).sum()

    return {
        "pts": float(pts),
        "gf": float(gf),
        "ga": float(ga),
        "gd": float(gf - ga),
        "wins": int(wins),
        "games": len(df),
    }


def get_h2h(matches: pd.DataFrame, home_team: str, away_team: str, before_date, n: int = 6) -> dict:
    """Head-to-head record between two teams."""
    mask = (
        (
            ((matches["HomeTeam"] == home_team) & (matches["AwayTeam"] == away_team))
            | ((matches["HomeTeam"] == away_team) & (matches["AwayTeam"] == home_team))
        )
        & (matches["Date"] < before_date)
    )
    h2h = matches[mask].tail(n)

    if h2h.empty:
        return {"h2h_hw": 0.0, "h2h_d": 0.0, "h2h_aw": 0.0, "h2h_n": 0}

    hw = draw = aw = 0
    for _, row in h2h.iterrows():
        if row["HomeTeam"] == home_team:
            if row["FTR"] == "H":
                hw += 1
            elif row["FTR"] == "D":
                draw += 1
            else:
                aw += 1
        else:
            if row["FTR"] == "A":
                hw += 1
            elif row["FTR"] == "D":
                draw += 1
            else:
                aw += 1

    ng = len(h2h)
    return {
        "h2h_hw": hw / ng,
        "h2h_d": draw / ng,
        "h2h_aw": aw / ng,
        "h2h_n": ng,
    }


def get_features_for_match(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    date,
    dc_model=None,
) -> dict:
    """Build feature dict for a single match."""
    hf = calculate_form(matches, home_team, date, n=5)
    af = calculate_form(matches, away_team, date, n=5)
    hfh = calculate_form(matches, home_team, date, n=5, venue="home")
    afa = calculate_form(matches, away_team, date, n=5, venue="away")
    h2h = get_h2h(matches, home_team, away_team, date)

    feat = {
        # Overall form
        "home_form_pts": hf["pts"],
        "away_form_pts": af["pts"],
        "form_pts_diff": hf["pts"] - af["pts"],
        "home_form_gf": hf["gf"],
        "away_form_gf": af["gf"],
        "home_form_ga": hf["ga"],
        "away_form_ga": af["ga"],
        "home_form_gd": hf["gd"],
        "away_form_gd": af["gd"],
        # Venue-specific form
        "home_venue_pts": hfh["pts"],
        "away_venue_pts": afa["pts"],
        "home_venue_gf": hfh["gf"],
        "away_venue_gf": afa["gf"],
        "home_venue_ga": hfh["ga"],
        "away_venue_ga": afa["ga"],
        # H2H
        "h2h_hw": h2h["h2h_hw"],
        "h2h_d": h2h["h2h_d"],
        "h2h_aw": h2h["h2h_aw"],
        "h2h_n": h2h["h2h_n"],
    }

    # Dixon-Coles features
    if dc_model is not None and dc_model.fitted:
        avg_atk = np.mean(list(dc_model.attack.values()))
        avg_def = np.mean(list(dc_model.defense.values()))
        probs = dc_model.predict_outcome_probs(home_team, away_team)
        feat.update({
            "dc_hw": probs["home_win"],
            "dc_d": probs["draw"],
            "dc_aw": probs["away_win"],
            "dc_home_atk": dc_model.attack.get(home_team, avg_atk),
            "dc_home_def": dc_model.defense.get(home_team, avg_def),
            "dc_away_atk": dc_model.attack.get(away_team, avg_atk),
            "dc_away_def": dc_model.defense.get(away_team, avg_def),
        })
    else:
        feat.update({
            "dc_hw": 0.45, "dc_d": 0.25, "dc_aw": 0.30,
            "dc_home_atk": 0.0, "dc_home_def": 0.0,
            "dc_away_atk": 0.0, "dc_away_def": 0.0,
        })

    return feat


def build_feature_matrix(matches: pd.DataFrame, dc_model=None) -> pd.DataFrame:
    """Build feature matrix for all completed matches (for model training)."""
    rows = []
    for _, row in matches.iterrows():
        feat = get_features_for_match(matches, row["HomeTeam"], row["AwayTeam"], row["Date"], dc_model)
        feat["Date"] = row["Date"]
        feat["HomeTeam"] = row["HomeTeam"]
        feat["AwayTeam"] = row["AwayTeam"]
        feat["FTHG"] = row["FTHG"]
        feat["FTAG"] = row["FTAG"]
        feat["result"] = {"H": 0, "D": 1, "A": 2}.get(row["FTR"], -1)
        rows.append(feat)

    df = pd.DataFrame(rows)
    return df[df["result"] >= 0].reset_index(drop=True)


FEATURE_COLS = [
    "home_form_pts", "away_form_pts", "form_pts_diff",
    "home_form_gf", "away_form_gf",
    "home_form_ga", "away_form_ga",
    "home_form_gd", "away_form_gd",
    "home_venue_pts", "away_venue_pts",
    "home_venue_gf", "away_venue_gf",
    "home_venue_ga", "away_venue_ga",
    "h2h_hw", "h2h_d", "h2h_aw", "h2h_n",
    "dc_hw", "dc_d", "dc_aw",
    "dc_home_atk", "dc_home_def",
    "dc_away_atk", "dc_away_def",
]
