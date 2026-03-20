import pandas as pd
import numpy as np


def _team_matches(matches: pd.DataFrame, team: str, before_date, venue: str = "all"):
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
        home["GF"] = home["FTHG"]; home["GA"] = home["FTAG"]
        home["Points"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})
        away["GF"] = away["FTAG"]; away["GA"] = away["FTHG"]
        away["Points"] = away["FTR"].map({"A": 3, "D": 1, "H": 0})
        df = pd.concat([home, away]).sort_values("Date")
    return df.sort_values("Date")


def calculate_form(matches: pd.DataFrame, team: str, before_date, n: int = 5, venue: str = "all") -> dict:
    df = _team_matches(matches, team, before_date, venue).tail(n)
    if df.empty:
        return {
            "pts": 0.0, "gf": 0.0, "ga": 0.0, "gd": 0.0,
            "wins": 0, "games": 0, "cs_rate": 0.0, "scoring_rate": 0.0,
        }
    pts = df["Points"].sum()
    gf = df["GF"].mean()
    ga = df["GA"].mean()
    wins = (df["Points"] == 3).sum()
    cs_rate = (df["GA"] == 0).mean()          # clean sheet rate
    scoring_rate = (df["GF"] > 0).mean()      # scored in X% of games

    return {
        "pts": float(pts),
        "gf": float(gf),
        "ga": float(ga),
        "gd": float(gf - ga),
        "wins": int(wins),
        "games": len(df),
        "cs_rate": float(cs_rate),
        "scoring_rate": float(scoring_rate),
    }


def _days_since_last_match(matches: pd.DataFrame, team: str, before_date) -> float:
    """Days since team last played (freshness / fatigue signal)."""
    past = matches[
        ((matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)) &
        (matches["Date"] < before_date)
    ]
    if past.empty:
        return 7.0  # assume rested
    last = past["Date"].max()
    return float((pd.Timestamp(before_date) - last).days)


def get_h2h(matches: pd.DataFrame, home_team: str, away_team: str, before_date, n: int = 6) -> dict:
    mask = (
        (
            ((matches["HomeTeam"] == home_team) & (matches["AwayTeam"] == away_team))
            | ((matches["HomeTeam"] == away_team) & (matches["AwayTeam"] == home_team))
        )
        & (matches["Date"] < before_date)
    )
    h2h = matches[mask].tail(n)
    if h2h.empty:
        return {"h2h_hw": 0.0, "h2h_d": 0.0, "h2h_aw": 0.0, "h2h_n": 0, "h2h_home_avg_gf": 0.0, "h2h_away_avg_gf": 0.0}

    hw = draw = aw = 0
    home_gf_list, away_gf_list = [], []
    for _, row in h2h.iterrows():
        if row["HomeTeam"] == home_team:
            home_gf_list.append(row["FTHG"])
            away_gf_list.append(row["FTAG"])
            if row["FTR"] == "H": hw += 1
            elif row["FTR"] == "D": draw += 1
            else: aw += 1
        else:
            home_gf_list.append(row["FTAG"])
            away_gf_list.append(row["FTHG"])
            if row["FTR"] == "A": hw += 1
            elif row["FTR"] == "D": draw += 1
            else: aw += 1

    ng = len(h2h)
    return {
        "h2h_hw": hw / ng,
        "h2h_d": draw / ng,
        "h2h_aw": aw / ng,
        "h2h_n": ng,
        "h2h_home_avg_gf": float(np.mean(home_gf_list)),
        "h2h_away_avg_gf": float(np.mean(away_gf_list)),
    }


def get_features_for_match(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    date,
    dc_model=None,
) -> dict:
    # Multiple form windows
    hf5  = calculate_form(matches, home_team, date, n=5)
    af5  = calculate_form(matches, away_team, date, n=5)
    hf3  = calculate_form(matches, home_team, date, n=3)
    af3  = calculate_form(matches, away_team, date, n=3)
    hf10 = calculate_form(matches, home_team, date, n=10)
    af10 = calculate_form(matches, away_team, date, n=10)

    # Venue-specific form
    hfh = calculate_form(matches, home_team, date, n=5, venue="home")
    afa = calculate_form(matches, away_team, date, n=5, venue="away")

    # H2H
    h2h = get_h2h(matches, home_team, away_team, date)

    # Rest / fatigue
    home_rest = _days_since_last_match(matches, home_team, date)
    away_rest = _days_since_last_match(matches, away_team, date)

    feat = {
        # Last 5 overall form
        "home_form_pts": hf5["pts"],
        "away_form_pts": af5["pts"],
        "form_pts_diff": hf5["pts"] - af5["pts"],
        "home_form_gf": hf5["gf"],
        "away_form_gf": af5["gf"],
        "home_form_ga": hf5["ga"],
        "away_form_ga": af5["ga"],
        "home_form_gd": hf5["gd"],
        "away_form_gd": af5["gd"],
        "home_cs_rate": hf5["cs_rate"],
        "away_cs_rate": af5["cs_rate"],
        "home_scoring_rate": hf5["scoring_rate"],
        "away_scoring_rate": af5["scoring_rate"],
        # Last 3 form (short-term momentum)
        "home_form3_pts": hf3["pts"],
        "away_form3_pts": af3["pts"],
        "home_form3_gf": hf3["gf"],
        "away_form3_gf": af3["gf"],
        "home_form3_gd": hf3["gd"],
        "away_form3_gd": af3["gd"],
        # Last 10 form (long-term consistency)
        "home_form10_pts": hf10["pts"],
        "away_form10_pts": af10["pts"],
        "home_form10_gd": hf10["gd"],
        "away_form10_gd": af10["gd"],
        # Venue-specific form
        "home_venue_pts": hfh["pts"],
        "away_venue_pts": afa["pts"],
        "home_venue_gf": hfh["gf"],
        "away_venue_gf": afa["gf"],
        "home_venue_ga": hfh["ga"],
        "away_venue_ga": afa["ga"],
        "home_venue_cs": hfh["cs_rate"],
        "away_venue_cs": afa["cs_rate"],
        # H2H
        "h2h_hw": h2h["h2h_hw"],
        "h2h_d": h2h["h2h_d"],
        "h2h_aw": h2h["h2h_aw"],
        "h2h_n": h2h["h2h_n"],
        "h2h_home_avg_gf": h2h["h2h_home_avg_gf"],
        "h2h_away_avg_gf": h2h["h2h_away_avg_gf"],
        # Rest
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_diff": home_rest - away_rest,
    }

    # Dixon-Coles ratings
    if dc_model is not None and dc_model.fitted:
        avg_atk = np.mean(list(dc_model.attack.values()))
        avg_def = np.mean(list(dc_model.defense.values()))
        probs = dc_model.predict_outcome_probs(home_team, away_team)
        xg_h, xg_a = dc_model.get_expected_goals(home_team, away_team)
        feat.update({
            "dc_hw": probs["home_win"],
            "dc_d": probs["draw"],
            "dc_aw": probs["away_win"],
            "dc_home_atk": dc_model.attack.get(home_team, avg_atk),
            "dc_home_def": dc_model.defense.get(home_team, avg_def),
            "dc_away_atk": dc_model.attack.get(away_team, avg_atk),
            "dc_away_def": dc_model.defense.get(away_team, avg_def),
            "dc_xg_home": xg_h,
            "dc_xg_away": xg_a,
            "dc_xg_diff": xg_h - xg_a,
        })
    else:
        feat.update({
            "dc_hw": 0.45, "dc_d": 0.25, "dc_aw": 0.30,
            "dc_home_atk": 0.0, "dc_home_def": 0.0,
            "dc_away_atk": 0.0, "dc_away_def": 0.0,
            "dc_xg_home": 1.4, "dc_xg_away": 1.1, "dc_xg_diff": 0.3,
        })

    return feat


def build_feature_matrix(matches: pd.DataFrame, dc_model=None) -> pd.DataFrame:
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
    # Last 5 overall
    "home_form_pts", "away_form_pts", "form_pts_diff",
    "home_form_gf", "away_form_gf",
    "home_form_ga", "away_form_ga",
    "home_form_gd", "away_form_gd",
    "home_cs_rate", "away_cs_rate",
    "home_scoring_rate", "away_scoring_rate",
    # Last 3 (momentum)
    "home_form3_pts", "away_form3_pts",
    "home_form3_gf", "away_form3_gf",
    "home_form3_gd", "away_form3_gd",
    # Last 10 (consistency)
    "home_form10_pts", "away_form10_pts",
    "home_form10_gd", "away_form10_gd",
    # Venue-specific
    "home_venue_pts", "away_venue_pts",
    "home_venue_gf", "away_venue_gf",
    "home_venue_ga", "away_venue_ga",
    "home_venue_cs", "away_venue_cs",
    # H2H
    "h2h_hw", "h2h_d", "h2h_aw", "h2h_n",
    "h2h_home_avg_gf", "h2h_away_avg_gf",
    # Rest
    "home_rest_days", "away_rest_days", "rest_diff",
    # DC model
    "dc_hw", "dc_d", "dc_aw",
    "dc_home_atk", "dc_home_def",
    "dc_away_atk", "dc_away_def",
    "dc_xg_home", "dc_xg_away", "dc_xg_diff",
]
