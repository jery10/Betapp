import pandas as pd
import numpy as np
from src.features import calculate_form, get_h2h


def explain_prediction(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    dc_model,
    prediction: dict,
    adjustment: dict = None,
) -> dict:
    """
    Generate a human-readable explanation for a prediction.
    Returns a dict with factors, narrative, and comparison data.
    """
    now = pd.Timestamp.now()

    # ── Form ────────────────────────────────────────────────────────────────
    hf  = calculate_form(matches, home_team, now, n=5)
    af  = calculate_form(matches, away_team, now, n=5)
    hfh = calculate_form(matches, home_team, now, n=5, venue="home")
    afa = calculate_form(matches, away_team, now, n=5, venue="away")

    # ── H2H ─────────────────────────────────────────────────────────────────
    h2h = get_h2h(matches, home_team, away_team, now, n=6)

    # ── DC Ratings ──────────────────────────────────────────────────────────
    avg_atk = np.mean(list(dc_model.attack.values())) if dc_model.attack else 0
    avg_def = np.mean(list(dc_model.defense.values())) if dc_model.defense else 0
    home_atk = dc_model.attack.get(home_team, avg_atk)
    away_atk = dc_model.attack.get(away_team, avg_atk)
    home_def = dc_model.defense.get(home_team, avg_def)
    away_def = dc_model.defense.get(away_team, avg_def)

    hw = prediction["home_win_prob"]
    d  = prediction["draw_prob"]
    aw = prediction["away_win_prob"]
    xg_h = prediction["xg_home"]
    xg_a = prediction["xg_away"]
    rec  = prediction["recommendation"]
    conf = prediction["confidence"]

    # ── Build factor list ───────────────────────────────────────────────────
    factors = []

    # Form
    if hf["pts"] > af["pts"] + 2:
        factors.append({
            "icon": "📈",
            "label": "Better recent form",
            "detail": f"{home_team} {int(hf['pts'])} pts vs {away_team} {int(af['pts'])} pts (last 5)",
            "favor": "home",
        })
    elif af["pts"] > hf["pts"] + 2:
        factors.append({
            "icon": "📈",
            "label": "Better recent form",
            "detail": f"{away_team} {int(af['pts'])} pts vs {home_team} {int(hf['pts'])} pts (last 5)",
            "favor": "away",
        })
    else:
        factors.append({
            "icon": "📊",
            "label": "Similar recent form",
            "detail": f"{home_team} {int(hf['pts'])} pts · {away_team} {int(af['pts'])} pts (last 5)",
            "favor": "neutral",
        })

    # Venue form
    if hfh["gf"] > 0 or afa["gf"] > 0:
        if hfh["pts"] > afa["pts"] + 1:
            factors.append({
                "icon": "🏟️",
                "label": "Home venue advantage",
                "detail": f"{home_team} scores {hfh['gf']:.1f} avg at home · {away_team} scores {afa['gf']:.1f} avg away",
                "favor": "home",
            })
        elif afa["pts"] > hfh["pts"] + 1:
            factors.append({
                "icon": "✈️",
                "label": "Away team travels well",
                "detail": f"{away_team} {int(afa['pts'])} pts away · {home_team} {int(hfh['pts'])} pts at home (last 5)",
                "favor": "away",
            })

    # Attack strength
    if home_atk > away_atk + 0.1:
        factors.append({
            "icon": "⚔️",
            "label": "Stronger attacking model",
            "detail": f"{home_team} attack {home_atk:.2f} vs {away_team} attack {away_atk:.2f} (Dixon-Coles)",
            "favor": "home",
        })
    elif away_atk > home_atk + 0.1:
        factors.append({
            "icon": "⚔️",
            "label": "Stronger attacking model",
            "detail": f"{away_team} attack {away_atk:.2f} vs {home_team} attack {home_atk:.2f} (Dixon-Coles)",
            "favor": "away",
        })

    # Defense strength (lower = better)
    if home_def < away_def - 0.1:
        factors.append({
            "icon": "🛡️",
            "label": "Stronger defensive model",
            "detail": f"{home_team} defense {home_def:.2f} vs {away_team} defense {away_def:.2f} (lower is better)",
            "favor": "home",
        })
    elif away_def < home_def - 0.1:
        factors.append({
            "icon": "🛡️",
            "label": "Stronger defensive model",
            "detail": f"{away_team} defense {away_def:.2f} vs {home_team} defense {home_def:.2f} (lower is better)",
            "favor": "away",
        })

    # H2H
    if h2h["h2h_n"] >= 3:
        if h2h["h2h_hw"] > 0.55:
            factors.append({
                "icon": "🔁",
                "label": "H2H record favors home",
                "detail": f"{home_team} wins {h2h['h2h_hw']:.0%} of last {h2h['h2h_n']} meetings",
                "favor": "home",
            })
        elif h2h["h2h_aw"] > 0.55:
            factors.append({
                "icon": "🔁",
                "label": "H2H record favors away",
                "detail": f"{away_team} wins {h2h['h2h_aw']:.0%} of last {h2h['h2h_n']} meetings",
                "favor": "away",
            })
        elif h2h["h2h_d"] > 0.45:
            factors.append({
                "icon": "🔁",
                "label": "H2H tends to draw",
                "detail": f"{h2h['h2h_d']:.0%} of last {h2h['h2h_n']} meetings ended level",
                "favor": "draw",
            })
        else:
            factors.append({
                "icon": "🔁",
                "label": "H2H record",
                "detail": f"Home wins {h2h['h2h_hw']:.0%} · Draws {h2h['h2h_d']:.0%} · Away wins {h2h['h2h_aw']:.0%} (last {h2h['h2h_n']})",
                "favor": "neutral",
            })

    # xG / expected goals
    factors.append({
        "icon": "🎯",
        "label": "Expected goals (xG)",
        "detail": f"{home_team} xG {xg_h} · {away_team} xG {xg_a} — model expects {xg_h + xg_a:.1f} total goals",
        "favor": "home" if xg_h > xg_a + 0.3 else ("away" if xg_a > xg_h + 0.3 else "neutral"),
    })

    # Home advantage from DC model
    factors.append({
        "icon": "🏠",
        "label": "Home advantage",
        "detail": f"Home advantage factor: {dc_model.home_adv:.2f} (built into model from all historical data)",
        "favor": "home",
    })

    # Adjustments applied
    if adjustment and adjustment.get("log"):
        for item in adjustment["log"]:
            factors.append({
                "icon": "🔧",
                "label": "Manual adjustment",
                "detail": item,
                "favor": "adjusted",
            })

    # ── Narrative summary ───────────────────────────────────────────────────
    narrative = _build_narrative(
        home_team, away_team, rec, conf, hw, d, aw, factors,
        xg_h, xg_a, hf, af, h2h, prediction
    )

    return {
        "factors": factors,
        "narrative": narrative,
        "form_comparison": {
            "home_pts": hf["pts"], "away_pts": af["pts"],
            "home_gf": hf["gf"], "away_gf": af["gf"],
            "home_ga": hf["ga"], "away_ga": af["ga"],
            "home_venue_pts": hfh["pts"], "away_venue_pts": afa["pts"],
        },
        "h2h": h2h,
        "ratings": {
            "home_atk": round(home_atk, 3), "away_atk": round(away_atk, 3),
            "home_def": round(home_def, 3), "away_def": round(away_def, 3),
        },
    }


def _build_narrative(home, away, rec, conf, hw, d, aw, factors, xg_h, xg_a, hf, af, h2h, prediction) -> str:
    """Build a short plain-English explanation."""
    lines = []

    # Opening
    outcome_map = {"Home Win": home, "Draw": "a draw", "Away Win": away}
    subject = outcome_map.get(rec, rec)
    lines.append(f"**We favour {subject}** ({conf.lower()} confidence — {max(hw, d, aw):.0%} probability).")

    # Key reasons
    home_favors = [f for f in factors if f["favor"] == "home"]
    away_favors = [f for f in factors if f["favor"] == "away"]
    draw_favors = [f for f in factors if f["favor"] == "draw"]

    if rec == "Home Win" and home_favors:
        reasons = [f["detail"] for f in home_favors[:2]]
        lines.append("**Why:** " + " · ".join(reasons) + ".")
    elif rec == "Away Win" and away_favors:
        reasons = [f["detail"] for f in away_favors[:2]]
        lines.append("**Why:** " + " · ".join(reasons) + ".")
    elif rec == "Draw":
        if draw_favors:
            lines.append("**Why:** " + draw_favors[0]["detail"] + ".")
        lines.append(f"The teams appear closely matched — {home} {int(hf['pts'])} pts vs {away} {int(af['pts'])} pts in recent form.")

    # xG note
    lines.append(f"**Expected goals:** {home} {xg_h} — {away} {xg_a}. Most likely score: **{prediction['predicted_home_goals']}–{prediction['predicted_away_goals']}** ({prediction['score_probability']:.1%} chance).")

    # H2H note
    if h2h["h2h_n"] >= 3:
        lines.append(f"**History:** In their last {h2h['h2h_n']} meetings — {home} won {h2h['h2h_hw']:.0%}, drew {h2h['h2h_d']:.0%}, {away} won {h2h['h2h_aw']:.0%}.")

    # Confidence caveat
    if conf == "Low":
        lines.append("⚠️ *Low confidence — this is a genuinely unpredictable match. Consider smaller stake.*")

    return "\n\n".join(lines)
