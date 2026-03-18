"""
Simple Q&A engine for the context chat box.
Detects questions and answers them from match data.
Also detects context notes (injuries etc.) and passes to adjustments.
"""
import re
import pandas as pd
from src.features import calculate_form, get_h2h


def is_question(text: str) -> bool:
    """Detect if the input is a question rather than a context note."""
    text = text.lower().strip()
    question_starters = ["what", "when", "how", "who", "where", "did", "do", "does",
                          "is", "are", "was", "were", "show", "tell", "last match",
                          "last game", "last result", "recent", "form", "h2h", "history"]
    return text.endswith("?") or any(text.startswith(w) for w in question_starters)


def answer_question(text: str, home_team: str, away_team: str, matches: pd.DataFrame) -> str:
    """Answer a question about the two teams using match data."""
    t = text.lower().strip().rstrip("?")
    now = pd.Timestamp.now()

    # ── Last match / recent result ──────────────────────────────────────────
    if any(w in t for w in ["last match", "last game", "last result", "recent result", "last time"]):
        # Figure out which team they're asking about
        team = _detect_team_in_question(t, home_team, away_team)

        if "met" in t or "each other" in t or "between" in t or "h2h" in t or team == "both":
            return _h2h_summary(matches, home_team, away_team, now)

        if team == "home" or team == "unknown":
            result1 = _last_match(matches, home_team, now)
            result2 = _last_match(matches, away_team, now)
            return f"**{home_team} last match:** {result1}\n\n**{away_team} last match:** {result2}"
        elif team == "away":
            return f"**{away_team} last match:** {_last_match(matches, away_team, now)}"

    # ── Form ────────────────────────────────────────────────────────────────
    if any(w in t for w in ["form", "how are they", "how have they", "recent games",
                             "last 5", "playing well", "in form", "out of form"]):
        team = _detect_team_in_question(t, home_team, away_team)
        if team in ("home", "unknown"):
            hf = _form_summary(matches, home_team, now)
            af = _form_summary(matches, away_team, now)
            return f"**{home_team} form:**\n{hf}\n\n**{away_team} form:**\n{af}"
        elif team == "away":
            return f"**{away_team} form:**\n{_form_summary(matches, away_team, now)}"

    # ── H2H ─────────────────────────────────────────────────────────────────
    if any(w in t for w in ["h2h", "head to head", "head-to-head", "history between",
                             "last time they met", "previous meetings", "past meetings"]):
        return _h2h_summary(matches, home_team, away_team, now)

    # ── Goals / scoring ─────────────────────────────────────────────────────
    if any(w in t for w in ["goals", "scored", "scoring", "goals per game", "average goals"]):
        team = _detect_team_in_question(t, home_team, away_team)
        hf = calculate_form(matches, home_team, now, n=5)
        af = calculate_form(matches, away_team, now, n=5)
        if team == "away":
            return (f"**{away_team}** scores **{af['gf']:.1f}** goals/game and "
                    f"concedes **{af['ga']:.1f}** (last 5 games).")
        return (f"**{home_team}** scores **{hf['gf']:.1f}** goals/game, "
                f"concedes **{hf['ga']:.1f}**.\n\n"
                f"**{away_team}** scores **{af['gf']:.1f}** goals/game, "
                f"concedes **{af['ga']:.1f}**.\n\n"
                f"Combined avg: **{hf['gf'] + af['gf']:.1f}** goals expected per game.")

    # ── Clean sheets ────────────────────────────────────────────────────────
    if any(w in t for w in ["clean sheet", "shut out", "kept out"]):
        hf = calculate_form(matches, home_team, now, n=5)
        af = calculate_form(matches, away_team, now, n=5)
        h_cs = sum(1 for _, r in _get_recent(matches, home_team, now, 5).iterrows() if r["GA"] == 0)
        a_cs = sum(1 for _, r in _get_recent(matches, away_team, now, 5).iterrows() if r["GA"] == 0)
        return (f"**{home_team}**: {h_cs} clean sheet(s) in last 5 games.\n\n"
                f"**{away_team}**: {a_cs} clean sheet(s) in last 5 games.")

    # ── Win streak / unbeaten ───────────────────────────────────────────────
    if any(w in t for w in ["win streak", "unbeaten", "losing streak", "without a win"]):
        h_streak = _streak(matches, home_team, now)
        a_streak = _streak(matches, away_team, now)
        return f"**{home_team}**: {h_streak}\n\n**{away_team}**: {a_streak}"

    # ── Fallback ─────────────────────────────────────────────────────────────
    hf = _form_summary(matches, home_team, now)
    af = _form_summary(matches, away_team, now)
    return (f"Here's what I know about this match:\n\n"
            f"**{home_team} form:**\n{hf}\n\n"
            f"**{away_team} form:**\n{af}\n\n"
            f"**H2H:**\n{_h2h_summary(matches, home_team, away_team, now)}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_team_in_question(text: str, home_team: str, away_team: str) -> str:
    t = text.lower()
    home_words = set(home_team.lower().split()) - {"fc", "the", "cf", "ac"}
    away_words = set(away_team.lower().split()) - {"fc", "the", "cf", "ac"}
    words = set(re.findall(r'\w+', t))
    h = len(home_words & words)
    a = len(away_words & words)
    if h > a:
        return "home"
    elif a > h:
        return "away"
    elif "they" in t or "their" in t:
        return "both"
    return "unknown"


def _get_recent(matches: pd.DataFrame, team: str, before, n: int = 5) -> pd.DataFrame:
    home = matches[(matches["HomeTeam"] == team) & (matches["Date"] < before)].copy()
    away = matches[(matches["AwayTeam"] == team) & (matches["Date"] < before)].copy()
    home["GF"] = home["FTHG"]; home["GA"] = home["FTAG"]
    home["Result"] = home["FTR"].map({"H": "W", "D": "D", "A": "L"})
    away["GF"] = away["FTAG"]; away["GA"] = away["FTHG"]
    away["Result"] = away["FTR"].map({"A": "W", "D": "D", "H": "L"})
    combined = pd.concat([home, away]).sort_values("Date").tail(n)
    return combined


def _last_match(matches: pd.DataFrame, team: str, before) -> str:
    recent = _get_recent(matches, team, before, n=1)
    if recent.empty:
        return "No recent data found."
    row = recent.iloc[-1]
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    hg = int(row["FTHG"])
    ag = int(row["FTAG"])
    date_str = pd.Timestamp(row["Date"]).strftime("%d %b %Y")
    result_str = "W" if row["Result"] == "W" else ("D" if row["Result"] == "D" else "L")
    return f"{home} {hg}–{ag} {away}  ({date_str}) → **{result_str}**"


def _form_summary(matches: pd.DataFrame, team: str, before) -> str:
    recent = _get_recent(matches, team, before, n=5)
    if recent.empty:
        return "No recent data available."

    results = []
    for _, row in recent.iterrows():
        opp = row["AwayTeam"] if row["HomeTeam"] == team else row["HomeTeam"]
        gf, ga = int(row["GF"]), int(row["GA"])
        r = row["Result"]
        emoji = "✅" if r == "W" else ("🟡" if r == "D" else "❌")
        results.append(f"{emoji} {gf}–{ga} vs {opp}")

    f = calculate_form(matches, team, before, n=5)
    summary = " · ".join(results)
    return f"{summary}\n_Pts: {int(f['pts'])} | Avg scored: {f['gf']:.1f} | Avg conceded: {f['ga']:.1f}_"


def _h2h_summary(matches: pd.DataFrame, home_team: str, away_team: str, before) -> str:
    mask = (
        ((matches["HomeTeam"] == home_team) & (matches["AwayTeam"] == away_team))
        | ((matches["HomeTeam"] == away_team) & (matches["AwayTeam"] == home_team))
    ) & (matches["Date"] < before)
    h2h = matches[mask].tail(6)

    if h2h.empty:
        return "No previous meetings found in the dataset."

    lines = []
    hw = d = aw = 0
    for _, row in h2h.iterrows():
        date_str = pd.Timestamp(row["Date"]).strftime("%d %b %Y")
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        ftr = row["FTR"]
        if row["HomeTeam"] == home_team:
            winner = home_team if ftr == "H" else (away_team if ftr == "A" else "Draw")
            if ftr == "H": hw += 1
            elif ftr == "A": aw += 1
            else: d += 1
            lines.append(f"• {row['HomeTeam']} {hg}–{ag} {row['AwayTeam']}  ({date_str}) → **{winner}**")
        else:
            winner = home_team if ftr == "A" else (away_team if ftr == "H" else "Draw")
            if ftr == "A": hw += 1
            elif ftr == "H": aw += 1
            else: d += 1
            lines.append(f"• {row['HomeTeam']} {hg}–{ag} {row['AwayTeam']}  ({date_str}) → **{winner}**")

    n = len(h2h)
    summary = f"Last {n} meetings — **{home_team} wins: {hw}** · Draws: {d} · **{away_team} wins: {aw}**\n\n"
    return summary + "\n".join(reversed(lines))


def _streak(matches: pd.DataFrame, team: str, before) -> str:
    recent = _get_recent(matches, team, before, n=10)
    if recent.empty:
        return "No data."
    results = list(recent["Result"])[::-1]
    streak_type = results[0]
    count = 0
    for r in results:
        if r == streak_type:
            count += 1
        else:
            break
    label = {"W": "winning", "D": "unbeaten run (draws)", "L": "without a win"}.get(streak_type, "")
    return f"Currently on a **{count}-game {label} streak**."
