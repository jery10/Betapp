import re


# Predefined player role impact multipliers
ROLE_IMPACTS = {
    "striker":    {"atk": 0.78, "def": 1.00, "label": "Key striker out"},
    "forward":    {"atk": 0.78, "def": 1.00, "label": "Key forward out"},
    "winger":     {"atk": 0.82, "def": 1.00, "label": "Key winger out"},
    "playmaker":  {"atk": 0.85, "def": 0.95, "label": "Key playmaker out"},
    "midfielder": {"atk": 0.88, "def": 0.95, "label": "Key midfielder out"},
    "defender":   {"atk": 1.00, "def": 0.82, "label": "Key defender out"},
    "centre back":{"atk": 1.00, "def": 0.80, "label": "Centre back out"},
    "center back":{"atk": 1.00, "def": 0.80, "label": "Centre back out"},
    "goalkeeper": {"atk": 1.00, "def": 0.85, "label": "Goalkeeper out"},
    "keeper":     {"atk": 1.00, "def": 0.85, "label": "Goalkeeper out"},
    "default":    {"atk": 0.82, "def": 0.97, "label": "Key player out"},
}

ABSENCE_KEYWORDS = [
    "no ", "without", " out", "absent", "missing", "miss",
    "injured", "injury", "suspended", "suspension", "unavailable",
    "ruled out", "doubt", "doesn't play", "does not play", "not playing",
    "won't play", "will not play", "not available", "not in squad",
]

NEW_COACH_KEYWORDS = [
    "new coach", "new manager", "sacked", "interim", "first game",
    "first match", "managerial change", "caretaker",
]

MOTIVATION_KEYWORDS = [
    "must win", "must-win", "crucial", "do or die", "elimination",
    "final", "desperately need", "need a win", "backs against the wall",
    "revenge", "derby", "local derby",
]

FATIGUE_KEYWORDS = [
    "tired", "fatigue", "heavy schedule", "3 games", "midweek",
    "rotation expected", "rotated", "legs are tired", "busy schedule",
]

GOOD_FORM_KEYWORDS = [
    "on fire", "great form", "winning streak", "clean sheet run",
    "unstoppable", "in form", "flying", "momentum",
]


def _detect_team(sentence: str, home_team: str, away_team: str) -> str:
    """Detect which team a sentence refers to."""
    s = sentence.lower()

    # Check for team name words
    home_words = set(home_team.lower().split())
    away_words = set(away_team.lower().split())
    sentence_words = set(re.findall(r'\w+', s))

    # Remove common short words
    stop = {"fc", "the", "a", "of", "de", "and", "ac", "sc"}
    home_score = len((home_words - stop) & sentence_words)
    away_score = len((away_words - stop) & sentence_words)

    if home_score > away_score:
        return "home"
    elif away_score > home_score:
        return "away"

    # Positional hints
    if any(w in s for w in ["home team", "hosts", "home side"]):
        return "home"
    if any(w in s for w in ["away team", "visitors", "away side"]):
        return "away"

    return "unknown"


def _detect_role(sentence: str) -> dict:
    """Detect player role from sentence."""
    s = sentence.lower()
    for role, impact in ROLE_IMPACTS.items():
        if role != "default" and role in s:
            return impact
    return ROLE_IMPACTS["default"]


def parse_context(notes: str, home_team: str, away_team: str) -> dict:
    """
    Parse free-text context notes and return adjustment multipliers.

    Returns:
        {
            'home_atk': float,   # multiplier (1.0 = no change)
            'home_def': float,
            'away_atk': float,
            'away_def': float,
            'log': list[str],    # human-readable list of applied adjustments
        }
    """
    result = {
        "home_atk": 1.0, "home_def": 1.0,
        "away_atk": 1.0, "away_def": 1.0,
        "log": [],
    }

    if not notes or not notes.strip():
        return result

    text = notes.lower()

    # Split into clauses for analysis
    clauses = [c.strip() for c in re.split(r'[.,;\n]|\band\b', text) if c.strip()]

    for clause in clauses:
        team = _detect_team(clause, home_team, away_team)

        # ── Player absence ──────────────────────────────────
        if any(kw in clause for kw in ABSENCE_KEYWORDS):
            impact = _detect_role(clause)
            if team == "home":
                result["home_atk"] *= impact["atk"]
                result["home_def"] *= impact["def"]
                result["log"].append(f"Home — {impact['label']}")
            elif team == "away":
                result["away_atk"] *= impact["atk"]
                result["away_def"] *= impact["def"]
                result["log"].append(f"Away — {impact['label']}")
            else:
                # Can't determine team — apply lightly to both
                result["home_atk"] *= (impact["atk"] + 1) / 2
                result["away_atk"] *= (impact["atk"] + 1) / 2
                result["log"].append(f"Unknown team — {impact['label']} (mild both)")

        # ── New coach ────────────────────────────────────────
        if any(kw in clause for kw in NEW_COACH_KEYWORDS):
            if team == "home":
                result["home_atk"] *= 0.92
                result["home_def"] *= 0.92
                result["log"].append("Home — New coach: added uncertainty")
            elif team == "away":
                result["away_atk"] *= 0.92
                result["away_def"] *= 0.92
                result["log"].append("Away — New coach: added uncertainty")

        # ── Must win / high motivation ───────────────────────
        if any(kw in clause for kw in MOTIVATION_KEYWORDS):
            if team == "home":
                result["home_atk"] *= 1.10
                result["log"].append("Home — Must-win: attack boost +10%")
            elif team == "away":
                result["away_atk"] *= 1.10
                result["log"].append("Away — Must-win: attack boost +10%")
            else:
                result["home_atk"] *= 1.05
                result["away_atk"] *= 1.05
                result["log"].append("Derby/final: mild boost to both")

        # ── Fatigue ───────────────────────────────────────────
        if any(kw in clause for kw in FATIGUE_KEYWORDS):
            if team == "home":
                result["home_atk"] *= 0.90
                result["home_def"] *= 0.93
                result["log"].append("Home — Fatigue: attack/defense penalty")
            elif team == "away":
                result["away_atk"] *= 0.90
                result["away_def"] *= 0.93
                result["log"].append("Away — Fatigue: attack/defense penalty")

        # ── Good form (positive) ─────────────────────────────
        if any(kw in clause for kw in GOOD_FORM_KEYWORDS):
            if team == "home":
                result["home_atk"] *= 1.08
                result["log"].append("Home — In great form: attack boost +8%")
            elif team == "away":
                result["away_atk"] *= 1.08
                result["log"].append("Away — In great form: attack boost +8%")

    return result


def format_adjustments(adj: dict) -> str:
    """Return a readable summary of adjustments applied."""
    if not adj["log"]:
        return "No adjustments applied."
    lines = ["**Adjustments applied:**"]
    for item in adj["log"]:
        lines.append(f"- {item}")
    ha = adj["home_atk"]
    hd = adj["home_def"]
    aa = adj["away_atk"]
    ad = adj["away_def"]
    lines.append(f"\nNet effect — Home: atk×{ha:.2f} def×{hd:.2f} | Away: atk×{aa:.2f} def×{ad:.2f}")
    return "\n".join(lines)
