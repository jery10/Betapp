"""
AI-powered chat using Claude API.
Gives the AI full context about the match (form, H2H, stats, prediction)
so it can reason intelligently about any question or context note.
"""
import os
import anthropic
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.features import calculate_form, get_h2h
from src.chatbot import _last_match, _form_summary, _h2h_summary, _get_recent

load_dotenv()

_client = None

def _get_api_key() -> str:
    """Get API key from Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY", "")

def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=_get_api_key())
    return _client


def _build_match_context(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    prediction: dict = None,
    dc_model=None,
) -> str:
    """Build a rich context string about the match for the AI."""
    now = pd.Timestamp.now()
    lines = []

    # Basic match info
    lines.append(f"MATCH: {home_team} (home) vs {away_team} (away)")
    lines.append("")

    # Model prediction
    if prediction:
        lines.append("STATISTICAL MODEL PREDICTION:")
        lines.append(f"  Home win: {prediction['home_win_prob']:.1%}")
        lines.append(f"  Draw: {prediction['draw_prob']:.1%}")
        lines.append(f"  Away win: {prediction['away_win_prob']:.1%}")
        lines.append(f"  Predicted score: {prediction['predicted_home_goals']}–{prediction['predicted_away_goals']}")
        lines.append(f"  xG: {prediction['xg_home']} – {prediction['xg_away']}")
        m = prediction.get("markets", {})
        if m:
            lines.append(f"  BTTS: {m.get('btts_yes', 0):.1%} Yes / {m.get('btts_no', 0):.1%} No")
            lines.append(f"  Over 2.5: {m.get('over_25', 0):.1%} | Under 2.5: {m.get('under_25', 0):.1%}")
        lines.append("")

    # Recent form
    if not matches.empty:
        lines.append(f"{home_team.upper()} RECENT FORM (last 5):")
        lines.append(_form_summary(matches, home_team, now))
        lines.append("")
        lines.append(f"{away_team.upper()} RECENT FORM (last 5):")
        lines.append(_form_summary(matches, away_team, now))
        lines.append("")

        # H2H
        lines.append("HEAD-TO-HEAD (last 6 meetings):")
        lines.append(_h2h_summary(matches, home_team, away_team, now))
        lines.append("")

    # DC ratings
    if dc_model and dc_model.fitted:
        avg_atk = np.mean(list(dc_model.attack.values()))
        avg_def = np.mean(list(dc_model.defense.values()))
        lines.append("DIXON-COLES MODEL RATINGS:")
        lines.append(f"  {home_team}: attack {dc_model.attack.get(home_team, avg_atk):.3f}, "
                     f"defense {dc_model.defense.get(home_team, avg_def):.3f}")
        lines.append(f"  {away_team}: attack {dc_model.attack.get(away_team, avg_atk):.3f}, "
                     f"defense {dc_model.defense.get(away_team, avg_def):.3f}")
        lines.append(f"  Home advantage factor: {dc_model.home_adv:.3f}")

    return "\n".join(lines)


def chat_with_ai(
    user_message: str,
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    prediction: dict = None,
    dc_model=None,
    history: list = None,
) -> tuple[str, list]:
    """
    Send a message to Claude with full match context.
    Returns (response_text, updated_history).
    history is a list of {"role": "user"/"assistant", "content": str}
    """
    client = _get_client()

    match_context = _build_match_context(home_team, away_team, matches, prediction, dc_model)

    system_prompt = f"""You are an expert football analyst and betting advisor built into a personal prediction app.
You have access to real statistical data about the upcoming match.

{match_context}

Your role:
1. Answer questions about the match using the data above
2. If the user mentions context like injuries, suspensions, new coach, or team news — explain how that would affect the prediction and what adjustments should be made
3. Give honest, calibrated opinions. Don't overclaim certainty about football predictions
4. Be concise but insightful. Use bullet points where helpful
5. If the user asks you to update/adjust the prediction based on new info, explain the direction of the change

Important: You are a personal tool. Be direct, practical, and focused on helping the user make a good betting decision.
Always remind the user that football is unpredictable and to bet responsibly."""

    # Build message history
    if history is None:
        history = []

    history.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Fast + cheap for chat
        max_tokens=600,
        system=system_prompt,
        messages=history,
    )

    reply = response.content[0].text
    history.append({"role": "assistant", "content": reply})

    return reply, history
