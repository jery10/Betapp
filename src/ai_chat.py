"""
AI-powered chat using Claude API with web search tool.
Gives the AI full context about the match (form, H2H, stats, prediction)
and the ability to search the web for current news, injuries, suspensions.
"""
import os
import json
import anthropic
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.features import calculate_form, get_h2h
from src.chatbot import _last_match, _form_summary, _h2h_summary, _get_recent

load_dotenv()

_client = None

def _get_api_key() -> str:
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


def _web_search(query: str, max_results: int = 4) -> str:
    """Search DuckDuckGo and return a summary of results."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', '')}**")
            lines.append(r.get('body', ''))
            lines.append(f"Source: {r.get('href', '')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


# Tool definition for Claude
_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current football news, injury updates, suspensions, "
        "team news, press conference quotes, or any recent information about teams or players. "
        "Use this whenever the user asks about current news or when you need up-to-date info."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'Arsenal injury news March 2025' or 'Yamal Barcelona suspension'"
            }
        },
        "required": ["query"]
    }
}


def _build_match_context(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    prediction: dict = None,
    dc_model=None,
    crowd: dict = None,
) -> str:
    now = pd.Timestamp.now()
    lines = []

    lines.append(f"MATCH: {home_team} (home) vs {away_team} (away)")
    lines.append("")

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

    if not matches.empty:
        lines.append(f"{home_team.upper()} RECENT FORM (last 5):")
        lines.append(_form_summary(matches, home_team, now))
        lines.append("")
        lines.append(f"{away_team.upper()} RECENT FORM (last 5):")
        lines.append(_form_summary(matches, away_team, now))
        lines.append("")

        lines.append("HEAD-TO-HEAD (last 6 meetings):")
        lines.append(_h2h_summary(matches, home_team, away_team, now))
        lines.append("")

    if dc_model and dc_model.fitted:
        avg_atk = np.mean(list(dc_model.attack.values()))
        avg_def = np.mean(list(dc_model.defense.values()))
        lines.append("DIXON-COLES MODEL RATINGS:")
        lines.append(f"  {home_team}: attack {dc_model.attack.get(home_team, avg_atk):.3f}, "
                     f"defense {dc_model.defense.get(home_team, avg_def):.3f}")
        lines.append(f"  {away_team}: attack {dc_model.attack.get(away_team, avg_atk):.3f}, "
                     f"defense {dc_model.defense.get(away_team, avg_def):.3f}")
        lines.append(f"  Home advantage factor: {dc_model.home_adv:.3f}")
        lines.append("")

    if crowd and crowd.get("total_predictions", 0) > 0:
        con = crowd["consensus"]
        lines.append("TIPKING CROWD PREDICTIONS:")
        lines.append(f"  Total predictions from real users: {crowd['total_predictions']}")
        lines.append(f"  Crowd says — Home win: {con.get('home_win_pct', 0)}% | "
                     f"Draw: {con.get('draw_pct', 0)}% | Away win: {con.get('away_win_pct', 0)}%")
        lines.append(f"  Crowd top pick: {con.get('top_pick', 'N/A')}")
        if crowd.get("avg_predicted_home") is not None:
            lines.append(f"  Avg predicted score: {crowd['avg_predicted_home']} – {crowd['avg_predicted_away']}")
        if crowd.get("over_25_pct") is not None:
            lines.append(f"  % predicting over 2.5 goals: {crowd['over_25_pct']}%")
        reasons = crowd.get("reasoning", [])
        if reasons:
            lines.append("  User reasoning from TipKing:")
            for r in reasons[:5]:
                lines.append(f"    - \"{r}\"")

    return "\n".join(lines)


def chat_with_ai(
    user_message: str,
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    prediction: dict = None,
    dc_model=None,
    history: list = None,
    crowd: dict = None,
) -> tuple[str, list]:
    """
    Send a message to Claude with full match context and web search tool.
    Returns (response_text, updated_history).
    """
    client = _get_client()
    match_context = _build_match_context(home_team, away_team, matches, prediction, dc_model, crowd)

    system_prompt = f"""You are an expert football analyst and betting advisor built into a personal prediction app.
You have access to real statistical data about the upcoming match AND a web search tool to look up current news.

{match_context}

Your role:
1. Answer questions about the match using the data above
2. Use web_search to find current injury news, suspensions, team news, or any recent developments — especially when the user mentions a player or asks about current form
3. If the user mentions context like injuries or suspensions, search for confirmation and explain how it affects the prediction
4. Give honest, calibrated opinions. Don't overclaim certainty about football predictions
5. Be concise but insightful. Use bullet points where helpful
6. If asked to update/adjust the prediction based on new info, explain the direction of the change

Important: You are a personal tool. Be direct, practical, and focused on helping the user make a good betting decision.
Always remind the user that football is unpredictable and to bet responsibly.
Today's date: {pd.Timestamp.now().strftime('%d %B %Y')}"""

    if history is None:
        history = []

    history.append({"role": "user", "content": user_message})

    # Agentic loop — Claude may call web_search one or more times
    messages = list(history)
    final_reply = ""

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            tools=[_SEARCH_TOOL],
            messages=messages,
        )

        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Add Claude's response (with tool call) to messages
            messages.append({"role": "assistant", "content": response.content})

            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    query = block.input.get("query", "")
                    search_result = _web_search(query)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": search_result,
                    })

            # Add tool results and continue
            messages.append({"role": "user", "content": tool_results})

        else:
            # Final response
            for block in response.content:
                if hasattr(block, "text"):
                    final_reply += block.text
            break

    history.append({"role": "assistant", "content": final_reply})
    return final_reply, history
