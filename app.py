import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.config import COMPETITIONS, MODELS_DIR
from src.data_fetcher import get_all_results, fetch_upcoming_fixtures, get_standings, get_todays_results
from src.predictor import MatchPredictor
from src.adjustments import parse_context, format_adjustments
from src.explainer import explain_prediction
from src.chatbot import is_question, answer_question
from src.ai_chat import chat_with_ai

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .big-title { font-size: 1.9rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.2rem; }
    .comp-badge { font-size: 1rem; font-weight: 600; color: #555; margin-bottom: 1rem; }
    .prob-hw { color: #00966e; font-weight: 700; font-size: 1.05rem; }
    .prob-d  { color: #e8720c; font-weight: 700; font-size: 1.05rem; }
    .prob-aw { color: #c0392b; font-weight: 700; font-size: 1.05rem; }
    .score-big { font-size: 1.5rem; font-weight: 800; text-align: center; }
    .market-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px;
    }
    .chip-green { background: #d4edda; color: #155724; }
    .chip-orange { background: #fff3cd; color: #856404; }
    .chip-red { background: #f8d7da; color: #721c24; }
    .chip-blue { background: #d1ecf1; color: #0c5460; }
    .adj-box {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .day-mood {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 16px;
    }
    .conf-high   { background:#00966e; color:white; padding:2px 9px; border-radius:5px; font-size:0.82rem; }
    .conf-medium { background:#e8720c; color:white; padding:2px 9px; border-radius:5px; font-size:0.82rem; }
    .conf-low    { background:#888; color:white; padding:2px 9px; border-radius:5px; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────

def model_path(comp: str) -> Path:
    return MODELS_DIR / f"predictor_{comp}.pkl"


@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(comp):
    return get_all_results(comp)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_fixtures(comp):
    return fetch_upcoming_fixtures(comp)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_standings(comp):
    return get_standings(comp)


@st.cache_data(ttl=600, show_spinner=False)
def _load_today(comp):
    return get_todays_results(comp)


@st.cache_resource(show_spinner=False)
def _get_model(comp, _retrain=False):
    mp = model_path(comp)
    if mp.exists() and not _retrain:
        try:
            return MatchPredictor.load(mp)
        except Exception:
            pass
    matches = get_all_results(comp)
    if matches.empty:
        return None
    m = MatchPredictor()
    m.train(matches)
    m.save(mp)
    return m


def render_prob_bar(hw, d, aw, home, away):
    fig = go.Figure()
    for val, color, label in [
        (hw, "#00966e", f"{home} Win"),
        (d,  "#e8720c", "Draw"),
        (aw, "#c0392b", f"{away} Win"),
    ]:
        fig.add_trace(go.Bar(
            y=[""], x=[val], orientation="h",
            marker_color=color, name=label,
            text=f"{val:.0%}", textposition="inside",
        ))
    fig.update_layout(
        barmode="stack", height=55,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def market_chips(markets: dict) -> str:
    """Render market chips as HTML."""
    def chip(label, prob, threshold=0.55):
        cls = "chip-green" if prob >= threshold else ("chip-orange" if prob >= 0.45 else "chip-red")
        return f'<span class="market-chip {cls}">{label}: {prob:.0%}</span>'

    chips = [
        chip("BTTS ✓", markets["btts_yes"]),
        chip("BTTS ✗", markets["btts_no"]),
        chip("O 1.5", markets["over_15"]),
        chip("O 2.5", markets["over_25"]),
        chip("O 3.5", markets["over_35"]),
        chip("U 2.5", markets["under_25"]),
    ]
    return " ".join(chips)


def render_explanation(exp: dict, home: str, away: str):
    """Render prediction explanation in the UI."""
    st.markdown(exp["narrative"])

    st.markdown("**Key factors:**")
    favor_colors = {"home": "🟢", "away": "🔴", "draw": "🟡", "neutral": "⚪", "adjusted": "🔧"}

    for f in exp["factors"]:
        dot = favor_colors.get(f["favor"], "⚪")
        st.markdown(f"{dot} {f['icon']} **{f['label']}** — {f['detail']}")

    # Form table
    fc = exp["form_comparison"]
    st.markdown("**Form comparison (last 5 games):**")
    form_df = pd.DataFrame({
        "": ["Points", "Goals scored (avg)", "Goals conceded (avg)", "Venue pts (last 5)"],
        home: [
            str(int(fc["home_pts"])),
            f"{fc['home_gf']:.1f}",
            f"{fc['home_ga']:.1f}",
            str(int(fc["home_venue_pts"])),
        ],
        away: [
            str(int(fc["away_pts"])),
            f"{fc['away_gf']:.1f}",
            f"{fc['away_ga']:.1f}",
            str(int(fc["away_venue_pts"])),
        ],
    })
    st.dataframe(form_df, use_container_width=True, hide_index=True)


def day_momentum_bar(comp: str):
    """Show same-day results momentum."""
    today_df = _load_today(comp)
    if today_df.empty:
        return
    finished = today_df[today_df["finished"] == True].copy()
    if finished.empty:
        return

    finished["result"] = finished.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else ("A" if r["away_goals"] > r["home_goals"] else "D"), axis=1
    )
    n = len(finished)
    hw = (finished["result"] == "H").sum()
    d  = (finished["result"] == "D").sum()
    aw = (finished["result"] == "A").sum()
    avg_goals = (finished["home_goals"] + finished["away_goals"]).mean()
    btts = ((finished["home_goals"] > 0) & (finished["away_goals"] > 0)).mean()

    mood = "🏠 Home Day" if hw / n > 0.5 else ("⚖️ Draw Day" if d / n > 0.35 else "✈️ Away Day")
    goals_tag = "🔥 High Scoring" if avg_goals >= 2.8 else ("😴 Low Scoring" if avg_goals < 1.8 else "⚽ Normal Scoring")

    st.markdown(f"""
    <div class="day-mood">
        <b>Same-Day Momentum</b> &nbsp;|&nbsp; {n} matches played today<br>
        {mood} &nbsp;·&nbsp; {goals_tag} &nbsp;·&nbsp;
        Home wins: {hw} · Draws: {d} · Away wins: {aw} · Avg goals: {avg_goals:.1f} · BTTS: {btts:.0%}
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚽ Football Predictor")
st.sidebar.markdown("---")

comp_options = {f"{v['flag']} {v['name']}": k for k, v in COMPETITIONS.items()}
comp_keys = list(comp_options.keys())
comp_label = st.sidebar.selectbox("Competition", comp_keys, index=0)
comp = comp_options[comp_label]

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Page",
    ["Upcoming Fixtures", "Match Predictor", "Team Ratings", "Model Performance"],
)
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Retrain Model"):
    _get_model.clear()
    _load_data.clear()
    st.sidebar.info("Retraining...")
    st.rerun()

if st.sidebar.button("🔃 Refresh Data"):
    _load_data.clear()
    _load_fixtures.clear()
    _load_standings.clear()
    _load_today.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data: football-data.org + football-data.co.uk")

# ─── Load ────────────────────────────────────────────────────────────────────

cfg = COMPETITIONS[comp]
with st.spinner(f"Loading {cfg['name']} model..."):
    model = _get_model(comp)

if model is None:
    st.error(f"Could not load data for {cfg['name']}. Check API key.")
    st.stop()

matches = _load_data(comp)

# ─── PAGE: Upcoming Fixtures ─────────────────────────────────────────────────

if page == "Upcoming Fixtures":
    st.markdown(f'<div class="big-title">{cfg["flag"]} {cfg["name"]} Predictions</div>', unsafe_allow_html=True)

    # Same-day momentum
    day_momentum_bar(comp)

    fixtures = _load_fixtures(comp)
    if fixtures.empty:
        st.warning("No upcoming fixtures found. Try clicking **🔃 Refresh Data** in the sidebar.")
        st.stop()

    now = pd.Timestamp.now()
    fixtures = fixtures.copy()
    fixtures["date_only"] = pd.to_datetime(fixtures["date"]).dt.date

    # ── Date filter dropdown ─────────────────────────────────────────────
    today_date = now.date()
    tomorrow_date = (now + pd.Timedelta(days=1)).date()
    available_dates = sorted(fixtures["date_only"].unique())

    date_labels = {}
    for d in available_dates:
        if d == today_date:
            date_labels[f"📅 Today — {pd.Timestamp(d).strftime('%a %d %b')}"] = d
        elif d == tomorrow_date:
            date_labels[f"📅 Tomorrow — {pd.Timestamp(d).strftime('%a %d %b')}"] = d
        else:
            date_labels[f"📅 {pd.Timestamp(d).strftime('%A %d %B')}"] = d

    date_labels = {"📅 All upcoming": None, **date_labels}

    col_date, col_market = st.columns(2)
    with col_date:
        selected_date_label = st.selectbox("Filter by date", list(date_labels.keys()))
    selected_date = date_labels[selected_date_label]

    if selected_date:
        upcoming = fixtures[fixtures["date_only"] == selected_date]
    else:
        upcoming = fixtures[fixtures["date"] >= now].head(30)
    if upcoming.empty:
        upcoming = fixtures.head(10)

    # Market filter
    with col_market:
        market_filter = st.selectbox(
            "Highlight market",
            ["All Markets", "1X2 Only", "BTTS", "Over/Under Goals"],
            index=0,
        )

    # Group fixtures by date
    date_groups = upcoming.groupby("date_only")

    for date_val, day_fixtures in sorted(date_groups, key=lambda x: x[0]):
        date_label = pd.Timestamp(date_val).strftime("%A %d %B %Y")
        st.markdown(
            f'<div style="background:#37003c;color:white;padding:8px 14px;border-radius:8px;'
            f'font-weight:700;font-size:1rem;margin:16px 0 8px 0">📅 {date_label}</div>',
            unsafe_allow_html=True,
        )

        for _, fix in day_fixtures.iterrows():
            home, away = fix["home_team"], fix["away_team"]
            kickoff = fix["date"].strftime("%H:%M")
            md = fix.get("matchday")
            md_str = f"MD{int(md)}" if pd.notna(md) else ""

            # Context chat box per match
            adj = {}
            with st.expander(f"💬 Chat: {home} vs {away}", expanded=False):
                fix_chat_key = f"fix_chat_{home}_{away}"
                if fix_chat_key not in st.session_state:
                    st.session_state[fix_chat_key] = []

                # Show chat history
                for msg in st.session_state[fix_chat_key]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Get a quick base prediction for AI context (no adjustments yet)
                try:
                    base_pred = model.predict(home, away, matches)
                except Exception:
                    base_pred = None

                user_msg = st.chat_input(
                    f"Ask about {home} vs {away} or add context...",
                    key=f"chat_input_{home}_{away}",
                )
                if user_msg:
                    with st.chat_message("user"):
                        st.markdown(user_msg)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                reply, updated = chat_with_ai(
                                    user_message=user_msg,
                                    home_team=home,
                                    away_team=away,
                                    matches=matches,
                                    prediction=base_pred,
                                    dc_model=model.dc_model,
                                    history=list(st.session_state[fix_chat_key]),
                                )
                                st.markdown(reply)
                                st.session_state[fix_chat_key] = updated
                            except Exception as e:
                                st.error(f"AI error: {e}")

                # Context adjustment notes (separate from chat)
                st.markdown("---")
                notes = st.text_input(
                    "⚙️ Add prediction adjustment (injuries, suspensions...)",
                    key=f"notes_{home}_{away}",
                    placeholder='e.g. "no Yamal" · "new coach" · "must win"',
                )
                if notes:
                    adj = parse_context(notes, home, away)
                    if adj["log"]:
                        st.markdown(f'<div class="adj-box">{format_adjustments(adj)}</div>', unsafe_allow_html=True)

            try:
                pred = model.predict(home, away, matches, adjustment=adj if adj else None)
            except Exception as e:
                st.warning(f"Could not predict {home} vs {away}: {e}")
                continue

            hw, d, aw = pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]
            markets = pred["markets"]

            c1, c2, c3 = st.columns([5, 4, 5])
            with c1:
                st.markdown(f"**{home}**")
                st.markdown(f'<span class="prob-hw">{hw:.0%}</span>', unsafe_allow_html=True)
            with c2:
                adj_icon = " 🔧" if pred.get("adjusted") else ""
                st.markdown(
                    f'<div class="score-big">{pred["predicted_home_goals"]} – {pred["predicted_away_goals"]}{adj_icon}</div>'
                    f'<div style="text-align:center;font-size:0.9rem;font-weight:600;color:#37003c">⏰ {kickoff} {md_str}</div>'
                    f'<div style="text-align:center"><span class="prob-d">{d:.0%} Draw</span></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(f"**{away}**")
                st.markdown(f'<span class="prob-aw">{aw:.0%}</span>', unsafe_allow_html=True)

            st.plotly_chart(render_prob_bar(hw, d, aw, home, away), use_container_width=True)

            # Markets row
            if market_filter in ["All Markets", "BTTS"]:
                st.markdown(market_chips(markets), unsafe_allow_html=True)

            # Confidence badge
            conf = pred["confidence"]
            conf_class = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-low")
            st.markdown(
                f'<span class="{conf_class}">{pred["recommendation"]} — {conf} confidence</span>'
                f'&nbsp; xG: {pred["xg_home"]} – {pred["xg_away"]}',
                unsafe_allow_html=True,
            )

            with st.expander(f"🧠 Why? — {home} vs {away}"):
                exp = explain_prediction(home, away, matches, model.dc_model, pred, adj if adj else None)
                render_explanation(exp, home, away)

            st.markdown("<hr style='margin:8px 0;border-color:#eee'>", unsafe_allow_html=True)


# ─── PAGE: Match Predictor ───────────────────────────────────────────────────

elif page == "Match Predictor":
    st.markdown(f'<div class="big-title">🎯 Custom Match Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="comp-badge">{cfg["flag"]} {cfg["name"]}</div>', unsafe_allow_html=True)

    teams = sorted(model.dc_model.teams)
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Home Team", teams)
    with c2:
        away = st.selectbox("Away Team", [t for t in teams if t != home])

    if st.button("⚽ Predict", type="primary"):
        pred = model.predict(home, away, matches, adjustment=adj if adj else None)
        hw, d, aw = pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]
        markets = pred["markets"]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{home} Win", f"{hw:.1%}")
        col2.metric("Draw", f"{d:.1%}")
        col3.metric(f"{away} Win", f"{aw:.1%}")

        adj_icon = " 🔧 (adjusted)" if pred.get("adjusted") else ""
        st.markdown(
            f'<div style="text-align:center;font-size:1.7rem;font-weight:800;margin:12px 0">'
            f'{home} <b>{pred["predicted_home_goals"]}</b> – <b>{pred["predicted_away_goals"]}</b> {away}{adj_icon}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Score probability: {pred['score_probability']:.1%}  ·  xG: {pred['xg_home']} – {pred['xg_away']}")

        st.markdown("---")
        st.subheader("Betting Markets")
        col1, col2, col3 = st.columns(3)
        col1.metric("BTTS Yes", f"{markets['btts_yes']:.1%}")
        col1.metric("BTTS No",  f"{markets['btts_no']:.1%}")
        col2.metric("Over 1.5", f"{markets['over_15']:.1%}")
        col2.metric("Over 2.5", f"{markets['over_25']:.1%}")
        col3.metric("Over 3.5", f"{markets['over_35']:.1%}")
        col3.metric("Under 2.5", f"{markets['under_25']:.1%}")

        st.markdown("---")
        st.subheader("Why this prediction?")
        with st.spinner("Building explanation..."):
            exp = explain_prediction(home, away, matches, model.dc_model, pred, None)
        render_explanation(exp, home, away)

        st.markdown("---")
        st.subheader("Score Probability Matrix")
        matrix = model.dc_model.predict_score_matrix(
            home, away, max_goals=6,
            home_atk_adj=adj.get("home_atk", 1.0) if adj else 1.0,
            home_def_adj=adj.get("home_def", 1.0) if adj else 1.0,
            away_atk_adj=adj.get("away_atk", 1.0) if adj else 1.0,
            away_def_adj=adj.get("away_def", 1.0) if adj else 1.0,
        )
        fig = px.imshow(
            matrix * 100,
            x=[f"{away} {i}" for i in range(7)],
            y=[f"{home} {i}" for i in range(7)],
            text_auto=".1f",
            color_continuous_scale="Blues",
            title="Scoreline probabilities (%)",
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

        # ── AI Chat ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💬 Chat about this match")
        st.caption("Ask anything — injuries, tactics, team news, betting advice...")

        # Store chat history in session state per match
        chat_key = f"chat_{home}_{away}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        # Display chat history
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input(f"Ask about {home} vs {away}...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply, updated_history = chat_with_ai(
                            user_message=user_input,
                            home_team=home,
                            away_team=away,
                            matches=matches,
                            prediction=pred,
                            dc_model=model.dc_model,
                            history=list(st.session_state[chat_key]),
                        )
                        st.markdown(reply)
                        st.session_state[chat_key] = updated_history
                    except Exception as e:
                        st.error(f"AI chat error: {e}")

        if st.session_state[chat_key] and st.button("🗑️ Clear chat", key=f"clear_{home}_{away}"):
            st.session_state[chat_key] = []
            st.rerun()

# ─── PAGE: Team Ratings ───────────────────────────────────────────────────────

elif page == "Team Ratings":
    st.markdown(f'<div class="big-title">📊 Team Ratings — {cfg["name"]}</div>', unsafe_allow_html=True)

    ratings = model.dc_model.get_team_ratings()
    tab1, tab2, tab3 = st.tabs(["Attack", "Defense", "League Table"])

    with tab1:
        fig = px.bar(
            ratings.sort_values("Attack"),
            x="Attack", y="Team", orientation="h",
            color="Attack", color_continuous_scale="Greens",
            title="Attack Strength",
        )
        fig.update_layout(height=max(400, len(ratings) * 22), yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            ratings.sort_values("Defense", ascending=False),
            x="Defense", y="Team", orientation="h",
            color="Defense", color_continuous_scale="Reds_r",
            title="Defense Strength (lower = better)",
        )
        fig.update_layout(height=max(400, len(ratings) * 22), yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        standings = _load_standings(comp)
        if not standings.empty:
            st.dataframe(standings, use_container_width=True, hide_index=True)
        else:
            st.dataframe(ratings, use_container_width=True, hide_index=True)


# ─── PAGE: Model Performance ──────────────────────────────────────────────────

elif page == "Model Performance":
    st.markdown(f'<div class="big-title">📈 Model Performance — {cfg["name"]}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    if model.cv_accuracy:
        c1.metric("XGBoost CV Accuracy", f"{model.cv_accuracy:.1%}")
    c2.metric("Training Matches", f"{model.n_matches:,}")
    c3.metric("Teams in Model", f"{len(model.dc_model.teams)}")

    st.markdown("---")
    st.subheader("Backtest — Last 50 Matches")
    with st.spinner("Running backtest..."):
        bt = model.backtest(matches, n_last=50)

    if not bt.empty:
        acc = bt["Correct"].mean()
        st.metric("Backtest Accuracy", f"{acc:.1%}")

        def highlight(row):
            color = "#d4edda" if row["Correct"] else "#f8d7da"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(bt.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)

        st.markdown("---")
        conf_acc = bt.groupby("Confidence")["Correct"].agg(["mean", "count"]).reset_index()
        conf_acc.columns = ["Confidence", "Accuracy", "Count"]
        conf_acc["Accuracy"] = conf_acc["Accuracy"].map("{:.1%}".format)
        st.subheader("Accuracy by Confidence")
        st.dataframe(conf_acc, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for backtest.")
