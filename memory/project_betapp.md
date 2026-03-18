---
name: BetApp project context
description: Context for the football prediction app being built in D:/Betapp
type: project
---

Building a personal football bet predictor app. Currently supports Premier League, La Liga, and Champions League.

**Why:** Personal use, zero/low cost, ML-based score predictions.

**Stack:** Python, Streamlit, Dixon-Coles model + XGBoost ensemble, SQLite-free (CSVs + API).

**Data sources:** football-data.co.uk (historical CSVs) + football-data.org API (live fixtures/results). API key stored in .env.

**How to apply:** When adding features, keep it free/local. User shapes the platform iteratively.

**Key features built:**
- PL, La Liga, UCL predictions
- BTTS, Over/Under, 1X2 markets
- Human context notes (keyword-based adjustments to attack/defense)
- Same-day momentum tracker
- Prediction explanations ("why" feature)
- Score probability matrix

**User observations to incorporate:**
- Same-day results on Fotmob often cluster (home day, away day, low scoring day) — already tracked
- User wants to add human intel like "no Yamal today" — keyword adjustment engine built

**Known facts about player movements to keep in mind:**
- Harry Kane plays for Bayern Munich (not Spurs) since summer 2023
