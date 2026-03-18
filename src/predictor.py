import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from src.dixon_coles import DixonColesModel
from src.features import build_feature_matrix, get_features_for_match, FEATURE_COLS


class MatchPredictor:
    DC_WEIGHT = 0.55
    XGB_WEIGHT = 0.45

    def __init__(self):
        self.dc_model = DixonColesModel()
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        )
        self.fitted = False
        self.cv_accuracy = None
        self.n_matches = 0

    def train(self, matches: pd.DataFrame) -> "MatchPredictor":
        self.n_matches = len(matches)
        print("Fitting Dixon-Coles model...")
        self.dc_model.fit(matches)

        print("Building training features...")
        feat_df = build_feature_matrix(matches, self.dc_model)
        feat_df = feat_df.dropna(subset=FEATURE_COLS)

        if len(feat_df) < 100:
            print(f"Only {len(feat_df)} samples — using DC only.")
            self.fitted = True
            return self

        X = feat_df[FEATURE_COLS].values
        y = feat_df["result"].values
        scores = cross_val_score(self.xgb, X, y, cv=5, scoring="accuracy")
        self.cv_accuracy = float(scores.mean())
        print(f"XGBoost CV accuracy: {self.cv_accuracy:.3f}")
        self.xgb.fit(X, y)
        self.fitted = True
        return self

    def predict(self, home_team: str, away_team: str, matches: pd.DataFrame,
                adjustment: dict = None) -> dict:
        """
        Predict match. adjustment dict from adjustments.parse_context():
        { home_atk, home_def, away_atk, away_def }
        """
        adj = adjustment or {}
        ha = adj.get("home_atk", 1.0)
        hd = adj.get("home_def", 1.0)
        aa = adj.get("away_atk", 1.0)
        ad = adj.get("away_def", 1.0)

        # Dixon-Coles with adjustments
        dc_probs = self.dc_model.predict_outcome_probs(
            home_team, away_team, home_atk_adj=ha, home_def_adj=hd,
            away_atk_adj=aa, away_def_adj=ad)
        pred_hg, pred_ag, score_prob = self.dc_model.predict_most_likely_score(
            home_team, away_team, home_atk_adj=ha, home_def_adj=hd,
            away_atk_adj=aa, away_def_adj=ad)
        xg_home, xg_away = self.dc_model.get_expected_goals(
            home_team, away_team, home_atk_adj=ha, home_def_adj=hd,
            away_atk_adj=aa, away_def_adj=ad)

        # Score matrix for markets
        matrix = self.dc_model.predict_score_matrix(
            home_team, away_team, home_atk_adj=ha, home_def_adj=hd,
            away_atk_adj=aa, away_def_adj=ad)
        markets = DixonColesModel.predict_markets(matrix)

        # XGBoost (no adjustments — uses historical features)
        xgb_probs = None
        feat = get_features_for_match(matches, home_team, away_team, pd.Timestamp.now(), self.dc_model)
        if self.fitted and hasattr(self.xgb, "feature_importances_"):
            X = np.array([[feat.get(c, 0.0) for c in FEATURE_COLS]])
            raw = self.xgb.predict_proba(X)[0]
            xgb_probs = {"home_win": float(raw[0]), "draw": float(raw[1]), "away_win": float(raw[2])}

        # Ensemble
        if xgb_probs:
            probs = {
                "home_win": self.DC_WEIGHT * dc_probs["home_win"] + self.XGB_WEIGHT * xgb_probs["home_win"],
                "draw": self.DC_WEIGHT * dc_probs["draw"] + self.XGB_WEIGHT * xgb_probs["draw"],
                "away_win": self.DC_WEIGHT * dc_probs["away_win"] + self.XGB_WEIGHT * xgb_probs["away_win"],
            }
        else:
            probs = dc_probs

        rec, conf = self._recommend(probs)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(probs["home_win"], 4),
            "draw_prob": round(probs["draw"], 4),
            "away_win_prob": round(probs["away_win"], 4),
            "predicted_home_goals": pred_hg,
            "predicted_away_goals": pred_ag,
            "score_probability": round(score_prob, 4),
            "xg_home": xg_home,
            "xg_away": xg_away,
            "markets": markets,
            "recommendation": rec,
            "confidence": conf,
            "adjusted": bool(adj),
        }

    def _recommend(self, probs: dict) -> tuple:
        best = max(probs, key=probs.get)
        prob = probs[best]
        label = {"home_win": "Home Win", "draw": "Draw", "away_win": "Away Win"}[best]
        confidence = "High" if prob >= 0.55 else ("Medium" if prob >= 0.40 else "Low")
        return label, confidence

    def backtest(self, matches: pd.DataFrame, n_last: int = 50) -> pd.DataFrame:
        recent = matches.tail(n_last).copy()
        rows = []
        for _, row in recent.iterrows():
            prior = matches[matches["Date"] < row["Date"]]
            if len(prior) < 30:
                continue
            try:
                pred = self.predict(row["HomeTeam"], row["AwayTeam"], prior)
                best = max([("home_win", pred["home_win_prob"]),
                            ("draw", pred["draw_prob"]),
                            ("away_win", pred["away_win_prob"])], key=lambda x: x[1])[0]
                actual = {"H": "home_win", "D": "draw", "A": "away_win"}[row["FTR"]]
                rows.append({
                    "Date": row["Date"].strftime("%d %b %Y"),
                    "Match": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    "Result": row["FTR"],
                    "Predicted": {"home_win": "H", "draw": "D", "away_win": "A"}[best],
                    "Correct": actual == best,
                    "Home%": f"{pred['home_win_prob']:.0%}",
                    "Draw%": f"{pred['draw_prob']:.0%}",
                    "Away%": f"{pred['away_win_prob']:.0%}",
                    "Confidence": pred["confidence"],
                })
            except Exception:
                continue
        return pd.DataFrame(rows)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "MatchPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
