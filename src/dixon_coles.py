import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
import pickle
from pathlib import Path


class DixonColesModel:
    def __init__(self):
        self.attack = {}
        self.defense = {}
        self.home_adv = 0.1
        self.rho = -0.1
        self.teams = []
        self.fitted = False

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return max(1e-10, 1 - lam * mu * rho)
        elif x == 0 and y == 1:
            return max(1e-10, 1 + lam * rho)
        elif x == 1 and y == 0:
            return max(1e-10, 1 + mu * rho)
        elif x == 1 and y == 1:
            return max(1e-10, 1 - rho)
        return 1.0

    def _neg_log_likelihood(self, params, matches):
        n = len(self.teams)
        attack = params[:n]
        defense = params[n:2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]

        log_lik = 0.0
        for _, row in matches.iterrows():
            try:
                h = self.teams.index(row["HomeTeam"])
                a = self.teams.index(row["AwayTeam"])
            except ValueError:
                continue
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            lam = np.exp(attack[h] - defense[a] + home_adv)
            mu = np.exp(attack[a] - defense[h])
            t = self._tau(hg, ag, lam, mu, rho)
            if t <= 0:
                return 1e10
            log_lik += np.log(t) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
        return -log_lik

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        self.teams = sorted(list(set(matches["HomeTeam"].unique()) | set(matches["AwayTeam"].unique())))
        n = len(self.teams)
        x0 = np.array([0.3] * n + [0.3] * n + [0.1, -0.1])
        bounds = [(-2.0, 3.0)] * n + [(-2.0, 3.0)] * n + [(0.0, 1.0)] + [(-0.99, 0.0)]
        result = minimize(
            self._neg_log_likelihood, x0, args=(matches,),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        params = result.x
        self.attack = {t: params[i] for i, t in enumerate(self.teams)}
        self.defense = {t: params[n + i] for i, t in enumerate(self.teams)}
        self.home_adv = params[2 * n]
        self.rho = params[2 * n + 1]
        self.fitted = True
        return self

    def _get_lambdas(self, home_team: str, away_team: str,
                     home_atk_adj=1.0, home_def_adj=1.0,
                     away_atk_adj=1.0, away_def_adj=1.0):
        avg_atk = np.mean(list(self.attack.values())) if self.attack else 0.3
        avg_def = np.mean(list(self.defense.values())) if self.defense else 0.3
        ha = self.attack.get(home_team, avg_atk)
        hd = self.defense.get(home_team, avg_def)
        aa = self.attack.get(away_team, avg_atk)
        ad = self.defense.get(away_team, avg_def)
        lam = np.exp(ha * home_atk_adj - ad * away_def_adj + self.home_adv)
        mu = np.exp(aa * away_atk_adj - hd * home_def_adj)
        return max(lam, 0.01), max(mu, 0.01)

    def predict_score_matrix(self, home_team: str, away_team: str, max_goals: int = 10,
                              home_atk_adj=1.0, home_def_adj=1.0,
                              away_atk_adj=1.0, away_def_adj=1.0) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        lam, mu = self._get_lambdas(home_team, away_team, home_atk_adj, home_def_adj, away_atk_adj, away_def_adj)
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                t = self._tau(hg, ag, lam, mu, self.rho)
                matrix[hg, ag] = t * poisson.pmf(hg, lam) * poisson.pmf(ag, mu)
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def predict_outcome_probs(self, home_team: str, away_team: str,
                               home_atk_adj=1.0, home_def_adj=1.0,
                               away_atk_adj=1.0, away_def_adj=1.0) -> dict:
        m = self.predict_score_matrix(home_team, away_team, home_atk_adj=home_atk_adj,
                                       home_def_adj=home_def_adj, away_atk_adj=away_atk_adj,
                                       away_def_adj=away_def_adj)
        return {
            "home_win": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away_win": float(np.triu(m, 1).sum()),
        }

    def predict_most_likely_score(self, home_team: str, away_team: str,
                                   home_atk_adj=1.0, home_def_adj=1.0,
                                   away_atk_adj=1.0, away_def_adj=1.0) -> tuple:
        m = self.predict_score_matrix(home_team, away_team, home_atk_adj=home_atk_adj,
                                       home_def_adj=home_def_adj, away_atk_adj=away_atk_adj,
                                       away_def_adj=away_def_adj)
        idx = np.unravel_index(m.argmax(), m.shape)
        return int(idx[0]), int(idx[1]), float(m[idx])

    def get_expected_goals(self, home_team: str, away_team: str,
                            home_atk_adj=1.0, home_def_adj=1.0,
                            away_atk_adj=1.0, away_def_adj=1.0) -> tuple:
        lam, mu = self._get_lambdas(home_team, away_team, home_atk_adj, home_def_adj, away_atk_adj, away_def_adj)
        return round(float(lam), 2), round(float(mu), 2)

    @staticmethod
    def predict_markets(score_matrix: np.ndarray) -> dict:
        """Calculate betting market probabilities from score matrix."""
        m = score_matrix
        n = m.shape[0]

        btts = sum(m[h, a] for h in range(1, n) for a in range(1, n))
        over_15 = sum(m[h, a] for h in range(n) for a in range(n) if h + a >= 2)
        over_25 = sum(m[h, a] for h in range(n) for a in range(n) if h + a >= 3)
        over_35 = sum(m[h, a] for h in range(n) for a in range(n) if h + a >= 4)

        return {
            "btts_yes": round(float(btts), 4),
            "btts_no": round(float(1 - btts), 4),
            "over_15": round(float(over_15), 4),
            "over_25": round(float(over_25), 4),
            "over_35": round(float(over_35), 4),
            "under_25": round(float(1 - over_25), 4),
        }

    @staticmethod
    def predict_goals_ranges(score_matrix: np.ndarray) -> dict:
        """Probability of total goals falling in each range."""
        m = score_matrix
        n = m.shape[0]
        ranges = {"0-1": 0.0, "2-3": 0.0, "4-5": 0.0, "6+": 0.0}
        for h in range(n):
            for a in range(n):
                total = h + a
                if total <= 1:
                    ranges["0-1"] += m[h, a]
                elif total <= 3:
                    ranges["2-3"] += m[h, a]
                elif total <= 5:
                    ranges["4-5"] += m[h, a]
                else:
                    ranges["6+"] += m[h, a]
        return {k: round(float(v), 4) for k, v in ranges.items()}

    def get_team_ratings(self) -> pd.DataFrame:
        data = [
            {
                "Team": t,
                "Attack": round(self.attack.get(t, 0), 3),
                "Defense": round(self.defense.get(t, 0), 3),
                "Net": round(self.attack.get(t, 0) - self.defense.get(t, 0), 3),
            }
            for t in self.teams
        ]
        return pd.DataFrame(data).sort_values("Net", ascending=False).reset_index(drop=True)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "DixonColesModel":
        with open(path, "rb") as f:
            return pickle.load(f)
