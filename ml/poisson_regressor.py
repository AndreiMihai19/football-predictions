"""Poisson regression base model with Dixon-Coles correction for stacked ensemble."""

import numpy as np
from xgboost import XGBRegressor
from scipy.stats import poisson
from scipy.optimize import minimize_scalar


MAX_GOALS = 6


def dixon_coles_tau(home_goals, away_goals, lambda_h, lambda_a, rho):
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_h * lambda_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lambda_h, lambda_a, rho=0.0, max_goals=MAX_GOALS):
    h_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_h)
    a_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_a)
    M = np.outer(h_pmf, a_pmf)
    if rho != 0.0:
        for i in range(2):
            for j in range(2):
                M[i, j] *= dixon_coles_tau(i, j, lambda_h, lambda_a, rho)
    s = M.sum()
    if s > 0:
        M = M / s
    return M


def wdl_from_lambdas(lambda_h, lambda_a, rho=0.0):
    M = score_matrix(lambda_h, lambda_a, rho)
    p_home = np.tril(M, k=-1).sum()
    p_away = np.triu(M, k=+1).sum()
    p_draw = np.trace(M)
    return np.array([p_away, p_home, p_draw])


def fit_dixon_coles_rho(y_home_goals, y_away_goals, lambda_h_pred, lambda_a_pred):
    def neg_log_lik(rho):
        ll = 0.0
        for hg, ag, lh, la in zip(y_home_goals, y_away_goals, lambda_h_pred, lambda_a_pred):
            if hg <= 1 and ag <= 1:
                tau = dixon_coles_tau(hg, ag, lh, la, rho)
                if tau <= 0:
                    return 1e10
                ll += np.log(tau)
        return -ll

    res = minimize_scalar(neg_log_lik, bounds=(-0.15, 0.15), method="bounded")
    return float(res.x) if res.success else 0.0


class PoissonGoalsRegressor:
    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05,
                 random_state=42, use_dixon_coles=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.use_dixon_coles = use_dixon_coles
        self.classes_ = np.array([0, 1, 2])
        self.rho_ = 0.0
        self.xgb_home_ = None
        self.xgb_away_ = None

    def _make_xgb(self):
        return XGBRegressor(
            objective="count:poisson",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(self, X, y, y_home_goals=None, y_away_goals=None, sample_weight=None):
        if y_home_goals is None or y_away_goals is None:
            raise ValueError(
                "PoissonGoalsRegressor.fit() necesita y_home_goals si y_away_goals"
            )

        y_home_goals = np.asarray(y_home_goals)
        y_away_goals = np.asarray(y_away_goals)

        self.xgb_home_ = self._make_xgb()
        self.xgb_away_ = self._make_xgb()

        self.xgb_home_.fit(X, y_home_goals)
        self.xgb_away_.fit(X, y_away_goals)

        if self.use_dixon_coles:
            lambda_h_train = self.xgb_home_.predict(X)
            lambda_a_train = self.xgb_away_.predict(X)
            lambda_h_train = np.maximum(lambda_h_train, 0.05)
            lambda_a_train = np.maximum(lambda_a_train, 0.05)
            self.rho_ = fit_dixon_coles_rho(
                y_home_goals, y_away_goals, lambda_h_train, lambda_a_train
            )

        return self

    def predict_proba(self, X):
        if self.xgb_home_ is None:
            raise RuntimeError("PoissonGoalsRegressor not fitted")
        lambda_h = np.maximum(self.xgb_home_.predict(X), 0.05)
        lambda_a = np.maximum(self.xgb_away_.predict(X), 0.05)
        probs = np.zeros((len(X), 3))
        for i in range(len(X)):
            probs[i] = wdl_from_lambdas(lambda_h[i], lambda_a[i], self.rho_)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_score_distribution(self, X):
        lambda_h = np.maximum(self.xgb_home_.predict(X), 0.05)
        lambda_a = np.maximum(self.xgb_away_.predict(X), 0.05)
        out = np.zeros((len(X), MAX_GOALS + 1, MAX_GOALS + 1))
        for i in range(len(X)):
            out[i] = score_matrix(lambda_h[i], lambda_a[i], self.rho_)
        return out
