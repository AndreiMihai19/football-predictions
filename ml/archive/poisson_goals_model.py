"""
poisson_goals_model.py
======================
Poisson Goals Classifier — al 4-lea base learner pentru stacked ensemble v3.

Idee academica (Maher 1982, Dixon-Coles 1997, Karlis-Ntzoufras 2003):
In loc sa cladim direct probabilitati pentru rezultat (Win/Draw/Loss),
modelam separat numarul de goluri marcate de fiecare echipa folosind
distributii Poisson, apoi calculam probabilitatile rezultatelor din
matricea scorurilor posibile.

Avantaj fata de clasificatori directi:
- Modelarea separata a atacului si apararii (interpretabilitate)
- Surprinde structura statistica a fotbalului (golurile sunt evenimente Poisson)
- Standard in literatura academica (citat in 1000+ articole)
- Surprinde corect ca egalurile sunt mai probabile cand λ_home ≈ λ_away

Implementare:
1. PoissonRegressor pentru home_goals (input: features → λ_home)
2. PoissonRegressor pentru away_goals (input: features → λ_away)
3. predict_proba: pentru fiecare meci, calculeaza P(rezultat) din matricea
   scorurilor posibile (i, j) cu i, j in 0..MAX_GOALS folosind PMF Poisson:
     P(home_win) = sum_{i>j} Pois(i; λ_h) * Pois(j; λ_a)
     P(away_win) = sum_{i<j} Pois(i; λ_h) * Pois(j; λ_a)
     P(draw)    = sum_{i==j} Pois(i; λ_h) * Pois(j; λ_a)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import poisson
from xgboost import XGBRegressor


MAX_GOALS = 10  # Maxim goluri pe care le consideram (P(>10 goluri) ≈ 0)


class PoissonGoalsClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificator probabilistic pentru rezultatele meciurilor de fotbal,
    bazat pe modelarea Poisson a golurilor.

    Compatibil cu sklearn API (fit, predict, predict_proba) pentru
    integrare in stacking ensembles.

    Note:
        - Necesita y_home_goals si y_away_goals la fit (separate de y_result)
        - In stacking: inlocuim y cu un wrapper care expune ambele
        - clases_ este [0, 1, 2] (away/home/draw) la fel ca XGB/LGB/Cat
    """

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, max_goals=MAX_GOALS):
        """
        Args:
            n_estimators, max_depth, learning_rate, subsample, colsample_bytree:
                hiperparametri XGBoost (obiectiv count:poisson)
            max_goals: numar maxim de goluri considerat in matricea scorurilor
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_goals = max_goals
        self.classes_ = np.array([0, 1, 2])

    def _make_regressor(self):
        return XGBRegressor(
            objective="count:poisson",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(self, X, y, sample_weight=None, y_home_goals=None, y_away_goals=None):
        """
        Antreneaza 2 regresii Poisson (XGBoost cu objective=count:poisson) —
        una pentru home_goals, alta pentru away_goals.

        Args:
            X: matrice features
            y: rezultatul (0=away, 1=home, 2=draw) — folosit doar pentru classes_
            y_home_goals: numar goluri marcate de echipa de acasa (target Poisson #1)
            y_away_goals: numar goluri marcate de echipa de deplasare (target Poisson #2)
            sample_weight: ponderi optionale per esantion
        """
        if y_home_goals is None or y_away_goals is None:
            raise ValueError(
                "PoissonGoalsClassifier necesita y_home_goals si y_away_goals la fit. "
                "Pass them explicit ca argumente keyword."
            )

        y_home_goals = np.asarray(y_home_goals).astype(float)
        y_away_goals = np.asarray(y_away_goals).astype(float)

        # XGBoost cu obiectiv Poisson — gestioneaza features standardizate
        # mult mai bine decat GLM linear (PoissonRegressor avea coeficienti
        # prea mici cand features erau StandardScaler-ate -> lambda uniforme).
        self.home_model_ = self._make_regressor()
        self.away_model_ = self._make_regressor()

        if sample_weight is not None:
            self.home_model_.fit(X, y_home_goals, sample_weight=sample_weight)
            self.away_model_.fit(X, y_away_goals, sample_weight=sample_weight)
        else:
            self.home_model_.fit(X, y_home_goals)
            self.away_model_.fit(X, y_away_goals)

        return self

    def _outcome_proba_from_lambdas(self, lambda_home, lambda_away):
        """
        Calculeaza (P(away_win), P(home_win), P(draw)) pentru o pereche (λ_h, λ_a)
        prin sumare peste matricea scorurilor posibile.
        """
        # PMF Poisson pentru i = 0..MAX_GOALS
        i_vals = np.arange(self.max_goals + 1)
        pmf_home = poisson.pmf(i_vals, lambda_home)  # vector (MAX_GOALS+1,)
        pmf_away = poisson.pmf(i_vals, lambda_away)

        # Matricea P(home_goals=i, away_goals=j) = pmf_home[i] * pmf_away[j]
        score_matrix = np.outer(pmf_home, pmf_away)

        # Indici i (rows) = home goals, j (cols) = away goals
        rows, cols = np.indices(score_matrix.shape)

        p_home_win = score_matrix[rows > cols].sum()
        p_away_win = score_matrix[rows < cols].sum()
        p_draw     = score_matrix[rows == cols].sum()

        # Renormalizam (truncatura la MAX_GOALS poate scoate ~0.001 din masa)
        total = p_away_win + p_home_win + p_draw
        if total > 0:
            p_away_win /= total
            p_home_win /= total
            p_draw     /= total

        return p_away_win, p_home_win, p_draw

    def predict_proba(self, X):
        """
        Returns: matrice (n_samples, 3) cu [P(away), P(home), P(draw)].
        Aceeasi ordonare ca XGB/LGB/Cat (clasele [0, 1, 2]).
        """
        # Lambda predictions (clipped la min 0.1 pentru stabilitate numerica)
        lambda_home = np.clip(self.home_model_.predict(X), 0.1, 10.0)
        lambda_away = np.clip(self.away_model_.predict(X), 0.1, 10.0)

        n = len(X)
        proba = np.zeros((n, 3))

        for k in range(n):
            p_a, p_h, p_d = self._outcome_proba_from_lambdas(
                lambda_home[k], lambda_away[k]
            )
            proba[k, 0] = p_a
            proba[k, 1] = p_h
            proba[k, 2] = p_d

        return proba

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_lambdas(self, X):
        """Helper diagnostic: returneaza (λ_home, λ_away) per meci."""
        lambda_home = np.clip(self.home_model_.predict(X), 0.1, 10.0)
        lambda_away = np.clip(self.away_model_.predict(X), 0.1, 10.0)
        return lambda_home, lambda_away


# ============================================================
# Test rapid daca ruleaza standalone
# ============================================================
if __name__ == "__main__":
    import os, json
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    ML_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  POISSON GOALS MODEL — Standalone Test")
    print("=" * 60)

    # Incarcam datele
    X_train = np.load(os.path.join(ML_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(ML_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(ML_DIR, "y_train.npy")).astype(int)
    y_test  = np.load(os.path.join(ML_DIR, "y_test.npy")).astype(int)

    with open(os.path.join(ML_DIR, "metadata.json")) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    dead_idx = [i for i, n in enumerate(feature_names) if n == "home_advantage"]
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test  = np.delete(X_test,  dead_idx, axis=1)

    # Avem nevoie de home_goals si away_goals (din data_processed.csv)
    df_proc = pd.read_csv(os.path.join(ML_DIR, "data_processed.csv"))
    df_proc = df_proc.sort_values("date").reset_index(drop=True)

    n_train = len(y_train)
    home_goals_train = df_proc.iloc[:n_train]["home_goals"].values
    away_goals_train = df_proc.iloc[:n_train]["away_goals"].values
    home_goals_test  = df_proc.iloc[n_train:n_train + len(y_test)]["home_goals"].values
    away_goals_test  = df_proc.iloc[n_train:n_train + len(y_test)]["away_goals"].values

    print(f"\n  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Mean home_goals train: {home_goals_train.mean():.2f}")
    print(f"  Mean away_goals train: {away_goals_train.mean():.2f}")

    # Antrenam
    print("\n[1/3] Training PoissonGoalsClassifier (XGBoost count:poisson)...")
    model = PoissonGoalsClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05
    )
    model.fit(X_train, y_train,
              y_home_goals=home_goals_train,
              y_away_goals=away_goals_train)
    print("  Done")

    # Evaluam
    print("\n[2/3] Evaluating...")
    y_proba = model.predict_proba(X_test)
    y_pred = y_proba.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    ll = log_loss(y_test, y_proba, labels=[0, 1, 2])

    print(f"\n  Accuracy:    {acc*100:.2f}%")
    print(f"  F1 macro:    {f1m:.4f}")
    print(f"  F1 Away:     {f1_each[0]:.4f}")
    print(f"  F1 Home:     {f1_each[1]:.4f}")
    print(f"  F1 Draw:     {f1_each[2]:.4f}")
    print(f"  Log Loss:    {ll:.4f}")

    # Diagnostic: λ-uri pentru cateva meciuri
    print("\n[3/3] Sample λ predictions (first 5 test matches):")
    lam_h, lam_a = model.predict_lambdas(X_test[:5])
    for i in range(5):
        print(f"  Match {i+1}: λ_home={lam_h[i]:.2f}, λ_away={lam_a[i]:.2f}, "
              f"actual={home_goals_test[i]}-{away_goals_test[i]}, "
              f"pred_proba=[A:{y_proba[i,0]:.2f}, H:{y_proba[i,1]:.2f}, D:{y_proba[i,2]:.2f}]")

    print("\n" + "=" * 60)
    print("  POISSON STANDALONE TEST COMPLETE")
    print("=" * 60)
