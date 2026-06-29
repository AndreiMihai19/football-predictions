"""
experiment_optuna.py
====================
Hyperparameter search cu Optuna (Bayesian optimization) pe XGBoost.
Foloseste params extinsi (reg_alpha, reg_lambda, min_child_weight, gamma).
Apoi antreneaza stacked ensemble (XGB+LGB+Cat) cu best params si compara.

Output: experiment_optuna.json
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import sklearn
sklearn.set_config(enable_metadata_routing=True)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from stacked_ensemble import StackedEnsemble

ML_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
N_TRIALS = 80


def p(filename):
    return os.path.join(ML_DIR, filename)


def load_data():
    X_train = np.load(p("X_train.npy"))
    X_test = np.load(p("X_test.npy"))
    y_train = np.load(p("y_train.npy")).astype(int)
    y_test = np.load(p("y_test.npy")).astype(int)

    with open(p("metadata.json")) as f:
        full_meta = json.load(f)
    full_feature_names = full_meta["feature_names"]

    DEAD = ["home_advantage"]
    dead_idx = [i for i, n in enumerate(full_feature_names) if n in DEAD]
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test = np.delete(X_test, dead_idx, axis=1)

    return X_train, X_test, y_train, y_test


def objective(trial, X, y, sample_weight):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    tscv = TimeSeriesSplit(n_splits=3)
    f1_scores = []

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        sw_tr = sample_weight[tr_idx]

        model = XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
        model.set_fit_request(sample_weight=True)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

    return np.mean(f1_scores)


def train_stacked(X_train, y_train, X_test, y_test, params, label):
    """Antreneaza stacked ensemble cu params dati si returneaza metrici."""
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    sample_weight = np.array([weight_dict[y] for y in y_train])

    xgb = XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.set_fit_request(sample_weight=True)

    # LGBM si CatBoost folosesc parametri compatibili (subset din XGB)
    lgb = LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 0.0),
        min_child_weight=params.get("min_child_weight", 1),
        subsample=params.get("subsample", 1.0),
        colsample_bytree=params.get("colsample_bytree", 1.0),
        class_weight=weight_dict,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    lgb.set_fit_request(sample_weight=True)

    cat_weights = [weight_dict[0], weight_dict[1], weight_dict[2]]
    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=params["n_estimators"],
        depth=min(params["max_depth"], 8),
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params.get("reg_lambda", 3.0),
        class_weights=cat_weights,
        random_state=RANDOM_STATE, verbose=0,
    )

    estimators = [("xgb", xgb), ("lgb", lgb), ("cat", cat)]
    stacked = StackedEnsemble(base_estimators=estimators)
    stacked.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = stacked.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    return {
        "label": label,
        "params": params,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_away": f1_each[0],
        "f1_home": f1_each[1],
        "f1_draw": f1_each[2],
        "cm": cm.tolist(),
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Optuna Hyperparameter Search")
    print("=" * 70)

    print("\n[1] Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    sample_weight = np.array([weight_dict[y] for y in y_train])

    print(f"\n[2] Optuna search ({N_TRIALS} trials, TSCV 3 folds, single XGBoost)...")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, sample_weight),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    print(f"\n  Best CV F1 macro: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k:<22}: {v:.6f}")
        else:
            print(f"    {k:<22}: {v}")

    # Baseline: GridSearchCV best (din v3)
    baseline_params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    print("\n[3] Training stacked ensemble — BASELINE (GridSearchCV best)...")
    r_base = train_stacked(X_train, y_train, X_test, y_test, baseline_params, "Baseline (GridSearchCV)")
    print(f"  Accuracy: {r_base['accuracy']*100:.2f}%  F1 macro: {r_base['f1_macro']:.4f}  F1 Draw: {r_base['f1_draw']:.4f}")

    print("\n[4] Training stacked ensemble — OPTUNA best params...")
    r_optuna = train_stacked(X_train, y_train, X_test, y_test, study.best_params, "Optuna best")
    print(f"  Accuracy: {r_optuna['accuracy']*100:.2f}%  F1 macro: {r_optuna['f1_macro']:.4f}  F1 Draw: {r_optuna['f1_draw']:.4f}")

    print("\n" + "=" * 70)
    print("  REZULTATE COMPARATIVE")
    print("=" * 70)
    print(f"  {'Scheme':<28} {'Accuracy':>10} {'F1 macro':>10} {'F1 Draw':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {r_base['label']:<28} {r_base['accuracy']*100:>9.2f}% {r_base['f1_macro']:>10.4f} {r_base['f1_draw']:>10.4f}")
    print(f"  {r_optuna['label']:<28} {r_optuna['accuracy']*100:>9.2f}% {r_optuna['f1_macro']:>10.4f} {r_optuna['f1_draw']:>10.4f}")

    delta_acc = (r_optuna['accuracy'] - r_base['accuracy']) * 100
    delta_f1 = r_optuna['f1_macro'] - r_base['f1_macro']
    delta_draw = r_optuna['f1_draw'] - r_base['f1_draw']
    print(f"\n  Delta:                    {delta_acc:>+9.2f}pp {delta_f1:>+10.4f} {delta_draw:>+10.4f}")

    # Save
    out = {
        "n_trials": N_TRIALS,
        "best_cv_f1_macro": float(study.best_value),
        "optuna_best_params": study.best_params,
        "baseline": r_base,
        "optuna": r_optuna,
        "delta": {
            "accuracy_pp": round(delta_acc, 2),
            "f1_macro": round(delta_f1, 4),
            "f1_draw": round(delta_draw, 4),
        },
    }
    with open(p("experiment_optuna.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: experiment_optuna.json")


if __name__ == "__main__":
    main()
