"""
train_model_v3_poisson.py
=========================
Re-antreneaza StackedEnsemble v3 cu un al 4-lea base learner: PoissonGoalsClassifier.

Reuseste best_params din results_v3.json (skip GridSearchCV, ~30 min castig timp).
Pastreaza XGB + LGB + Cat exact ca v3, adauga Poisson, antreneaza meta-LR.

Output: model_v3_poisson.pkl, results_v3_poisson.json (NU suprascrie v3 original).
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, log_loss,
)
from sklearn.utils.class_weight import compute_class_weight
import sklearn
sklearn.set_config(enable_metadata_routing=True)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from stacked_ensemble import StackedEnsemble
from poisson_goals_model import PoissonGoalsClassifier

ML_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ML_DIR, "output")
RANDOM_STATE = 42


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
    feature_names = [n for n in full_feature_names if n not in DEAD]
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test = np.delete(X_test, dead_idx, axis=1)

    # Goluri reale (din data_processed.csv) — necesar pentru Poisson
    df = pd.read_csv(p("data_processed.csv"))
    df = df.sort_values("date").reset_index(drop=True)
    n_train = len(y_train)
    home_goals_train = df.iloc[:n_train]["home_goals"].values
    away_goals_train = df.iloc[:n_train]["away_goals"].values

    return (X_train, X_test, y_train, y_test, feature_names,
            home_goals_train, away_goals_train)


def main():
    print("=" * 60)
    print("  TRAIN v3 + POISSON — 4 base learners stacking")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    (X_train, X_test, y_train, y_test, feature_names,
     hg_train, ag_train) = load_data()
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Mean home_goals: {hg_train.mean():.2f}  away_goals: {ag_train.mean():.2f}")

    print("\n[2/5] Computing class weights...")
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    sample_weight = np.array([weight_dict[y] for y in y_train])
    for c, w in weight_dict.items():
        print(f"  Class {c}: weight {w:.4f}")

    # Best params din v3 (skip GridSearchCV)
    best_params = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 500}
    print(f"\n  Reusing best_params from v3: {best_params}")

    print("\n[3/5] Building 4-base ensemble...")
    xgb = XGBClassifier(
        **best_params, objective="multi:softprob", num_class=3,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
        verbosity=0, n_jobs=-1,
    )
    xgb.set_fit_request(sample_weight=True)

    lgb = LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        class_weight=weight_dict,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    lgb.set_fit_request(sample_weight=True)

    cat_weights = [weight_dict[0], weight_dict[1], weight_dict[2]]
    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=best_params["n_estimators"],
        depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        class_weights=cat_weights,
        random_state=RANDOM_STATE, verbose=0,
    )

    poisson = PoissonGoalsClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05
    )

    estimators = [
        ("xgb", xgb),
        ("lgb", lgb),
        ("cat", cat),
        ("poisson", poisson),
    ]
    print(f"  Bases: {[n for n, _ in estimators]}")

    print("\n[4/5] Training stacked ensemble...")
    stacked = StackedEnsemble(base_estimators=estimators)
    stacked.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        y_home_goals=hg_train,
        y_away_goals=ag_train,
    )

    print("\n[5/5] Evaluating on test set...")
    y_pred = stacked.predict(X_test)
    y_proba = stacked.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    ll = log_loss(y_test, y_proba, labels=[0, 1, 2])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    class_names = {0: "Away Win", 1: "Home Win", 2: "Draw"}

    print("\n  =====================================================")
    print(f"  Accuracy           : {acc*100:.2f}%")
    print(f"  Precision (macro)  : {prec:.4f}")
    print(f"  Recall (macro)     : {rec:.4f}")
    print(f"  F1 (macro)         : {f1m:.4f}")
    print(f"  F1 Away            : {f1_each[0]:.4f}")
    print(f"  F1 Home            : {f1_each[1]:.4f}")
    print(f"  F1 Draw            : {f1_each[2]:.4f}")
    print(f"  Log Loss           : {ll:.4f}")
    print("  =====================================================")

    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>10} {'Away':>8} {'Home':>8} {'Draw':>8}")
    for i in range(3):
        print(f"  {class_names[i]:>10} "
              f"{cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")

    print("\n  Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Away Win (0)", "Home Win (1)", "Draw (2)"],
        digits=4,
    ))

    # Comparatie cu v3 original
    with open(p("results_v3.json")) as f:
        v3_old = json.load(f)
    v3m = v3_old["test_metrics"]
    print("\n  Comparatie v3 (3 bases) vs v3+Poisson (4 bases):")
    print(f"  {'Metric':<14} {'v3':>10} {'v3+Pois':>10} {'delta':>10}")
    print(f"  {'Accuracy':<14} {v3m['accuracy']*100:>9.2f}% {acc*100:>9.2f}% "
          f"{(acc-v3m['accuracy'])*100:>+9.2f}pp")
    print(f"  {'F1 macro':<14} {v3m['f1_macro']:>10.4f} {f1m:>10.4f} "
          f"{f1m-v3m['f1_macro']:>+10.4f}")
    print(f"  {'F1 Draw':<14} {v3m['f1_per_class']['Draw']:>10.4f} "
          f"{f1_each[2]:>10.4f} "
          f"{f1_each[2]-v3m['f1_per_class']['Draw']:>+10.4f}")

    # Salvam
    print("\n  Saving artifacts...")
    joblib.dump(stacked, p("model_v3_poisson.pkl"))
    print("  model_v3_poisson.pkl saved")

    results = {
        "model": "Stacked Ensemble v3+Poisson (XGB+LGB+CB+Poisson+LR meta)",
        "version": "v3_poisson",
        "n_classes": 3,
        "base_learners": ["xgb", "lgb", "cat", "poisson"],
        "best_params_xgb_lgb_cat": best_params,
        "poisson_params": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                           "objective": "count:poisson"},
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1m, 4),
            "f1_per_class": {class_names[i]: round(float(f1_each[i]), 4) for i in range(3)},
            "log_loss": round(ll, 4),
            "confusion_matrix": cm.tolist(),
            "test_samples": int(len(y_test)),
        },
        "comparison_to_v3": {
            "accuracy_delta_pp": round((acc - v3m["accuracy"]) * 100, 2),
            "f1_macro_delta": round(f1m - v3m["f1_macro"], 4),
            "f1_draw_delta": round(f1_each[2] - v3m["f1_per_class"]["Draw"], 4),
        },
    }
    with open(p("results_v3_poisson.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  results_v3_poisson.json saved")

    print("\n" + "=" * 60)
    print("  v3 + POISSON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
