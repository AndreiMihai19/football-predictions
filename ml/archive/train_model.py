"""
train_model.py
XGBoost cu GridSearchCV pe datele generate de pipeline.py.
"""

import json
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from xgboost import XGBClassifier

ML_DIR       = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


def p(filename: str) -> str:
    return os.path.join(ML_DIR, filename)


# ──────────────────────────────────────────────────────────────
# 1. INCARCARE DATE
# ──────────────────────────────────────────────────────────────
def load_data():
    print("=" * 54)
    print("  FOOTBALL PREDICTIONS — XGBoost + GridSearchCV")
    print("=" * 54)
    print("\n[1/4] Incarcare tensori si metadata...")

    for f in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "metadata.json"]:
        if not os.path.exists(p(f)):
            raise FileNotFoundError(f"Fisier lipsa: {f}. Ruleaza pipeline.py intai.")

    X_train = np.load(p("X_train.npy"))
    X_test  = np.load(p("X_test.npy"))
    y_train = np.load(p("y_train.npy"))
    y_test  = np.load(p("y_test.npy"))

    with open(p("metadata.json")) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    dist = np.bincount(y_train.astype(int))
    print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    print(f"  Away wins (0): {dist[0]}  |  Home wins (1): {dist[1]}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    return X_train, X_test, y_train, y_test, feature_names


# ──────────────────────────────────────────────────────────────
# 2. GRIDSEARCHCV
# ──────────────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
}


def tune_xgboost(X_train, y_train) -> tuple[XGBClassifier, dict, float]:
    print("\n[2/4] GridSearchCV (27 combinatii × 5-fold CV = 135 fits)...")
    print("      Parametri cautati:")
    for k, v in PARAM_GRID.items():
        print(f"      {k:<20}: {v}")
    print()

    base = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best CV F1   : {grid.best_score_:.4f}")
    print(f"  Best params  :")
    for k, v in grid.best_params_.items():
        print(f"    {k:<20}: {v}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ──────────────────────────────────────────────────────────────
# 3. EVALUARE
# ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    print("\n[3/4] Evaluare pe test set (20% hold-out)...")

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    # Metrici
    sep = "─" * 36
    print(f"\n  ┌{sep}┐")
    print(f"  │  {'Metrica':<18} {'Valoare':>15} │")
    print(f"  ├{sep}┤")
    print(f"  │  {'Accuracy':<18} {acc*100:>14.2f}% │")
    print(f"  │  {'Precision':<18} {prec:>15.4f} │")
    print(f"  │  {'Recall':<18} {rec:>15.4f} │")
    print(f"  │  {'F1 Score':<18} {f1:>15.4f} │")
    print(f"  └{sep}┘")

    # Confusion matrix
    print("\n  Confusion Matrix:")
    print(f"  {'':>18} Pred Away  Pred Home")
    print(f"  {'':>18} {'(0)':>8}    {'(1)':>8}")
    print(f"  {'Real Away (0)':>18} {cm[0,0]:>8}    {cm[0,1]:>8}")
    print(f"  {'Real Home (1)':>18} {cm[1,0]:>8}    {cm[1,1]:>8}")

    # Classification report
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Away Win (0)", "Home Win (1)"],
        digits=4,
    ))

    return {
        "accuracy":         round(acc, 4),
        "precision":        round(prec, 4),
        "recall":           round(rec, 4),
        "f1_score":         round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "test_samples":     int(len(y_test)),
    }


# ──────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────
def print_feature_importance(model: XGBClassifier, feature_names: list) -> list:
    ranked = sorted(
        zip(feature_names, model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    print("  Feature Importance — top 10:")
    print(f"  {'Feature':<28} {'Score':>7}  Bar")
    print(f"  {'─'*28}  {'─'*7}  {'─'*20}")
    for name, imp in ranked[:10]:
        bar = "█" * max(1, int(imp * 200))
        print(f"  {name:<28} {imp:>7.4f}  {bar}")

    return [{"feature": n, "importance": round(i, 6)} for n, i in ranked]


# ──────────────────────────────────────────────────────────────
# 5. LEARNING CURVE
# ──────────────────────────────────────────────────────────────
def compute_learning_curve(model: XGBClassifier, X_train, y_train) -> dict:
    print("\n  Learning Curve (10 puncte, 5-fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
    )

    result = {
        "train_sizes":       train_sizes_abs.tolist(),
        "train_scores_mean": np.mean(train_scores, axis=1).round(4).tolist(),
        "train_scores_std":  np.std(train_scores,  axis=1).round(4).tolist(),
        "val_scores_mean":   np.mean(val_scores,   axis=1).round(4).tolist(),
        "val_scores_std":    np.std(val_scores,    axis=1).round(4).tolist(),
    }

    print(f"  {'Size':>6}  {'Train Acc':>10}  {'Val Acc':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    for sz, tr, vl in zip(result["train_sizes"], result["train_scores_mean"], result["val_scores_mean"]):
        print(f"  {sz:>6}  {tr:>10.4f}  {vl:>10.4f}")

    return result


# ──────────────────────────────────────────────────────────────
# 6. SALVARE
# ──────────────────────────────────────────────────────────────
def save_artifacts(model, best_params, best_cv_f1, metrics, feature_importance):
    print("\n[4/4] Salvare artefacte...")

    # model.pkl
    joblib.dump(model, p("model.pkl"))
    print(f"  model.pkl        -> {p('model.pkl')}")

    # best_params.json
    best_params_out = {
        "params":     best_params,
        "cv_best_f1": round(best_cv_f1, 4),
        "cv_folds":   5,
        "scoring":    "f1",
    }
    with open(p("best_params.json"), "w") as f:
        json.dump(best_params_out, f, indent=2)
    print(f"  best_params.json -> {p('best_params.json')}")

    # results.json
    results = {
        "model":              "XGBoost",
        "tuning":             "GridSearchCV",
        "best_params":        best_params,
        "cv_best_f1":         round(best_cv_f1, 4),
        "test_metrics":       metrics,
        "feature_importance": feature_importance,
        "config": {
            "param_grid":    PARAM_GRID,
            "cv_folds":      5,
            "test_split":    "80/20 stratified",
            "data_scaled":   True,
            "random_state":  RANDOM_STATE,
            "target":        "1=home_win, 0=away_win",
        },
    }
    with open(p("results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  results.json     -> {p('results.json')}")


def save_learning_curve(lc: dict):
    with open(p("learning_curve.json"), "w") as f:
        json.dump(lc, f, indent=2)
    print(f"  learning_curve.json -> {p('learning_curve.json')}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_data()

    model, best_params, best_cv_f1 = tune_xgboost(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    print("\n  Feature Importance:\n")
    fi = print_feature_importance(model, feature_names)

    print("\n[5/5] Learning curve...")
    lc = compute_learning_curve(model, X_train, y_train)

    save_artifacts(model, best_params, best_cv_f1, metrics, fi)
    save_learning_curve(lc)

    print("\n" + "=" * 54)
    print(f"  Accuracy      : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score      : {metrics['f1_score']:.4f}")
    print(f"  Best params   : {best_params}")
    print(f"  Total features: {len(feature_names)}")
    print("=" * 54 + "\n")
