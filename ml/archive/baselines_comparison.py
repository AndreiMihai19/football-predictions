"""
baselines_comparison.py
=======================
Comparatia modelului v3 cu baselines academice clasice.

Justificare academica:
- Pentru a demonstra ca modelul aduce valoare reala
- Necesita comparatie cu modele simple (lower bound)
- Standard in literatura ML aplicata in sport

Baselines evaluate (toate pe ACELASI test set ca v3):
1. Random Uniform (1/3 fiecare clasa)
2. Always Home Win (clasa cea mai frecventa)
3. Frequency-based prior (din distributia train)
4. Naive Bayes Gaussian
5. Logistic Regression simpla (no ensemble)
6. Single XGBoost (no stacking)
7. v3 Stacked Ensemble (modelul nostru)
"""

import os
import json
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, log_loss
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from stacked_ensemble import StackedEnsemble
from calibration_analysis import multiclass_brier

ML_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ML_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    X_train = np.load(os.path.join(ML_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(ML_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(ML_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(ML_DIR, "y_test.npy"))

    # Eliminam dead features
    with open(os.path.join(ML_DIR, "metadata.json")) as f:
        full_meta = json.load(f)
    dead_idx = [i for i, n in enumerate(full_meta["feature_names"]) if n == "home_advantage"]
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test = np.delete(X_test, dead_idx, axis=1)

    return X_train, X_test, y_train.astype(int), y_test.astype(int)


def evaluate_model(name, y_true, y_pred, y_proba=None):
    """Calculeaza metrici standard pentru un baseline/model."""
    metrics = {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    f1_each = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    metrics["f1_away"] = float(f1_each[0])
    metrics["f1_home"] = float(f1_each[1])
    metrics["f1_draw"] = float(f1_each[2])

    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
            metrics["brier"] = multiclass_brier(y_true, y_proba)
        except Exception:
            metrics["log_loss"] = None
            metrics["brier"] = None
    else:
        metrics["log_loss"] = None
        metrics["brier"] = None

    return metrics


# ============================================================
# BASELINES
# ============================================================

def baseline_random(y_train, y_test, rng_seed=42):
    """1: Random uniform — 1/3 sansa fiecare clasa."""
    rng = np.random.RandomState(rng_seed)
    y_pred = rng.randint(0, 3, size=len(y_test))
    y_proba = np.full((len(y_test), 3), 1 / 3)
    return evaluate_model("1. Random Uniform", y_test, y_pred, y_proba)


def baseline_always_home(y_train, y_test):
    """2: Always Home Win (clasa majoritara in fotbal)."""
    y_pred = np.full(len(y_test), 1)  # Home Win
    # Probabilitati: tot pe Home Win
    y_proba = np.zeros((len(y_test), 3))
    y_proba[:, 1] = 1.0
    return evaluate_model("2. Always Home Win", y_test, y_pred, y_proba)


def baseline_prior(y_train, y_test):
    """3: Frequency-based prior — preziciem mereu clasa majoritara, dar cu probabilitati = frecventele din train."""
    counts = np.bincount(y_train, minlength=3)
    prior = counts / counts.sum()
    y_pred = np.full(len(y_test), int(np.argmax(prior)))
    y_proba = np.tile(prior, (len(y_test), 1))
    return evaluate_model("3. Frequency Prior", y_test, y_pred, y_proba)


def baseline_naive_bayes(X_train, y_train, X_test, y_test):
    """4: Naive Bayes Gaussian — model probabilistic clasic."""
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)
    return evaluate_model("4. Naive Bayes (Gaussian)", y_test, y_pred, y_proba)


def baseline_logistic(X_train, y_train, X_test, y_test):
    """5: Logistic Regression simpla (multinomial)."""
    lr = LogisticRegression(
        solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)
    return evaluate_model("5. Logistic Regression", y_test, y_pred, y_proba)


def baseline_single_xgb(X_train, y_train, X_test, y_test):
    """6: Single XGBoost (fara stacking, fara LightGBM/CatBoost)."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weight = np.array([weights[y] for y in y_train])

    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=200, max_depth=4, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=-1,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)
    return evaluate_model("6. Single XGBoost", y_test, y_pred, y_proba)


def baseline_v3(X_test, y_test):
    """7: v3 Stacked Ensemble (modelul nostru)."""
    model = joblib.load(os.path.join(ML_DIR, "model_v3.pkl"))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluate_model("7. v3 Stacked Ensemble (NOSTRU)", y_test, y_pred, y_proba)


def print_table(results):
    """Tabel formatat ASCII pentru consola si dizertatie."""
    print("\n" + "=" * 100)
    print(f"{'Model':<35} {'Acc':>7} {'F1mac':>7} {'F1home':>7} {'F1away':>7} {'F1draw':>7} {'Brier':>7} {'LogL':>7}")
    print("-" * 100)
    for r in results:
        brier_str = f"{r['brier']:.4f}" if r['brier'] is not None else "  N/A "
        ll_str = f"{r['log_loss']:.4f}" if r['log_loss'] is not None else "  N/A "
        print(
            f"{r['name']:<35} "
            f"{r['accuracy']*100:>6.2f}% "
            f"{r['f1_macro']:>7.4f} "
            f"{r['f1_home']:>7.4f} "
            f"{r['f1_away']:>7.4f} "
            f"{r['f1_draw']:>7.4f} "
            f"{brier_str:>7} "
            f"{ll_str:>7}"
        )
    print("=" * 100)


def main():
    print("=" * 60)
    print("  BASELINES COMPARISON — Football v3")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    results = []

    print("\n[2/7] Baseline 1: Random Uniform...")
    results.append(baseline_random(y_train, y_test))

    print("[3/7] Baseline 2: Always Home Win...")
    results.append(baseline_always_home(y_train, y_test))

    print("[4/7] Baseline 3: Frequency Prior...")
    results.append(baseline_prior(y_train, y_test))

    print("[5/7] Baseline 4: Naive Bayes...")
    results.append(baseline_naive_bayes(X_train, y_train, X_test, y_test))

    print("[6/7] Baseline 5: Logistic Regression...")
    results.append(baseline_logistic(X_train, y_train, X_test, y_test))

    print("[7/7] Baseline 6: Single XGBoost...")
    results.append(baseline_single_xgb(X_train, y_train, X_test, y_test))

    print("[FINAL] Loading v3 Stacked Ensemble...")
    results.append(baseline_v3(X_test, y_test))

    print_table(results)

    # Diferenta v3 vs cele mai bune baseline-uri
    v3 = results[-1]
    best_baseline = max(results[:-1], key=lambda r: r["f1_macro"])
    naive = next(r for r in results if "Always" in r["name"])

    print(f"\nIMBUNATATIRE v3 fata de:")
    print(f"  Random Uniform        :  +{(v3['accuracy']-results[0]['accuracy'])*100:+.2f}pp acc, +{v3['f1_macro']-results[0]['f1_macro']:+.4f} F1")
    print(f"  Always Home Win       :  {(v3['accuracy']-naive['accuracy'])*100:+.2f}pp acc, {v3['f1_macro']-naive['f1_macro']:+.4f} F1")
    print(f"  Best baseline ({best_baseline['name'][:20]:<20}): {(v3['accuracy']-best_baseline['accuracy'])*100:+.2f}pp acc, {v3['f1_macro']-best_baseline['f1_macro']:+.4f} F1")

    # Salvam JSON
    out_path = os.path.join(OUTPUT_DIR, "baselines_comparison.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": [{k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()} for r in results],
            "v3_improvements": {
                "vs_random": round((v3['accuracy']-results[0]['accuracy'])*100, 2),
                "vs_always_home": round((v3['accuracy']-naive['accuracy'])*100, 2),
                "vs_best_baseline": round((v3['accuracy']-best_baseline['accuracy'])*100, 2),
                "best_baseline_name": best_baseline['name'],
            },
            "n_test_samples": int(len(y_test)),
        }, f, indent=2)
    print(f"\n  Salvat: {out_path}")

    print("\n" + "=" * 60)
    print("  BASELINES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
