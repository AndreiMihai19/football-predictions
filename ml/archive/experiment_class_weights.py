"""
experiment_class_weights.py
===========================
Testeaza diferite scheme de class_weights pe modelul v3 stacked ensemble
fara a face full GridSearchCV (foloseste best_params din results_v3.json).

Output: tabel comparativ Accuracy / F1 macro / F1 Draw pentru fiecare scheme.
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score, recall_score,
)
import sklearn
sklearn.set_config(enable_metadata_routing=True)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from stacked_ensemble import StackedEnsemble

ML_DIR = os.path.dirname(os.path.abspath(__file__))
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
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test = np.delete(X_test, dead_idx, axis=1)

    return X_train, X_test, y_train, y_test


def train_and_eval(X_train, y_train, X_test, y_test, weight_dict, label, best_params):
    sample_weight = np.array([weight_dict[y] for y in y_train])

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

    estimators = [("xgb", xgb), ("lgb", lgb), ("cat", cat)]
    stacked = StackedEnsemble(base_estimators=estimators)
    stacked.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = stacked.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    pred_dist = np.bincount(y_pred, minlength=3)

    return {
        "label": label,
        "weights": weight_dict,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_away": f1_each[0],
        "f1_home": f1_each[1],
        "f1_draw": f1_each[2],
        "cm": cm,
        "pred_dist": pred_dist,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Class Weight Tuning")
    print("=" * 70)

    print("\n[1] Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    real_dist_test = np.bincount(y_test, minlength=3)
    print(f"  Test distribution: Away={real_dist_test[0]}, Home={real_dist_test[1]}, Draw={real_dist_test[2]}")

    best_params = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 500}
    print(f"  Reusing best_params from v3: {best_params}")

    # Scheme to test
    # Note: balanced este ~ {0: 1.0987, 1: 0.7435, 2: 1.3426}
    schemes = [
        ("balanced (current)", {0: 1.0987, 1: 0.7435, 2: 1.3426}),
        ("uniform",            {0: 1.0,    1: 1.0,    2: 1.0}),
        ("light_draw_boost",   {0: 1.05,   1: 0.85,   2: 1.15}),
        ("medium_draw_boost",  {0: 1.05,   1: 0.80,   2: 1.20}),
        ("draw_neutral",       {0: 1.10,   1: 0.75,   2: 1.10}),
    ]

    results = []
    for i, (name, weights) in enumerate(schemes, 1):
        print(f"\n[{i+1}] Testing scheme: {name}")
        print(f"     Weights: Away={weights[0]:.4f}, Home={weights[1]:.4f}, Draw={weights[2]:.4f}")
        try:
            r = train_and_eval(X_train, y_train, X_test, y_test, weights, name, best_params)
            results.append(r)
            print(f"     Accuracy: {r['accuracy']*100:.2f}%  F1 macro: {r['f1_macro']:.4f}  F1 Draw: {r['f1_draw']:.4f}")
            print(f"     Predicted: Away={r['pred_dist'][0]}, Home={r['pred_dist'][1]}, Draw={r['pred_dist'][2]}")
        except Exception as e:
            print(f"     FAILED: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  REZULTATE COMPARATIVE")
    print("=" * 70)
    print(f"  {'Scheme':<22} {'Accuracy':>10} {'F1 macro':>10} {'F1 Draw':>10} {'PredDraw':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['label']:<22} {r['accuracy']*100:>9.2f}% {r['f1_macro']:>10.4f} "
              f"{r['f1_draw']:>10.4f} {r['pred_dist'][2]:>10}")

    # Best by accuracy
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1_macro'])
    print(f"\n  Best by Accuracy: {best_acc['label']} ({best_acc['accuracy']*100:.2f}%)")
    print(f"  Best by F1 macro: {best_f1['label']} ({best_f1['f1_macro']:.4f})")

    # Save results
    out = {
        "schemes": [
            {
                "label": r["label"],
                "weights": r["weights"],
                "accuracy": round(r["accuracy"], 4),
                "f1_macro": round(r["f1_macro"], 4),
                "f1_away": round(r["f1_away"], 4),
                "f1_home": round(r["f1_home"], 4),
                "f1_draw": round(r["f1_draw"], 4),
                "pred_dist": r["pred_dist"].tolist(),
                "cm": r["cm"].tolist(),
            } for r in results
        ],
        "best_by_accuracy": best_acc["label"],
        "best_by_f1_macro": best_f1["label"],
    }
    with open(p("experiment_class_weights.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: experiment_class_weights.json")


if __name__ == "__main__":
    main()
