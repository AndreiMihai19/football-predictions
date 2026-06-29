"""
experiment_feature_selection.py
================================
Testeaza eliminarea features cu importanta foarte mica.
Foloseste best_params + balanced weights din v3.
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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
    feature_names = [n for n in full_feature_names if n not in DEAD]
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test = np.delete(X_test, dead_idx, axis=1)

    return X_train, X_test, y_train, y_test, feature_names


def train_and_eval(X_train, y_train, X_test, y_test, label, best_params):
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
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

    return {
        "label": label,
        "n_features": X_train.shape[1],
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_away": f1_each[0],
        "f1_home": f1_each[1],
        "f1_draw": f1_each[2],
    }


def remove_features(X_train, X_test, feature_names, to_remove):
    keep_idx = [i for i, n in enumerate(feature_names) if n not in to_remove]
    return (
        X_train[:, keep_idx],
        X_test[:, keep_idx],
        [feature_names[i] for i in keep_idx],
    )


def main():
    print("=" * 70)
    print("  EXPERIMENT: Feature Selection")
    print("=" * 70)

    print("\n[1] Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    best_params = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 500}

    # Schemele de testat
    schemes = [
        ("Baseline (40 features)", []),
        ("Drop 3 useless",       ["comp_BL1", "away_form_pts_away", "home_form_pts_home"]),
        ("Drop all comp_*",      ["comp_BL1", "comp_PL", "comp_PD", "comp_FL1", "comp_SA"]),
        ("Drop comp_* + 2 form", ["comp_BL1", "comp_PL", "comp_PD", "comp_FL1", "comp_SA",
                                   "away_form_pts_away", "home_form_pts_home"]),
        ("Drop bottom 10",       ["comp_BL1", "away_form_pts_away", "home_form_pts_home",
                                   "comp_PL", "comp_PD", "comp_FL1", "comp_SA",
                                   "h2h_away_wins", "away_rank", "h2h_home_wins"]),
    ]

    results = []
    for i, (name, to_remove) in enumerate(schemes, 1):
        print(f"\n[{i+1}] Testing: {name}")
        if to_remove:
            print(f"     Removing: {to_remove}")
            Xtr, Xte, fn = remove_features(X_train, X_test, feature_names, to_remove)
        else:
            Xtr, Xte, fn = X_train, X_test, feature_names
        print(f"     Features remaining: {len(fn)}")

        try:
            r = train_and_eval(Xtr, y_train, Xte, y_test, name, best_params)
            results.append(r)
            print(f"     Accuracy: {r['accuracy']*100:.2f}%  F1 macro: {r['f1_macro']:.4f}  F1 Draw: {r['f1_draw']:.4f}")
        except Exception as e:
            print(f"     FAILED: {e}")

    print("\n" + "=" * 70)
    print("  REZULTATE COMPARATIVE")
    print("=" * 70)
    print(f"  {'Scheme':<28} {'#Feat':>6} {'Accuracy':>10} {'F1 macro':>10} {'F1 Draw':>10}")
    print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['label']:<28} {r['n_features']:>6} {r['accuracy']*100:>9.2f}% "
              f"{r['f1_macro']:>10.4f} {r['f1_draw']:>10.4f}")

    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1_macro'])
    print(f"\n  Best by Accuracy: {best_acc['label']} ({best_acc['accuracy']*100:.2f}%)")
    print(f"  Best by F1 macro: {best_f1['label']} ({best_f1['f1_macro']:.4f})")

    out = {
        "schemes": [
            {
                "label": r["label"],
                "n_features": r["n_features"],
                "accuracy": round(r["accuracy"], 4),
                "f1_macro": round(r["f1_macro"], 4),
                "f1_away": round(r["f1_away"], 4),
                "f1_home": round(r["f1_home"], 4),
                "f1_draw": round(r["f1_draw"], 4),
            } for r in results
        ],
    }
    with open(p("experiment_feature_selection.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: experiment_feature_selection.json")


if __name__ == "__main__":
    main()
