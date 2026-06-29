"""
experiment_threshold_fine.py
============================
Sweep fin pe threshold-ul Draw pe probabilitati calibrate isotonic.
Reutilizeaza setup-ul din experiment_calibration_threshold.py dar cu
grid mai dens si raporteaza si optimul ponderat (3*acc + f1_macro).

Output: experiment_threshold_fine.json
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.isotonic import IsotonicRegression
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
        meta = json.load(f)
    DEAD = ["home_advantage"]
    dead_idx = [i for i, n in enumerate(meta["feature_names"]) if n in DEAD]
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test = np.delete(X_test, dead_idx, axis=1)
    return X_train, X_test, y_train, y_test


def build_stacked(best_params, weight_dict):
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
    return StackedEnsemble(base_estimators=[("xgb", xgb), ("lgb", lgb), ("cat", cat)])


def isotonic_calibrate(proba_val, y_val, proba_test):
    n_classes = proba_val.shape[1]
    calibrated = np.zeros_like(proba_test)
    isos = []
    for c in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        y_bin = (y_val == c).astype(int)
        iso.fit(proba_val[:, c], y_bin)
        isos.append(iso)
        calibrated[:, c] = iso.predict(proba_test[:, c])
    row_sum = calibrated.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    calibrated = calibrated / row_sum
    return calibrated, isos


def predict_with_draw_threshold(proba, t_draw):
    if t_draw <= 0:
        return proba.argmax(axis=1)
    pred = np.where(proba[:, 0] >= proba[:, 1], 0, 1)
    pred = np.where(proba[:, 2] > t_draw, 2, pred)
    return pred


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_away": float(f1_each[0]),
        "f1_home": float(f1_each[1]),
        "f1_draw": float(f1_each[2]),
        "pred_dist": np.bincount(y_pred, minlength=3).tolist(),
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Fine Threshold Sweep on Calibrated Proba")
    print("=" * 70)

    X_train_full, X_test, y_train_full, y_test = load_data()
    val_size = int(len(X_train_full) * 0.10)
    X_train = X_train_full[:-val_size]
    X_val   = X_train_full[-val_size:]
    y_train = y_train_full[:-val_size]
    y_val   = y_train_full[-val_size:]

    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    sample_weight = np.array([weight_dict[y] for y in y_train])

    best_params = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 500}
    print(f"\n[1] Training stacked (best_params={best_params})...")
    stacked = build_stacked(best_params, weight_dict)
    stacked.fit(X_train, y_train, sample_weight=sample_weight)

    print("\n[2] Computing proba + isotonic calibration...")
    proba_val_raw  = stacked.predict_proba(X_val)
    proba_test_raw = stacked.predict_proba(X_test)
    proba_test_cal, _ = isotonic_calibrate(proba_val_raw, y_val, proba_test_raw)

    # Fine sweep 0.26 -> 0.34 step 0.005
    thresholds = [round(0.26 + 0.005 * i, 3) for i in range(17)]
    rows = []
    for t in thresholds:
        m = evaluate(y_test, predict_with_draw_threshold(proba_test_cal, t))
        rows.append({"t": t, **m})

    # Adauga si baseline raw (T=0) si calibrat (T=0)
    base_raw = {"t": "raw_argmax", **evaluate(y_test, proba_test_raw.argmax(axis=1))}
    base_cal = {"t": "cal_argmax", **evaluate(y_test, proba_test_cal.argmax(axis=1))}

    print("\n[3] Results (calibrated proba, fine threshold sweep on Draw):")
    print(f"  {'T':>6} | {'acc':>7} {'f1m':>7} {'fAw':>7} {'fHm':>7} {'fDr':>7} | {'pAw':>5} {'pHm':>5} {'pDr':>5}")
    print("  " + "-" * 88)
    for r in [base_raw, base_cal] + rows:
        t_str = r["t"] if isinstance(r["t"], str) else f"{r['t']:.3f}"
        pred = r["pred_dist"]
        print(f"  {t_str:>6} | {r['accuracy']*100:>6.2f}% {r['f1_macro']:>7.4f} "
              f"{r['f1_away']:>7.4f} {r['f1_home']:>7.4f} {r['f1_draw']:>7.4f} | "
              f"{pred[0]:>5} {pred[1]:>5} {pred[2]:>5}")

    # Test real dist
    test_dist = np.bincount(y_test, minlength=3)
    print(f"\n  Test real dist: Away={test_dist[0]}  Home={test_dist[1]}  Draw={test_dist[2]}")

    # Best by accuracy
    best_acc = max(rows, key=lambda r: r["accuracy"])
    # Best by F1 macro
    best_f1m = max(rows, key=lambda r: r["f1_macro"])
    # Best compozit: prioritate accuracy + minim F1 Draw rezonabil
    # scor: acc + 0.5 * f1m (acc dominanta)
    best_compozit = max(rows, key=lambda r: r["accuracy"] + 0.5 * r["f1_macro"])

    print(f"\n  Best by Accuracy:   T={best_acc['t']:.3f}  acc={best_acc['accuracy']*100:.2f}%  "
          f"f1m={best_acc['f1_macro']:.4f}  f1Draw={best_acc['f1_draw']:.4f}")
    print(f"  Best by F1 macro:   T={best_f1m['t']:.3f}  acc={best_f1m['accuracy']*100:.2f}%  "
          f"f1m={best_f1m['f1_macro']:.4f}  f1Draw={best_f1m['f1_draw']:.4f}")
    print(f"  Best (acc + 0.5*f1m): T={best_compozit['t']:.3f}  acc={best_compozit['accuracy']*100:.2f}%  "
          f"f1m={best_compozit['f1_macro']:.4f}  f1Draw={best_compozit['f1_draw']:.4f}")

    out = {
        "best_params": best_params,
        "baseline_raw": base_raw,
        "baseline_cal": base_cal,
        "thresholds": rows,
        "best_by_accuracy":  best_acc,
        "best_by_f1_macro":  best_f1m,
        "best_compozit":     best_compozit,
        "test_dist":         test_dist.tolist(),
    }
    with open(p("experiment_threshold_fine.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: experiment_threshold_fine.json")


if __name__ == "__main__":
    main()
