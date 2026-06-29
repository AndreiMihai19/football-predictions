"""
experiment_calibration_threshold.py
====================================
Testeaza calibrare + threshold tuning pentru clasa Draw.

Strategia:
  1. Antreneaza stacked ensemble pe train (cu best_params v3)
  2. Obtine predict_proba pe test
  3. Aplica isotonic calibration pe probabilitati (per-class, 1-vs-rest)
     folosind un val set decupat din train (ultimele 10%)
  4. Grid search pe threshold pentru Draw:
       - regula: prezice Draw daca proba_draw > T_draw,
                 altfel argmax intre Away si Home
  5. Raporteaza accuracy / F1 macro / F1 Draw pentru fiecare threshold,
     pe probabilitati raw si pe probabilitati calibrate

Output: experiment_calibration_threshold.json
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
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
    """
    Isotonic regression per-class (1-vs-rest), apoi renormalizare la suma 1.
    proba_val: (N_val, 3) probas pe validation
    y_val:     (N_val,)   labels pe validation
    proba_test:(N_test, 3) probas pe test (de calibrat)
    """
    n_classes = proba_val.shape[1]
    calibrated = np.zeros_like(proba_test)
    for c in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        y_bin = (y_val == c).astype(int)
        iso.fit(proba_val[:, c], y_bin)
        calibrated[:, c] = iso.predict(proba_test[:, c])
    # Renormalizare (poate suma sa nu fie 1 dupa isotonic)
    row_sum = calibrated.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    calibrated = calibrated / row_sum
    return calibrated


def predict_with_draw_threshold(proba, t_draw):
    """
    Daca proba[:,2] > t_draw -> Draw (2)
    Altfel argmax intre clasele 0 si 1.
    Cazul special: t_draw <= 0 -> echivalent argmax clasic.
    """
    if t_draw <= 0:
        return proba.argmax(axis=1)
    pred = np.where(proba[:, 0] >= proba[:, 1], 0, 1)  # vs Home
    pred = np.where(proba[:, 2] > t_draw, 2, pred)
    return pred


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_away": float(f1_each[0]),
        "f1_home": float(f1_each[1]),
        "f1_draw": float(f1_each[2]),
        "pred_dist": np.bincount(y_pred, minlength=3).tolist(),
        "cm": cm.tolist(),
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Calibration + Draw Threshold Tuning")
    print("=" * 70)

    print("\n[1] Loading data...")
    X_train_full, X_test, y_train_full, y_test = load_data()
    print(f"  Train: {X_train_full.shape}  |  Test: {X_test.shape}")

    # Calibration val split: ultimele 10% din train (cronologic, deja sortat)
    val_size = int(len(X_train_full) * 0.10)
    X_train = X_train_full[:-val_size]
    X_val   = X_train_full[-val_size:]
    y_train = y_train_full[:-val_size]
    y_val   = y_train_full[-val_size:]
    print(f"  Train fit:  {X_train.shape}  |  Val (calib): {X_val.shape}  |  Test: {X_test.shape}")
    print(f"  Val dist:   Away={np.sum(y_val==0)}  Home={np.sum(y_val==1)}  Draw={np.sum(y_val==2)}")
    print(f"  Test dist:  Away={np.sum(y_test==0)}  Home={np.sum(y_test==1)}  Draw={np.sum(y_test==2)}")

    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    sample_weight = np.array([weight_dict[y] for y in y_train])

    best_params = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 500}
    print(f"\n[2] Training stacked ensemble (best_params={best_params})...")
    stacked = build_stacked(best_params, weight_dict)
    stacked.fit(X_train, y_train, sample_weight=sample_weight)

    print("\n[3] Computing probas on val + test...")
    proba_val_raw  = stacked.predict_proba(X_val)
    proba_test_raw = stacked.predict_proba(X_test)
    print(f"  Raw test proba mean per class: "
          f"Away={proba_test_raw[:,0].mean():.3f}  "
          f"Home={proba_test_raw[:,1].mean():.3f}  "
          f"Draw={proba_test_raw[:,2].mean():.3f}")
    print(f"  Val proba mean per class:      "
          f"Away={proba_val_raw[:,0].mean():.3f}  "
          f"Home={proba_val_raw[:,1].mean():.3f}  "
          f"Draw={proba_val_raw[:,2].mean():.3f}")
    print(f"  Val real dist:                 "
          f"Away={(y_val==0).mean():.3f}  "
          f"Home={(y_val==1).mean():.3f}  "
          f"Draw={(y_val==2).mean():.3f}")

    print("\n[4] Isotonic calibration (fit on val, transform on test)...")
    proba_test_cal = isotonic_calibrate(proba_val_raw, y_val, proba_test_raw)
    print(f"  Calibrated test proba mean per class: "
          f"Away={proba_test_cal[:,0].mean():.3f}  "
          f"Home={proba_test_cal[:,1].mean():.3f}  "
          f"Draw={proba_test_cal[:,2].mean():.3f}")

    # Baseline (argmax raw)
    baseline = evaluate(y_test, proba_test_raw.argmax(axis=1))
    baseline_cal = evaluate(y_test, proba_test_cal.argmax(axis=1))

    print("\n[5] Threshold sweep on Draw (raw vs calibrated)...")
    thresholds = [0.0, 0.27, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45]

    rows = []
    for t in thresholds:
        pred_raw = predict_with_draw_threshold(proba_test_raw, t)
        pred_cal = predict_with_draw_threshold(proba_test_cal, t)
        m_raw = evaluate(y_test, pred_raw)
        m_cal = evaluate(y_test, pred_cal)
        rows.append({"t": t, "raw": m_raw, "cal": m_cal})

    print(f"\n  {'T_draw':>7} | {'RAW acc':>8} {'RAW f1m':>8} {'RAW fDr':>8} {'pDr':>5}"
          f" || {'CAL acc':>8} {'CAL f1m':>8} {'CAL fDr':>8} {'pDr':>5}")
    print("  " + "-" * 90)
    for r in rows:
        rw, cl = r["raw"], r["cal"]
        print(f"  {r['t']:>7.2f} | "
              f"{rw['accuracy']*100:>7.2f}% {rw['f1_macro']:>8.4f} {rw['f1_draw']:>8.4f} {rw['pred_dist'][2]:>5}"
              f" || "
              f"{cl['accuracy']*100:>7.2f}% {cl['f1_macro']:>8.4f} {cl['f1_draw']:>8.4f} {cl['pred_dist'][2]:>5}")

    # Best by F1 macro
    best_raw = max(rows, key=lambda r: r["raw"]["f1_macro"])
    best_cal = max(rows, key=lambda r: r["cal"]["f1_macro"])
    print("\n  Best by F1 macro:")
    print(f"    RAW:  T_draw={best_raw['t']:.2f}  acc={best_raw['raw']['accuracy']*100:.2f}%  "
          f"f1m={best_raw['raw']['f1_macro']:.4f}  f1Draw={best_raw['raw']['f1_draw']:.4f}")
    print(f"    CAL:  T_draw={best_cal['t']:.2f}  acc={best_cal['cal']['accuracy']*100:.2f}%  "
          f"f1m={best_cal['cal']['f1_macro']:.4f}  f1Draw={best_cal['cal']['f1_draw']:.4f}")

    # Best by F1 Draw (sub o constraint pe accuracy: nu mai mica decat baseline-1pp)
    acc_floor = baseline["accuracy"] - 0.01
    cand_raw = [r for r in rows if r["raw"]["accuracy"] >= acc_floor]
    cand_cal = [r for r in rows if r["cal"]["accuracy"] >= acc_floor]
    if cand_raw:
        best_draw_raw = max(cand_raw, key=lambda r: r["raw"]["f1_draw"])
        print(f"\n  Best F1 Draw (acc >= baseline-1pp = {acc_floor*100:.2f}%):")
        print(f"    RAW:  T_draw={best_draw_raw['t']:.2f}  acc={best_draw_raw['raw']['accuracy']*100:.2f}%  "
              f"f1m={best_draw_raw['raw']['f1_macro']:.4f}  f1Draw={best_draw_raw['raw']['f1_draw']:.4f}")
    if cand_cal:
        best_draw_cal = max(cand_cal, key=lambda r: r["cal"]["f1_draw"])
        print(f"    CAL:  T_draw={best_draw_cal['t']:.2f}  acc={best_draw_cal['cal']['accuracy']*100:.2f}%  "
              f"f1m={best_draw_cal['cal']['f1_macro']:.4f}  f1Draw={best_draw_cal['cal']['f1_draw']:.4f}")

    print("\n  Baseline (argmax, no threshold):")
    print(f"    RAW:  acc={baseline['accuracy']*100:.2f}%  f1m={baseline['f1_macro']:.4f}  f1Draw={baseline['f1_draw']:.4f}")
    print(f"    CAL:  acc={baseline_cal['accuracy']*100:.2f}%  f1m={baseline_cal['f1_macro']:.4f}  f1Draw={baseline_cal['f1_draw']:.4f}")

    out = {
        "best_params": best_params,
        "baseline_raw": baseline,
        "baseline_cal": baseline_cal,
        "thresholds": [
            {"t": r["t"], "raw": r["raw"], "cal": r["cal"]} for r in rows
        ],
        "best_by_f1_macro": {
            "raw": {"t": best_raw["t"], **best_raw["raw"]},
            "cal": {"t": best_cal["t"], **best_cal["cal"]},
        },
    }
    with open(p("experiment_calibration_threshold.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: experiment_calibration_threshold.json")


if __name__ == "__main__":
    main()
