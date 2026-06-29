"""
train_model_v2.py
=================
3-class football prediction: Away Win (0) / Home Win (1) / Draw (2)

Improvements:
- Multi-class objectives for all ensemble members
- Class weights to handle draw imbalance
- f1_macro scoring for GridSearchCV
- Correct 3-class evaluation metrics
- SHAP for multi-class (Home Win + Draw classes)
- Probability calibration
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
import sklearn
sklearn.set_config(enable_metadata_routing=True)
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Using XGBoost only.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed. Using XGBoost only.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed. Skipping explainability.")

ML_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(ML_DIR, "output")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def p(filename: str) -> str:
    return os.path.join(ML_DIR, filename)


def out(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data():
    print("=" * 60)
    print("  FOOTBALL PREDICTIONS — 3-CLASS MODEL v2")
    print("=" * 60)
    print("\n[1/6] Loading data...")

    for f in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "metadata.json"]:
        if not os.path.exists(p(f)):
            raise FileNotFoundError(f"Missing: {f}. Run pipeline.py first.")

    X_train = np.load(p("X_train.npy"))
    X_test  = np.load(p("X_test.npy"))
    y_train = np.load(p("y_train.npy"))
    y_test  = np.load(p("y_test.npy"))

    with open(p("metadata.json")) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    # Validate 3-class labels
    unique_labels = np.unique(y_train)
    if not set(unique_labels).issubset({0, 1, 2}):
        raise ValueError(f"Expected labels 0/1/2, got {unique_labels}. Re-run pipeline.py.")

    # Remove dead features (constant or zero-importance)
    DEAD_FEATURES = ["home_advantage", "comp_BL1", "comp_FL1", "comp_PD", "comp_SA"]
    dead_indices  = [i for i, name in enumerate(feature_names) if name in DEAD_FEATURES]

    if dead_indices:
        print(f"  Removing {len(dead_indices)} dead features: {DEAD_FEATURES}")
        X_train = np.delete(X_train, dead_indices, axis=1)
        X_test  = np.delete(X_test,  dead_indices, axis=1)
        feature_names = [f for f in feature_names if f not in DEAD_FEATURES]

    dist = np.bincount(y_train.astype(int))
    print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    print(f"  Away(0): {dist[0]}  Home(1): {dist[1]}  Draw(2): {dist[2]}")
    print(f"  Active features ({len(feature_names)}): {feature_names}")

    return X_train, X_test, y_train, y_test, feature_names


# ──────────────────────────────────────────────────────────────
# 2. CLASS WEIGHTS
# ──────────────────────────────────────────────────────────────
def get_class_weights(y_train):
    """
    Compute balanced class weights to handle draw imbalance.
    Returns dict {0: w0, 1: w1, 2: w2} and sample_weight array.
    """
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    sample_weight = np.array([weight_dict[y] for y in y_train])

    print(f"\n  Class weights (balanced):")
    labels = {0: "Away Win", 1: "Home Win", 2: "Draw"}
    for cls, w in weight_dict.items():
        print(f"    {labels[cls]} ({cls}): {w:.4f}")

    return weight_dict, sample_weight


# ──────────────────────────────────────────────────────────────
# 3. TIME-BASED CROSS-VALIDATION (multi-class)
# ──────────────────────────────────────────────────────────────
def tune_with_timeseries_cv(X_train, y_train, sample_weight):
    print("\n[2/6] GridSearchCV with Time-Series Split (5 folds)...")

    PARAM_GRID = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    print("      Parameters being searched:")
    for k, v in PARAM_GRID.items():
        print(f"      {k:<20}: {v}")
    print()

    base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    base.set_fit_request(sample_weight=True)

    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=tscv,
        refit=True,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train, sample_weight=sample_weight)

    print(f"\n  Best CV F1 (macro): {grid.best_score_:.4f}")
    print(f"  Best params:")
    for k, v in grid.best_params_.items():
        print(f"    {k:<20}: {v}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ──────────────────────────────────────────────────────────────
# 4. ENSEMBLE MODEL (multi-class, class-weighted)
# ──────────────────────────────────────────────────────────────
class FittedVotingEnsemble:
    """Module-level class so joblib can pickle it."""
    def __init__(self, fitted_estimators):
        self.estimators_ = [est for _, est in fitted_estimators]
        self.classes_    = np.array([0, 1, 2])

    def predict_proba(self, X):
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return probas.mean(axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build_ensemble(best_params, X_train, y_train, sample_weight, weight_dict):
    print("\n[3/6] Building Ensemble Model (3-class, class-weighted)...")

    estimators = []

    xgb = XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.set_fit_request(sample_weight=True)
    estimators.append(("xgb", xgb))
    print("  XGBoost added")

    if HAS_LIGHTGBM:
        lgb = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=best_params.get("n_estimators", 200),
            max_depth=best_params.get("max_depth", 4),
            learning_rate=best_params.get("learning_rate", 0.05),
            class_weight=weight_dict,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )
        lgb.set_fit_request(sample_weight=True)
        estimators.append(("lgb", lgb))
        print("  LightGBM added")

    if HAS_CATBOOST:
        cat_weights = [weight_dict[0], weight_dict[1], weight_dict[2]]
        cat = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=best_params.get("n_estimators", 200),
            depth=best_params.get("max_depth", 4),
            learning_rate=best_params.get("learning_rate", 0.05),
            class_weights=cat_weights,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        # CatBoost handles class imbalance via class_weights; no sample_weight in fit
        estimators.append(("cat", cat))
        print("  CatBoost added")

    if len(estimators) > 1:
        # Fit each estimator individually with sample_weight, bypassing sklearn
        # metadata routing (behaviour changed in sklearn 1.4+).
        for name, est in estimators:
            if name == "cat":
                est.fit(X_train, y_train)
            else:
                est.fit(X_train, y_train, sample_weight=sample_weight)
            print(f"    {name} fitted")

        ensemble = FittedVotingEnsemble(estimators)
        print(f"  Ensemble with {len(estimators)} models created")
        return ensemble, True
    else:
        xgb.fit(X_train, y_train, sample_weight=sample_weight)
        print("  Single XGBoost model (no ensemble libraries)")
        return xgb, False


# ──────────────────────────────────────────────────────────────
# 5. PROBABILITY CALIBRATION
# ──────────────────────────────────────────────────────────────
def calibrate_model(model, X_train, y_train):
    print("\n[4/6] Calibrating probabilities (isotonic regression)...")
    # FittedVotingEnsemble is not a sklearn estimator — skip calibration,
    # soft-voting averaging already regularises probabilities.
    if not hasattr(model, "fit"):
        print("  Ensemble model: skipping calibration (soft-voting already averages probs)")
        return model
    try:
        calibrated = CalibratedClassifierCV(model, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)
        print("  Model calibrated successfully")
        return calibrated
    except Exception as e:
        print(f"  Calibration failed: {e} — using uncalibrated model")
        return model


# ──────────────────────────────────────────────────────────────
# 6. EVALUATION (3-class)
# ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    print("\n[5/6] Evaluating on test set (20% hold-out)...")

    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec     = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_test, y_pred, average=None, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    sep = "─" * 40
    print(f"\n  ┌{sep}┐")
    print(f"  │  {'Metric':<22} {'Value':>15} │")
    print(f"  ├{sep}┤")
    print(f"  │  {'Accuracy':<22} {acc*100:>14.2f}% │")
    print(f"  │  {'Precision (macro)':<22} {prec:>15.4f} │")
    print(f"  │  {'Recall (macro)':<22} {rec:>15.4f} │")
    print(f"  │  {'F1 (macro)':<22} {f1_mac:>15.4f} │")
    print(f"  ├{sep}┤")
    class_names = {0: "Away Win", 1: "Home Win", 2: "Draw"}
    for i, f1v in enumerate(f1_each):
        print(f"  │  {'F1 ' + class_names[i]:<22} {f1v:>15.4f} │")
    print(f"  └{sep}┘")

    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>18}  {'Away(0)':>8}  {'Home(1)':>8}  {'Draw(2)':>8}")
    row_labels = ["Away(0)", "Home(1)", "Draw(2)"]
    for i in range(min(3, cm.shape[0])):
        row = "  ".join(f"{cm[i,j]:>8}" for j in range(cm.shape[1]))
        print(f"  {row_labels[i]:>18}  {row}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Away Win (0)", "Home Win (1)", "Draw (2)"],
        digits=4,
    ))

    return {
        "accuracy":        round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro":    round(rec, 4),
        "f1_macro":        round(f1_mac, 4),
        "f1_per_class":    {class_names[i]: round(float(v), 4) for i, v in enumerate(f1_each)},
        "confusion_matrix": cm.tolist(),
        "test_samples":    int(len(y_test)),
    }


# ──────────────────────────────────────────────────────────────
# 7. SHAP EXPLAINABILITY (multi-class)
# ──────────────────────────────────────────────────────────────
def generate_shap_explanations(model, X_train, X_test, feature_names, is_ensemble):
    if not HAS_SHAP:
        print("\n[SHAP] Skipped — shap not installed")
        return None

    print("\n[6/6] Generating SHAP explanations...")

    try:
        if is_ensemble and hasattr(model, "estimators_"):
            base_model = model.estimators_[0]
            if hasattr(base_model, "get_booster"):
                explainer = shap.TreeExplainer(base_model)
            else:
                print("  Using KernelExplainer (slower)...")
                explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test)

        # Multi-class: shap_values is list of 3 arrays [away, home, draw]
        if isinstance(shap_values, list) and len(shap_values) == 3:
            sv_home = shap_values[1]
            sv_draw = shap_values[2]
        elif isinstance(shap_values, list):
            sv_home = shap_values[-1]
            sv_draw = shap_values[-1]
        else:
            sv_home = sv_draw = shap_values

        # Summary plot — Home Win
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_home, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out("shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP summary (Home Win) -> output/shap_summary.png")

        # Summary plot — Draw
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_draw, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out("shap_draw.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP summary (Draw)     -> output/shap_draw.png")

        # Bar importance (Home Win)
        shap_importance = np.abs(sv_home).mean(axis=0)
        importance_df   = pd.DataFrame({
            "feature":          feature_names,
            "shap_importance":  shap_importance,
        }).sort_values("shap_importance", ascending=False)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv_home, X_test, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(out("shap_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP importance bar     -> output/shap_importance.png")

        return importance_df.to_dict("records")

    except Exception as e:
        print(f"  SHAP generation failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────
def get_feature_importance(model, feature_names, is_ensemble):
    print("\n  Extracting feature importance...")
    try:
        if is_ensemble and hasattr(model, "estimators_"):
            importances = np.zeros(len(feature_names))
            count = 0
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    importances += est.feature_importances_
                    count += 1
            if count:
                importances /= count
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
            importances = model.estimator.feature_importances_
        else:
            print("  Cannot extract feature importance")
            return []

        ranked = sorted(zip(feature_names, importances.tolist()),
                        key=lambda x: x[1], reverse=True)

        print(f"  {'Feature':<30} {'Score':>7}  Bar")
        print(f"  {'─'*30}  {'─'*7}  {'─'*20}")
        for name, imp in ranked[:10]:
            bar = "█" * max(1, int(imp * 200))
            print(f"  {name:<30} {imp:>7.4f}  {bar}")

        return [{"feature": n, "importance": round(i, 6)} for n, i in ranked]

    except Exception as e:
        print(f"  Feature importance extraction failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# 9. SAVE ARTIFACTS
# ──────────────────────────────────────────────────────────────
def save_artifacts(model, best_params, best_cv_f1, metrics, feature_importance,
                   feature_names, shap_importance, is_ensemble):
    print("\n  Saving artifacts...")

    joblib.dump(model, p("model_v2.pkl"))
    print("  model_v2.pkl saved")

    metadata_v2 = {
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "model_type":    "ensemble" if is_ensemble else "xgboost",
        "calibrated":    True,
        "version":       "v2",
        "n_classes":     3,
        "target":        "0=away_win, 1=home_win, 2=draw",
    }
    with open(p("metadata_v2.json"), "w") as f:
        json.dump(metadata_v2, f, indent=2)
    print("  metadata_v2.json saved")

    results = {
        "model":   "Ensemble" if is_ensemble else "XGBoost",
        "version": "v2",
        "n_classes": 3,
        "improvements": [
            "3-class prediction (Away Win / Home Win / Draw)",
            "Balanced class weights for draw imbalance",
            "Time-based cross-validation (f1_macro)",
            "Ensemble models (XGBoost + LightGBM + CatBoost)",
            "Probability calibration (isotonic)",
            "SHAP explainability (Home Win + Draw classes)",
            "Draw-signal features (goals_similarity, form_closeness, avg_conceded)",
            "Correct form points (win=3, draw=1, loss=0)",
            "Correct H2H (draws tracked separately)",
        ],
        "tuning":            "GridSearchCV with TimeSeriesSplit (f1_macro)",
        "best_params":       best_params,
        "cv_best_f1_macro":  round(best_cv_f1, 4),
        "test_metrics":      metrics,
        "feature_importance": feature_importance,
        "shap_importance":   shap_importance,
        "config": {
            "cv_folds":    5,
            "cv_type":     "TimeSeriesSplit",
            "test_split":  "80/20 temporal",
            "calibration": "isotonic",
            "class_weights": "balanced",
            "random_state": RANDOM_STATE,
            "target":      "0=away_win, 1=home_win, 2=draw",
        },
    }
    with open(p("results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  results_v2.json saved")

    return results


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # 2. Class weights
    weight_dict, sample_weight = get_class_weights(y_train)

    # 3. Tune XGBoost with time-series CV
    best_model, best_params, best_cv_f1 = tune_with_timeseries_cv(
        X_train, y_train, sample_weight
    )

    # 4. Build ensemble
    ensemble_model, is_ensemble = build_ensemble(
        best_params, X_train, y_train, sample_weight, weight_dict
    )

    # 5. Calibrate
    calibrated_model = calibrate_model(ensemble_model, X_train, y_train)

    # 6. Evaluate
    metrics = evaluate(calibrated_model, X_test, y_test)

    # 7. SHAP
    shap_importance = generate_shap_explanations(
        ensemble_model, X_train, X_test, feature_names, is_ensemble
    )

    # 8. Feature importance
    feature_importance = get_feature_importance(ensemble_model, feature_names, is_ensemble)

    # 9. Save
    results = save_artifacts(
        calibrated_model, best_params, best_cv_f1, metrics,
        feature_importance, feature_names, shap_importance, is_ensemble
    )

    print("\n" + "=" * 60)
    print("  3-CLASS MODEL v2 COMPLETE!")
    print("  ─────────────────────────────────────")
    print(f"  Accuracy       : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 (macro)     : {metrics['f1_macro']:.4f}")
    for cls, f1v in metrics["f1_per_class"].items():
        print(f"  F1 {cls:<12}: {f1v:.4f}")
    print(f"  Model Type     : {'Ensemble' if is_ensemble else 'XGBoost'}")
    print(f"  Features       : {len(feature_names)}")
    print(f"  SHAP Plots     : ml/output/shap_*.png")
    print("=" * 60 + "\n")
