"""Train 3-class football outcome model: stacked ensemble with isotonic calibration."""

import json
import os
import copy
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
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


def load_data():
    print("\n[1/6] Loading data...")

    for f in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "metadata.json"]:
        if not os.path.exists(p(f)):
            raise FileNotFoundError(f"Missing: {f}. Run pipeline.py first.")

    X_train = np.load(p("X_train.npy"))
    X_test  = np.load(p("X_test.npy"))
    y_train = np.load(p("y_train.npy"))
    y_test  = np.load(p("y_test.npy"))

    # X_*.npy contain the FULL feature set from metadata.json (including dead ones).
    # Dead features must be removed here so column order matches metadata_v3.json.
    with open(p("metadata.json")) as f:
        full_meta = json.load(f)
    full_feature_names = full_meta["feature_names"]

    unique_labels = np.unique(y_train)
    if not set(unique_labels).issubset({0, 1, 2}):
        raise ValueError(f"Expected labels 0/1/2, got {unique_labels}. Re-run pipeline.py.")

    DEAD_FEATURES = ["home_advantage"]
    dead_indices  = [i for i, name in enumerate(full_feature_names) if name in DEAD_FEATURES]
    feature_names = [f for f in full_feature_names if f not in DEAD_FEATURES]

    if dead_indices:
        print(f"  Removing {len(dead_indices)} dead features: {DEAD_FEATURES}")
        X_train = np.delete(X_train, dead_indices, axis=1)
        X_test  = np.delete(X_test,  dead_indices, axis=1)

    y_home_goals_train = y_away_goals_train = None
    csv_path = p("data_processed.csv")
    if os.path.exists(csv_path):
        df_proc = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
        if {"home_goals", "away_goals"}.issubset(df_proc.columns):
            split_idx = len(y_train)
            if len(df_proc) >= split_idx + len(y_test):
                y_home_goals_train = df_proc["home_goals"].iloc[:split_idx].values.astype(np.int32)
                y_away_goals_train = df_proc["away_goals"].iloc[:split_idx].values.astype(np.int32)
                print(f"  Loaded goal targets for Poisson: home_goals (mean={y_home_goals_train.mean():.2f}), away_goals (mean={y_away_goals_train.mean():.2f})")
            else:
                print("  WARN: row count mismatch between csv and tensors, skipping Poisson")

    dist = np.bincount(y_train.astype(int))
    print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    print(f"  Away(0): {dist[0]}  Home(1): {dist[1]}  Draw(2): {dist[2]}")
    print(f"  Active features ({len(feature_names)}): {feature_names}")

    return X_train, X_test, y_train, y_test, feature_names, y_home_goals_train, y_away_goals_train


def get_class_weights(y_train):
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    sample_weight = np.array([weight_dict[y] for y in y_train])

    print(f"\n  Class weights (balanced):")
    labels = {0: "Away Win", 1: "Home Win", 2: "Draw"}
    for cls, w in weight_dict.items():
        print(f"    {labels[cls]} ({cls}): {w:.4f}")

    return weight_dict, sample_weight


def tune_with_timeseries_cv(X_train, y_train, sample_weight):
    print("\n[2/6] GridSearchCV with Time-Series Split (5 folds)...")

    PARAM_GRID = {
        "n_estimators":  [200, 300, 500],
        "max_depth":     [4, 5, 6, 7],
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


# Imported from external module so joblib pickles it as stacked_ensemble.StackedEnsemble,
# not __main__.StackedEnsemble — otherwise predict_api.py cannot unpickle the model.
from stacked_ensemble import StackedEnsemble, CalibratedStackedEnsemble


def build_ensemble(best_params, X_train, y_train, sample_weight, weight_dict,
                   y_home_goals=None, y_away_goals=None):
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
        estimators.append(("cat", cat))
        print("  CatBoost added")

    if y_home_goals is not None and y_away_goals is not None:
        from poisson_regressor import PoissonGoalsRegressor
        poisson = PoissonGoalsRegressor(
            n_estimators=best_params.get("n_estimators", 200),
            max_depth=best_params.get("max_depth", 4),
            learning_rate=best_params.get("learning_rate", 0.05),
            random_state=RANDOM_STATE,
            use_dixon_coles=True,
        )
        estimators.append(("poisson", poisson))
        print("  Poisson (count:poisson + Dixon-Coles) added")

    if len(estimators) > 1:
        stacked = StackedEnsemble(base_estimators=estimators)
        stacked.fit(X_train, y_train, sample_weight=sample_weight,
                    y_home_goals=y_home_goals, y_away_goals=y_away_goals)
        meta_name = getattr(stacked, "meta_learner_name_", "unknown")
        print(f"  Stacked ensemble with {len(estimators)} bases + {meta_name} meta-learner created")
        if hasattr(stacked, "fitted_bases_"):
            for name, est in stacked.fitted_bases_:
                if name == "poisson" and hasattr(est, "rho_"):
                    print(f"    Dixon-Coles rho estimated: {est.rho_:.4f}")
        return stacked, True
    else:
        xgb.fit(X_train, y_train, sample_weight=sample_weight)
        print("  Single XGBoost model (no ensemble libraries)")
        return xgb, False


DRAW_THRESHOLD_DEFAULT = 0.29

def calibrate_model(model, X_train, y_train, val_size_frac=0.10,
                    draw_threshold=DRAW_THRESHOLD_DEFAULT):
    print("\n[4/6] Fitting isotonic calibrators (per-class) for accuracy mode...")
    if isinstance(model, StackedEnsemble):
        val_size = max(1, int(len(X_train) * val_size_frac))
        X_val_cal = X_train[-val_size:]
        y_val_cal = y_train[-val_size:]
        wrapper = CalibratedStackedEnsemble(draw_threshold=draw_threshold)
        wrapper.fit(model, X_val_cal, y_val_cal)
        print(f"  CalibratedStackedEnsemble fitted on last {val_size} val samples (T_draw={draw_threshold})")
        return wrapper
    if not hasattr(model, "fit"):
        print("  Non-sklearn model: skipping calibration")
        return model
    try:
        calibrated = CalibratedClassifierCV(model, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)
        print("  Model calibrated successfully")
        return calibrated
    except Exception as e:
        print(f"  Calibration failed: {e} — using uncalibrated model")
        return model


def _eval_one(y_test, y_pred, mode_label):
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec     = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_test, y_pred, average=None, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    sep = "-" * 40
    print(f"\n  Mode: {mode_label}")
    print(f"  {sep}")
    print(f"  {'Accuracy':<22} {acc*100:>14.2f}%")
    print(f"  {'Precision (macro)':<22} {prec:>15.4f}")
    print(f"  {'Recall (macro)':<22} {rec:>15.4f}")
    print(f"  {'F1 (macro)':<22} {f1_mac:>15.4f}")
    print(f"  {sep}")
    class_names = {0: "Away Win", 1: "Home Win", 2: "Draw"}
    for i, f1v in enumerate(f1_each):
        print(f"  {'F1 ' + class_names[i]:<22} {f1v:>15.4f}")
    print(f"  {sep}")

    return {
        "mode":            mode_label,
        "accuracy":        round(float(acc), 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro":    round(float(rec), 4),
        "f1_macro":        round(float(f1_mac), 4),
        "f1_per_class":    {class_names[i]: round(float(v), 4) for i, v in enumerate(f1_each)},
        "confusion_matrix": cm.tolist(),
        "test_samples":    int(len(y_test)),
    }


def evaluate(model, X_test, y_test) -> dict:
    print("\n[5/6] Evaluating on test set (20% hold-out)...")

    class_names = {0: "Away Win", 1: "Home Win", 2: "Draw"}
    if isinstance(model, CalibratedStackedEnsemble):
        m_f1  = _eval_one(y_test, model.predict(X_test, mode="f1"),       "f1 (raw argmax)")
        m_acc = _eval_one(y_test, model.predict(X_test, mode="accuracy"), f"accuracy (cal + T={model.draw_threshold})")
        # Copy m_f1 instead of referencing it — embedding m_f1 inside primary["mode_metrics"]
        # would create a circular reference on JSON serialization.
        primary = dict(m_f1)
        primary["mode_metrics"] = {"f1": m_f1, "accuracy": m_acc}
        print(f"\n  -> Default mode reported: {primary['mode']}")
        print(f"     Accuracy mode also stored: {m_acc['accuracy']*100:.2f}% acc, "
              f"{m_acc['f1_macro']:.4f} f1m, {m_acc['f1_per_class']['Draw']:.4f} f1Draw")
        return primary

    y_pred = model.predict(X_test)
    return _eval_one(y_test, y_pred, "default (argmax)")


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

        if isinstance(shap_values, list) and len(shap_values) == 3:
            sv_home = shap_values[1]
            sv_draw = shap_values[2]
        elif isinstance(shap_values, list):
            sv_home = shap_values[-1]
            sv_draw = shap_values[-1]
        else:
            sv_home = sv_draw = shap_values

        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_home, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out("shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP summary (Home Win) -> output/shap_summary.png")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_draw, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out("shap_draw.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP summary (Draw)     -> output/shap_draw.png")

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
        print(f"  {'-'*30}  {'-'*7}  {'-'*20}")
        for name, imp in ranked[:10]:
            bar = "#" * max(1, int(imp * 200))
            print(f"  {name:<30} {imp:>7.4f}  {bar}")

        return [{"feature": n, "importance": round(i, 6)} for n, i in ranked]

    except Exception as e:
        print(f"  Feature importance extraction failed: {e}")
        return []


def save_artifacts(model, best_params, best_cv_f1, metrics, feature_importance,
                   feature_names, shap_importance, is_ensemble):
    print("\n  Saving artifacts...")

    joblib.dump(model, p("model_v3.pkl"))
    print("  model_v3.pkl saved")

    has_modes = isinstance(model, CalibratedStackedEnsemble)
    metadata_v3 = {
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "model_type":    "stacked_ensemble" if is_ensemble else "xgboost",
        "calibrated":    True,
        "version":       "v3",
        "n_classes":     3,
        "target":        "0=away_win, 1=home_win, 2=draw",
        "supports_modes": has_modes,
        "default_mode":  "f1" if has_modes else None,
        "draw_threshold": float(model.draw_threshold) if has_modes else None,
    }
    with open(p("metadata_v3.json"), "w") as f:
        json.dump(metadata_v3, f, indent=2)
    print("  metadata_v3.json saved")

    results = {
        "model":   "Stacked Ensemble (XGB+LGB+CB+LR meta)" if is_ensemble else "XGBoost",
        "version": "v3",
        "n_classes": 3,
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
            "calibration": "isotonic per-class on last 10% of train (val split)",
            "class_weights": "balanced",
            "random_state": RANDOM_STATE,
            "target":      "0=away_win, 1=home_win, 2=draw",
            "supports_modes": has_modes,
            "default_mode":  "f1" if has_modes else None,
            "draw_threshold": float(model.draw_threshold) if has_modes else None,
        },
    }

    inner = model.stacked_ if hasattr(model, "stacked_") else model
    if hasattr(inner, "meta_learner_name_"):
        results["meta_learner"] = {
            "selected": inner.meta_learner_name_,
            "cv_scores": inner.meta_scores_,
        }
    with open(p("results_v3.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  results_v3.json saved")

    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, y_home_goals_train, y_away_goals_train = load_data()

    weight_dict, sample_weight = get_class_weights(y_train)

    best_model, best_params, best_cv_f1 = tune_with_timeseries_cv(
        X_train, y_train, sample_weight
    )

    ensemble_model, is_ensemble = build_ensemble(
        best_params, X_train, y_train, sample_weight, weight_dict,
        y_home_goals=y_home_goals_train, y_away_goals=y_away_goals_train,
    )

    calibrated_model = calibrate_model(ensemble_model, X_train, y_train)

    metrics = evaluate(calibrated_model, X_test, y_test)

    shap_importance = generate_shap_explanations(
        ensemble_model, X_train, X_test, feature_names, is_ensemble
    )

    feature_importance = get_feature_importance(ensemble_model, feature_names, is_ensemble)

    results = save_artifacts(
        calibrated_model, best_params, best_cv_f1, metrics,
        feature_importance, feature_names, shap_importance, is_ensemble
    )

    print(f"\nTraining complete. Accuracy: {metrics['accuracy']*100:.2f}% | F1 macro: {metrics['f1_macro']:.4f}")
