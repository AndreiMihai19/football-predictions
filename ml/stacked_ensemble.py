import copy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_mlp_meta(random_state=RANDOM_STATE):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=False,
    )


def select_best_meta(oof_pred, y, sample_weight=None, random_state=RANDOM_STATE):
    candidates = {
        "LR":  LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=random_state,
        ),
        "MLP": build_mlp_meta(random_state),
    }

    scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for name, clf in candidates.items():
        cv_scores = cross_val_score(clf, oof_pred, y, cv=skf,
                                    scoring="f1_macro", n_jobs=-1)
        scores[name] = float(cv_scores.mean())
        print(f"    Meta {name} CV F1 macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    best_name = max(scores, key=scores.__getitem__)
    best_meta = candidates[best_name]
    print(f"    -> Selectat: {best_name} (F1={scores[best_name]:.4f})")
    return best_meta, best_name, scores


class StackedEnsemble:
    def __init__(self, base_estimators, meta_learner=None):
        self.base_estimators = base_estimators
        self.fitted_bases_ = []
        self._meta_learner_init = meta_learner
        self.meta_learner_ = meta_learner
        self.meta_learner_name_ = "custom" if meta_learner is not None else None
        self.meta_scores_ = {}
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y, sample_weight=None, y_home_goals=None, y_away_goals=None):
        n_base = len(self.base_estimators)
        oof_pred = np.zeros((len(X), 3 * n_base))
        skf = StratifiedKFold(n_splits=5, shuffle=False)

        print(f"  Generating OOF predictions (5 folds x {n_base} bases)...")
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr        = y[tr_idx]
            sw_tr       = sample_weight[tr_idx] if sample_weight is not None else None
            yh_tr       = y_home_goals[tr_idx] if y_home_goals is not None else None
            ya_tr       = y_away_goals[tr_idx] if y_away_goals is not None else None

            for b_idx, (name, est) in enumerate(self.base_estimators):
                est_fold = copy.deepcopy(est)
                if name == "cat":
                    est_fold.fit(X_tr, y_tr)
                elif name == "poisson":
                    est_fold.fit(X_tr, y_tr,
                                 y_home_goals=yh_tr, y_away_goals=ya_tr)
                else:
                    est_fold.fit(X_tr, y_tr, sample_weight=sw_tr)
                col_start = b_idx * 3
                oof_pred[val_idx, col_start:col_start + 3] = est_fold.predict_proba(X_val)

                
            print(f"    Fold {fold_idx + 1}/5 done")

        self._oof_scaler = StandardScaler()
        oof_scaled = self._oof_scaler.fit_transform(oof_pred)

        if self._meta_learner_init is None:
            print("  Selecting best meta-learner (LR vs MLP) via 5-fold CV...")
            best_meta, best_name, scores = select_best_meta(oof_scaled, y)
            self.meta_learner_ = best_meta
            self.meta_learner_name_ = best_name
            self.meta_scores_ = scores
        else:
            oof_scaled = oof_pred

        print(f"  Training {self.meta_learner_name_} meta-learner on OOF predictions...")
        if isinstance(self.meta_learner_, MLPClassifier):
            self.meta_learner_.fit(oof_scaled, y)
        else:
            self.meta_learner_.fit(oof_scaled, y, sample_weight=sample_weight)

        print("  Refitting base models on full training set...")
        self.fitted_bases_ = []
        for name, est in self.base_estimators:
            est_full = copy.deepcopy(est)
            if name == "cat":
                est_full.fit(X, y)
            elif name == "poisson":
                est_full.fit(X, y,
                             y_home_goals=y_home_goals, y_away_goals=y_away_goals)
            else:
                est_full.fit(X, y, sample_weight=sample_weight)
            self.fitted_bases_.append((name, est_full))
            print(f"    {name} refitted")

        self.estimators_ = [est for _, est in self.fitted_bases_]
        return self

    def predict_proba(self, X):
        stacked = np.hstack([est.predict_proba(X) for _, est in self.fitted_bases_])
        if hasattr(self, "_oof_scaler"):
            stacked = self._oof_scaler.transform(stacked)
        return self.meta_learner_.predict_proba(stacked)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class CalibratedStackedEnsemble:
    def __init__(self, draw_threshold=0.29):
        self.stacked_ = None
        self.isos_ = None
        self.draw_threshold = float(draw_threshold)
        self.classes_ = np.array([0, 1, 2])

    def fit(self, stacked, X_val, y_val):
        from sklearn.isotonic import IsotonicRegression
        self.stacked_ = stacked
        proba_val = stacked.predict_proba(X_val)
        self.isos_ = []
        for c in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            y_bin = (y_val == c).astype(int)
            iso.fit(proba_val[:, c], y_bin)
            self.isos_.append(iso)
        return self

    def predict_proba(self, X, mode="f1"):
        if self.stacked_ is None:
            raise RuntimeError("CalibratedStackedEnsemble not fitted")
        proba_raw = self.stacked_.predict_proba(X)
        if mode == "f1":
            return proba_raw
        if mode != "accuracy":
            raise ValueError(f"Unknown mode: {mode}. Use 'f1' or 'accuracy'.")
        cal = np.zeros_like(proba_raw)
        for c, iso in enumerate(self.isos_):
            cal[:, c] = iso.predict(proba_raw[:, c])
        row_sum = cal.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return cal / row_sum

    def predict(self, X, mode="f1"):
        proba = self.predict_proba(X, mode=mode)
        if mode == "f1":
            return proba.argmax(axis=1)
        t = self.draw_threshold
        pred = np.where(proba[:, 0] >= proba[:, 1], 0, 1)
        pred = np.where(proba[:, 2] > t, 2, pred)
        return pred

    @property
    def estimators_(self):
        return self.stacked_.estimators_ if self.stacked_ is not None else []
