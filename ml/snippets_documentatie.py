# Snippet 1 — Generarea OOF predictions (stacked_ensemble.py)


def fit(self, X, y, sample_weight=None):
    n_base = len(self.base_estimators)
    oof_pred = np.zeros((len(X), 3 * n_base))
    skf = StratifiedKFold(n_splits=5, shuffle=False)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr = y[tr_idx]

        for b_idx, (name, est) in enumerate(self.base_estimators):
            est_fold = copy.deepcopy(est)
            est_fold.fit(X_tr, y_tr, sample_weight=sample_weight[tr_idx])
            col_start = b_idx * 3
            oof_pred[val_idx, col_start:col_start + 3] = est_fold.predict_proba(X_val)

    self._oof_scaler = StandardScaler()
    oof_scaled = self._oof_scaler.fit_transform(oof_pred)
    self.meta_learner_.fit(oof_scaled, y)


# Snippet 2 — Selectia automata a meta-learner-ului (stacked_ensemble.py)
def select_best_meta(oof_pred, y):
    candidates = {
        "LR":  LogisticRegression(C=1.0, class_weight="balanced"),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True),
    }
    scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, clf in candidates.items():
        cv_scores = cross_val_score(clf, oof_pred, y, cv=skf, scoring="f1_macro")
        scores[name] = float(cv_scores.mean())

    best_name = max(scores, key=scores.__getitem__)
    return candidates[best_name], best_name, scores


# Snippet 3 — Inferenta finala (stacked_ensemble.py)
def predict_proba(self, X):
    stacked = np.hstack([est.predict_proba(X) for _, est in self.fitted_bases_])
    stacked = self._oof_scaler.transform(stacked)
    return self.meta_learner_.predict_proba(stacked)
