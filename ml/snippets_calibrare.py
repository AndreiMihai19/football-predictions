# Snippet 1 — Fit calibrare isotonica (stacked_ensemble.py)
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


# Snippet 2 — Aplicarea calibrarii + normalizare (stacked_ensemble.py)
def predict_proba(self, X, mode="f1"):
    proba_raw = self.stacked_.predict_proba(X)
    if mode == "f1":
        return proba_raw
    cal = np.zeros_like(proba_raw)
    for c, iso in enumerate(self.isos_):
        cal[:, c] = iso.predict(proba_raw[:, c])
    row_sum = cal.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return cal / row_sum


# Snippet 3 — Predictie cu draw threshold (stacked_ensemble.py)
def predict(self, X, mode="f1"):
    proba = self.predict_proba(X, mode=mode)
    if mode == "f1":
        return proba.argmax(axis=1)
    t = self.draw_threshold
    pred = np.where(proba[:, 0] >= proba[:, 1], 0, 1)
    pred = np.where(proba[:, 2] > t, 2, pred)
    return pred
