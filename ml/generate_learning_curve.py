"""Generate learning_curve.json from XGBoost with best_params (StratifiedKFold 5-fold)."""

import json
import os
import numpy as np
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

ML_DIR = os.path.dirname(os.path.abspath(__file__))


def p(f):
    return os.path.join(ML_DIR, f)


print("GENERATING LEARNING CURVE")

print("\n[1/3] Loading data...")
X_train = np.load(p("X_train.npy"))
y_train = np.load(p("y_train.npy"))

with open(p("metadata.json")) as f:
    full_meta = json.load(f)
full_feature_names = full_meta["feature_names"]
DEAD_FEATURES = ["home_advantage"]
dead_indices = [i for i, name in enumerate(full_feature_names) if name in DEAD_FEATURES]
if dead_indices:
    X_train = np.delete(X_train, dead_indices, axis=1)

with open(p("results_v3.json")) as f:
    results = json.load(f)

best_params = results.get("best_params", {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.01,
})
print(f"  X_train: {X_train.shape}")
print(f"  Best params: {best_params}")

classes = np.array([0, 1, 2])
weights = compute_class_weight("balanced", classes=classes, y=y_train)
weight_dict = {i: w for i, w in enumerate(weights)}
sample_weight = np.array([weight_dict[y] for y in y_train])

model = XGBClassifier(
    **best_params,
    objective="multi:softprob",
    num_class=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)

print("\n[2/3] Computing learning curve (10 points x 5 folds)...")

train_sizes_pct = np.linspace(0.05, 1.0, 10)
cv = StratifiedKFold(n_splits=5, shuffle=False)

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_train,
    y_train,
    train_sizes=train_sizes_pct,
    cv=cv,
    scoring="accuracy",
    params={"sample_weight": sample_weight},
    n_jobs=-1,
    verbose=1,
)

print("\n[3/3] Saving learning_curve.json...")

out = {
    "train_sizes": [int(x) for x in train_sizes],
    "train_scores_mean": [round(float(x), 4) for x in train_scores.mean(axis=1)],
    "train_scores_std":  [round(float(x), 4) for x in train_scores.std(axis=1)],
    "val_scores_mean":   [round(float(x), 4) for x in val_scores.mean(axis=1)],
    "val_scores_std":    [round(float(x), 4) for x in val_scores.std(axis=1)],
}

with open(p("learning_curve.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"  learning_curve.json salvat")
print(f"\n  Train sizes: {out['train_sizes']}")
print(f"  Val accuracy (final): {out['val_scores_mean'][-1]*100:.2f}%")
print("DONE")
