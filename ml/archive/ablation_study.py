"""
ablation_study.py
=================
Ablation study: cat contribuie fiecare grup de features la performanta modelului.

Justificare academica:
- Demonstreaza valoarea adusa de fiecare componenta
- Standard in publicatiile de ML aplicat
- Permite identificarea features importante vs zgomot

Procedura:
- Antrenam un Single XGBoost (rapid, ~30 sec/run vs 5 min pentru stacked)
- Pentru fiecare grup de features: eliminam grupul, antrenam, evaluam
- Comparam cu baseline (toate features)

Grupuri features:
1. ELO              : home_elo, away_elo, elo_diff
2. Form             : home_form_pts, gf, ga + away analoage + diffs
3. Venue Split      : home_form_pts_home, home_form_pts_away, away analoge
4. Goals Venue      : home_goals_scored_home/away + away analoge
5. Rank             : home_rank, away_rank, rank_diff
6. H2H              : h2h_home_wins, h2h_away_wins, h2h_draws
7. Draw Signals     : goals_scored_similarity, form_pts_closeness, avg_goals_conceded
8. Momentum         : home_goals_trend, away_goals_trend, exp form
9. Rest             : days_since_last_match_home/away
10. League dummies  : comp_PL, comp_PD, comp_BL1, comp_FL1, comp_SA
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

ML_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ML_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grupurile de features (numele coloanelor din metadata)
FEATURE_GROUPS = {
    "ELO":           ["home_elo", "away_elo", "elo_diff"],
    "Form":          ["home_form_pts", "home_form_gf", "home_form_ga",
                      "away_form_pts", "away_form_gf", "away_form_ga",
                      "form_pts_diff", "form_gf_diff", "form_ga_diff"],
    "Venue Split":   ["home_form_pts_home", "home_form_pts_away",
                      "away_form_pts_home", "away_form_pts_away"],
    "Goals Venue":   ["home_goals_scored_home", "home_goals_scored_away",
                      "away_goals_scored_home", "away_goals_scored_away"],
    "Rank":          ["home_rank", "away_rank", "rank_diff"],
    "H2H":           ["h2h_home_wins", "h2h_away_wins", "h2h_draws"],
    "Draw Signals":  ["goals_scored_similarity", "form_pts_closeness", "avg_goals_conceded"],
    "Momentum":      ["home_goals_trend", "away_goals_trend",
                      "home_form_pts_exp", "away_form_pts_exp"],
    "Rest":          ["days_since_last_match_home", "days_since_last_match_away"],
    "League":        ["comp_BL1", "comp_FL1", "comp_PD", "comp_PL", "comp_SA"],
}


def load_data():
    X_train = np.load(os.path.join(ML_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(ML_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(ML_DIR, "y_train.npy")).astype(int)
    y_test = np.load(os.path.join(ML_DIR, "y_test.npy")).astype(int)

    with open(os.path.join(ML_DIR, "metadata.json")) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    # Eliminam dead features
    dead_idx = [i for i, n in enumerate(feature_names) if n == "home_advantage"]
    if dead_idx:
        X_train = np.delete(X_train, dead_idx, axis=1)
        X_test = np.delete(X_test, dead_idx, axis=1)
        feature_names = [n for n in feature_names if n != "home_advantage"]

    return X_train, X_test, y_train, y_test, feature_names


def train_eval(X_train, y_train, X_test, y_test, label=""):
    """Antreneaza un Single XGBoost si evalueaza."""
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weight = np.array([weights[y] for y in y_train])

    model = XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=200, max_depth=4, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
        verbosity=0, n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_each = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_away": float(f1_each[0]),
        "f1_home": float(f1_each[1]),
        "f1_draw": float(f1_each[2]),
    }


def main():
    print("=" * 60)
    print("  ABLATION STUDY — Feature Group Contribution")
    print("=" * 60)

    print("\n[1/?] Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Total features: {len(feature_names)}")

    # 1. Baseline: toate features
    print("\n[BASELINE] Training with ALL features...")
    baseline = train_eval(X_train, y_train, X_test, y_test, "Baseline")
    print(f"  Acc: {baseline['accuracy']*100:.2f}%  F1mac: {baseline['f1_macro']:.4f}  F1draw: {baseline['f1_draw']:.4f}")

    # 2. Ablate fiecare grup
    results = {"baseline": baseline, "ablations": []}
    print(f"\n[ABLATIONS] Removing each feature group ({len(FEATURE_GROUPS)} groups)...")
    print(f"\n{'Group removed':<18} {'Acc':>7} {'ΔAcc':>8} {'F1mac':>7} {'ΔF1':>8} {'F1draw':>7} {'Impact':>10}")
    print("-" * 80)

    for group_name, features_in_group in FEATURE_GROUPS.items():
        # Indici features de eliminat
        remove_idx = [name_to_idx[f] for f in features_in_group if f in name_to_idx]
        if not remove_idx:
            continue

        # Eliminam coloanele
        keep_mask = np.ones(X_train.shape[1], dtype=bool)
        keep_mask[remove_idx] = False
        X_train_ab = X_train[:, keep_mask]
        X_test_ab = X_test[:, keep_mask]

        # Antrenam si evaluam
        ab_metrics = train_eval(X_train_ab, y_train, X_test_ab, y_test, group_name)

        d_acc = (ab_metrics["accuracy"] - baseline["accuracy"]) * 100
        d_f1 = ab_metrics["f1_macro"] - baseline["f1_macro"]

        # Impact = scaderea calitatii cand eliminam grupul
        # Un impact mare (negativ) = grupul e important
        impact = "MARE" if d_f1 < -0.01 else ("MEDIU" if d_f1 < -0.005 else "MIC")

        ab_metrics.update({
            "group": group_name,
            "n_features_removed": len(remove_idx),
            "delta_accuracy_pp": round(d_acc, 2),
            "delta_f1_macro": round(d_f1, 4),
            "impact": impact,
        })
        results["ablations"].append(ab_metrics)

        print(
            f"{group_name:<18} "
            f"{ab_metrics['accuracy']*100:>6.2f}% "
            f"{d_acc:>+7.2f}pp "
            f"{ab_metrics['f1_macro']:>7.4f} "
            f"{d_f1:>+8.4f} "
            f"{ab_metrics['f1_draw']:>7.4f} "
            f"{impact:>10}"
        )

    # Sortam dupa impact (delta F1 macro) — cele mai importante grupe au pierdere mare
    results["ablations"].sort(key=lambda x: x["delta_f1_macro"])

    print("\n[RANKING] Importanta grupurilor (sortate dupa impact pe F1 macro):")
    print(f"{'Rank':>4} {'Group':<18} {'ΔF1 macro':>12} {'Importanta':>12}")
    print("-" * 50)
    for i, ab in enumerate(results["ablations"], 1):
        print(f"{i:>4} {ab['group']:<18} {ab['delta_f1_macro']:>+12.4f} {ab['impact']:>12}")

    # Salvam
    out_path = os.path.join(OUTPUT_DIR, "ablation_study.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Salvat: {out_path}")

    print("\n" + "=" * 60)
    print("  ABLATION STUDY COMPLETE")
    print("=" * 60)
    print("\nInterpretare pentru dizertatie:")
    print("  - Grupurile de pe pozitiile 1-3 sunt cele mai valoroase")
    print("  - Grupurile de pe pozitiile 8-10 contribuie putin / pot fi eliminate")
    print("  - Daca un grup are ΔF1 ~ 0 sau pozitiv, e zgomot")


if __name__ == "__main__":
    main()
