"""
experiment_streaks.py
=====================
Adauga features de tip streaks/momentum si testeaza vs baseline v3.

Features noi (point-in-time, fara leakage):
  - home/away_win_streak       : nr consecutive de victorii imediat inainte de meci
  - home/away_loss_streak      : nr consecutive de infrangeri
  - home/away_unbeaten_streak  : nr consecutive fara infrangere (W sau D)
  - home/away_winless_streak   : nr consecutive fara victorie (L sau D)
  - home/away_clean_sheets     : nr meciuri consecutive fara gol primit
  - home/away_scoring_streak   : nr meciuri consecutive cu cel putin un gol marcat

Pentru fiecare meci, valorile sunt calculate doar pe meciurile dinaintea lui
(pe baza datei). Se evita orice leakage.
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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


# ============================================================
# Streaks computation (point-in-time, no leakage)
# ============================================================
def compute_streaks(df):
    """
    Pentru fiecare meci, calculeaza streaks-urile pe baza istoricului
    fiecarei echipe (pre-match). df trebuie sortat pe date crescator.

    Returneaza 12 array-uri paralele cu len(df).
    """
    # Per-team istoric: lista de tuple (outcome, gf, ga)
    # outcome: 'W', 'L', 'D' din perspectiva echipei
    history = {}

    h_win = []; h_loss = []; h_unb = []; h_winless = []; h_cs = []; h_sc = []
    a_win = []; a_loss = []; a_unb = []; a_winless = []; a_cs = []; a_sc = []

    def streak(hist, predicate):
        """Counteaza streak-ul curent (de la coada) cat timp predicate(item) e True."""
        n = 0
        for item in reversed(hist):
            if predicate(item):
                n += 1
            else:
                break
        return n

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]

        # PRE-match: citim streaks din istoric
        h_hist = history.get(htid, [])
        a_hist = history.get(atid, [])

        h_win.append(    streak(h_hist, lambda x: x[0] == 'W'))
        h_loss.append(   streak(h_hist, lambda x: x[0] == 'L'))
        h_unb.append(    streak(h_hist, lambda x: x[0] in ('W', 'D')))
        h_winless.append(streak(h_hist, lambda x: x[0] in ('L', 'D')))
        h_cs.append(     streak(h_hist, lambda x: x[2] == 0))
        h_sc.append(     streak(h_hist, lambda x: x[1] > 0))

        a_win.append(    streak(a_hist, lambda x: x[0] == 'W'))
        a_loss.append(   streak(a_hist, lambda x: x[0] == 'L'))
        a_unb.append(    streak(a_hist, lambda x: x[0] in ('W', 'D')))
        a_winless.append(streak(a_hist, lambda x: x[0] in ('L', 'D')))
        a_cs.append(     streak(a_hist, lambda x: x[2] == 0))
        a_sc.append(     streak(a_hist, lambda x: x[1] > 0))

        # POST-match: actualizam istoricul
        res = row["result"]
        hg, ag = row["home_goals"], row["away_goals"]

        if res == 1:
            h_outcome, a_outcome = 'W', 'L'
        elif res == 0:
            h_outcome, a_outcome = 'L', 'W'
        else:
            h_outcome, a_outcome = 'D', 'D'

        history.setdefault(htid, []).append((h_outcome, hg, ag))
        history.setdefault(atid, []).append((a_outcome, ag, hg))

    return {
        "home_win_streak":      np.array(h_win,     dtype=np.float32),
        "home_loss_streak":     np.array(h_loss,    dtype=np.float32),
        "home_unbeaten_streak": np.array(h_unb,     dtype=np.float32),
        "home_winless_streak":  np.array(h_winless, dtype=np.float32),
        "home_clean_sheets":    np.array(h_cs,      dtype=np.float32),
        "home_scoring_streak":  np.array(h_sc,      dtype=np.float32),
        "away_win_streak":      np.array(a_win,     dtype=np.float32),
        "away_loss_streak":     np.array(a_loss,    dtype=np.float32),
        "away_unbeaten_streak": np.array(a_unb,     dtype=np.float32),
        "away_winless_streak":  np.array(a_winless, dtype=np.float32),
        "away_clean_sheets":    np.array(a_cs,      dtype=np.float32),
        "away_scoring_streak":  np.array(a_sc,      dtype=np.float32),
    }


# ============================================================
# Train + eval
# ============================================================
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
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    return {
        "label": label,
        "n_features": X_train.shape[1],
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_away": f1_each[0],
        "f1_home": f1_each[1],
        "f1_draw": f1_each[2],
        "cm": cm.tolist(),
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  EXPERIMENT: Streaks / Momentum Features")
    print("=" * 70)

    print("\n[1] Loading data_processed.csv...")
    df = pd.read_csv(p("data_processed.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Total meciuri: {len(df)}")

    # Baseline feature names (din metadata.json, fara home_advantage)
    with open(p("metadata.json")) as f:
        meta = json.load(f)
    base_features = [n for n in meta["feature_names"] if n != "home_advantage"]
    print(f"  Baseline features: {len(base_features)}")

    # Compute streaks
    print("\n[2] Computing streaks (point-in-time)...")
    streaks = compute_streaks(df)
    streak_cols = list(streaks.keys())
    for col, arr in streaks.items():
        df[col] = arr
    print(f"  Streaks computed: {len(streak_cols)} features")
    # Sanity: distributii
    for col in streak_cols:
        print(f"    {col:<28} mean={df[col].mean():.2f}  max={df[col].max():.0f}")

    # Build feature sets
    full_features = base_features + streak_cols

    # Temporal split (acelasi 80/20 ca in pipeline)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
    y_train = df_train["result"].values.astype(int)
    y_test  = df_test["result"].values.astype(int)

    print(f"\n[3] Temporal split: {len(df_train)} train / {len(df_test)} test")
    print(f"  Train dates: {df_train['date'].iloc[0].date()} -> {df_train['date'].iloc[-1].date()}")
    print(f"  Test dates:  {df_test['date'].iloc[0].date()} -> {df_test['date'].iloc[-1].date()}")

    best_params = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 500}

    schemes = [
        ("Baseline (40 feat)",       base_features),
        ("+ Streaks (52 feat)",      full_features),
    ]

    results = []
    for i, (label, cols) in enumerate(schemes, 1):
        print(f"\n[{3+i}] Training: {label} ({len(cols)} features)")
        X_tr_raw = df_train[cols].values.astype(np.float32)
        X_te_raw = df_test[cols].values.astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        r = train_and_eval(X_tr, y_train, X_te, y_test, label, best_params)
        results.append(r)
        print(f"     Accuracy: {r['accuracy']*100:.2f}%  F1 macro: {r['f1_macro']:.4f}  F1 Draw: {r['f1_draw']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("  REZULTATE COMPARATIVE")
    print("=" * 70)
    print(f"  {'Scheme':<28} {'#Feat':>6} {'Accuracy':>10} {'F1 macro':>10} {'F1 Draw':>10}")
    print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['label']:<28} {r['n_features']:>6} {r['accuracy']*100:>9.2f}% "
              f"{r['f1_macro']:>10.4f} {r['f1_draw']:>10.4f}")

    if len(results) >= 2:
        b, s = results[0], results[1]
        d_acc  = (s["accuracy"] - b["accuracy"]) * 100
        d_f1   = s["f1_macro"] - b["f1_macro"]
        d_draw = s["f1_draw"] - b["f1_draw"]
        print(f"\n  Delta (Streaks vs Baseline): {d_acc:>+6.2f}pp acc  {d_f1:>+7.4f} f1m  {d_draw:>+7.4f} f1draw")

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
                "cm": r["cm"],
            } for r in results
        ],
        "streak_features": streak_cols,
    }
    with open(p("experiment_streaks.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: experiment_streaks.json")


if __name__ == "__main__":
    main()
