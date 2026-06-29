"""
calibration_analysis.py
=======================
Analiza calibrarii probabilitatilor pentru modelul v3.

Metrici calculate:
- Brier Score (multiclass, lower is better, 0 = perfect)
- Log Loss (cross-entropy)
- Expected Calibration Error (ECE)
- Reliability Diagram (per-class)

Justificare academica:
- Accuracy nu spune cat de "fiabile" sunt probabilitatile
- Un model poate avea acc 48% dar probabilitati slab calibrate
- Calibrarea conteaza enorm in pariuri sportive si in luarea deciziilor
- Brier Score < 0.20 = bun, < 0.18 = foarte bun in fotbal (3 clase)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss

# Importam clasa StackedEnsemble pentru unpickling
from stacked_ensemble import StackedEnsemble

ML_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ML_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Away Win", "Home Win", "Draw"]
CLASS_COLORS = ["#e74c3c", "#2ecc71", "#f39c12"]


def load_model_and_data():
    print("=" * 60)
    print("  CALIBRATION ANALYSIS — Football v3")
    print("=" * 60)

    model = joblib.load(os.path.join(ML_DIR, "model_v3.pkl"))
    X_test = np.load(os.path.join(ML_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(ML_DIR, "y_test.npy"))

    # Eliminam dead features la fel ca in train_model_v3.py
    with open(os.path.join(ML_DIR, "metadata.json")) as f:
        full_meta = json.load(f)
    full_features = full_meta["feature_names"]
    dead_idx = [i for i, n in enumerate(full_features) if n == "home_advantage"]
    if dead_idx:
        X_test = np.delete(X_test, dead_idx, axis=1)

    print(f"\n  Test samples: {len(y_test)}")
    print(f"  Features: {X_test.shape[1]}")
    return model, X_test, y_test


def y_to_onehot(y, n_classes=3):
    """One-hot encoding pentru Brier multiclass."""
    onehot = np.zeros((len(y), n_classes))
    onehot[np.arange(len(y)), y.astype(int)] = 1
    return onehot


def multiclass_brier(y_true, y_proba, n_classes=3):
    """
    Brier Score multiclass (suma peste clase) — varianta clasica folosita in
    literatura academica de fotbal (Constantinou 2013, Hubacek 2022).

    Formula: BS = (1/N) * sum_i sum_c (p_ic - y_ic)^2
    Lower is better. 0 = perfect, max = 2 (pentru 3 clase).

    Praguri tipice in literatura de fotbal:
    - < 0.55: foarte bun
    - 0.55-0.60: bun
    - 0.60-0.65: acceptabil
    - > 0.65: slab
    """
    onehot = y_to_onehot(y_true, n_classes)
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|
    Pentru multiclass folosim probabilitatea clasei prezise.
    """
    y_pred = y_proba.argmax(axis=1)
    confidences = y_proba.max(axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(avg_conf - avg_acc)

    return float(ece)


def reliability_diagram(y_true, y_proba, n_bins=10, save_path=None):
    """
    Reliability diagram: pentru fiecare clasa, compara probabilitatea
    prezisa medie (axa x) cu frecventa observata (axa y).
    Linia perfecta este y=x.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    for cls in range(3):
        ax = axes[cls]
        proba_cls = y_proba[:, cls]
        true_cls = (y_true == cls).astype(float)

        bin_pred_means = []
        bin_true_means = []
        bin_counts = []

        for i in range(n_bins):
            mask = (proba_cls > bin_edges[i]) & (proba_cls <= bin_edges[i + 1])
            if mask.sum() >= 10:
                bin_pred_means.append(proba_cls[mask].mean())
                bin_true_means.append(true_cls[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_pred_means.append(np.nan)
                bin_true_means.append(np.nan)
                bin_counts.append(0)

        bin_pred_means = np.array(bin_pred_means)
        bin_true_means = np.array(bin_true_means)
        bin_counts = np.array(bin_counts)

        # Linia perfecta y=x
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        # Punctele observate
        valid = ~np.isnan(bin_pred_means)
        ax.plot(
            bin_pred_means[valid],
            bin_true_means[valid],
            "o-",
            color=CLASS_COLORS[cls],
            markersize=8,
            linewidth=2,
            label=f"{CLASS_NAMES[cls]}",
        )

        # Histograma in fundal (cate predictii in fiecare bin)
        ax2 = ax.twinx()
        ax2.bar(
            bin_mids,
            bin_counts,
            width=0.08,
            alpha=0.2,
            color=CLASS_COLORS[cls],
        )
        ax2.set_ylabel("Number of predictions", color="gray", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="gray")

        ax.set_xlabel("Predicted probability", fontsize=11)
        ax.set_ylabel("Observed frequency", fontsize=11)
        ax.set_title(f"Reliability — {CLASS_NAMES[cls]}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Reliability diagram -> {save_path}")
    plt.close()


def confidence_distribution(y_proba, save_path=None):
    """Distributia probabilitatii maxime (cat de increzator e modelul)."""
    confidences = y_proba.max(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidences, bins=30, color="#3498db", alpha=0.7, edgecolor="black")
    ax.axvline(confidences.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {confidences.mean():.3f}")
    ax.set_xlabel("Max predicted probability (confidence)", fontsize=11)
    ax.set_ylabel("Number of predictions", fontsize=11)
    ax.set_title("Confidence Distribution — v3", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confidence distribution -> {save_path}")
    plt.close()


def per_class_brier(y_true, y_proba):
    """Brier score per clasa (one-vs-rest)."""
    out = {}
    for cls in range(3):
        true_cls = (y_true == cls).astype(int)
        out[CLASS_NAMES[cls]] = float(brier_score_loss(true_cls, y_proba[:, cls]))
    return out


def main():
    model, X_test, y_test = load_model_and_data()

    print("\n[1/4] Generating predictions...")
    y_proba = model.predict_proba(X_test)
    y_pred = y_proba.argmax(axis=1)

    print("\n[2/4] Computing calibration metrics...")
    brier_mc = multiclass_brier(y_test, y_proba)
    brier_per_cls = per_class_brier(y_test, y_proba)
    ll = log_loss(y_test, y_proba, labels=[0, 1, 2])
    ece = expected_calibration_error(y_test, y_proba)
    accuracy = (y_pred == y_test).mean()

    print(f"\n  Accuracy:                  {accuracy*100:.2f}%")
    print(f"  Brier Score (multiclass):  {brier_mc:.4f}  (lower is better, 0 = perfect)")
    print(f"  Log Loss:                  {ll:.4f}  (lower is better)")
    print(f"  Expected Calib. Error:     {ece:.4f}  (lower is better)")
    print(f"\n  Brier per class:")
    for cls, b in brier_per_cls.items():
        print(f"    {cls:<10}: {b:.4f}")

    # Interpretare (praguri din literatura de fotbal: Constantinou 2013, Hubacek 2022)
    print(f"\n  Interpretare:")
    if brier_mc < 0.55:
        print(f"    Brier {brier_mc:.4f} — FOARTE BUN pentru fotbal (literatura: <0.55)")
    elif brier_mc < 0.60:
        print(f"    Brier {brier_mc:.4f} — BUN pentru fotbal (literatura: <0.60)")
    elif brier_mc < 0.65:
        print(f"    Brier {brier_mc:.4f} — ACCEPTABIL pentru fotbal (literatura: <0.65)")
    else:
        print(f"    Brier {brier_mc:.4f} — necesita imbunatatiri de calibrare")

    if ece < 0.05:
        print(f"    ECE {ece:.4f} — calibrare excelenta")
    elif ece < 0.10:
        print(f"    ECE {ece:.4f} — calibrare buna")
    else:
        print(f"    ECE {ece:.4f} — calibrare slaba")

    print("\n[3/4] Generating reliability diagrams...")
    reliability_diagram(
        y_test, y_proba,
        save_path=os.path.join(OUTPUT_DIR, "reliability_diagram.png"),
    )
    confidence_distribution(
        y_proba,
        save_path=os.path.join(OUTPUT_DIR, "confidence_distribution.png"),
    )

    print("\n[4/4] Saving metrics...")
    results = {
        "accuracy": round(float(accuracy), 4),
        "brier_score_multiclass": round(brier_mc, 4),
        "brier_per_class": {k: round(v, 4) for k, v in brier_per_cls.items()},
        "log_loss": round(float(ll), 4),
        "expected_calibration_error": round(ece, 4),
        "interpretation": {
            "brier_threshold_excellent": 0.18,
            "brier_threshold_good": 0.20,
            "ece_threshold_excellent": 0.05,
            "ece_threshold_good": 0.10,
        },
        "n_test_samples": int(len(y_test)),
    }
    out_path = os.path.join(OUTPUT_DIR, "calibration_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  calibration_metrics.json -> {out_path}")

    print("\n" + "=" * 60)
    print("  CALIBRATION ANALYSIS COMPLETE")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
