"""
confusion_matrix_viz.py
=======================
Vizualizare confusion matrix pentru modelul v3 + analiza erorilor.

Output:
- output/confusion_matrix.png         — heatmap absolut
- output/confusion_matrix_normalized.png — normalizat per rand (recall per clasa)
- output/error_analysis.json          — unde greseste cel mai mult modelul
"""

import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from stacked_ensemble import StackedEnsemble

ML_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ML_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Away Win (0)", "Home Win (1)", "Draw (2)"]


def plot_confusion_matrix(cm, classes, title, normalize=False, save_path=None):
    if normalize:
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = cm_pct
        fmt = ".1f"
        suffix = "%"
        cmap = plt.cm.YlGnBu
    else:
        annot = cm
        fmt = "d"
        suffix = ""
        cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(annot, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_yticklabels(classes)

    # Adaugam valori in celule
    thresh = annot.max() / 2.0
    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            color = "white" if annot[i, j] > thresh else "black"
            text = f"{annot[i, j]:{fmt}}{suffix}"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def analyze_errors(cm, y_true, y_pred):
    """Identifica unde greseste cel mai mult modelul."""
    n_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    recall_per_class = correct_per_class / n_per_class

    # Cele mai frecvente erori (cu exclusarea diagonalei)
    errors = []
    labels_short = ["Away", "Home", "Draw"]
    for i in range(3):
        for j in range(3):
            if i != j:
                pct_of_actual = cm[i, j] / n_per_class[i] * 100
                errors.append({
                    "actual": labels_short[i],
                    "predicted": labels_short[j],
                    "count": int(cm[i, j]),
                    "pct_of_actual": round(float(pct_of_actual), 2),
                })

    errors.sort(key=lambda e: -e["count"])

    return {
        "recall_per_class": {
            labels_short[i]: round(float(recall_per_class[i]), 4)
            for i in range(3)
        },
        "n_per_class": {
            labels_short[i]: int(n_per_class[i]) for i in range(3)
        },
        "top_errors": errors,
    }


def main():
    print("=" * 60)
    print("  CONFUSION MATRIX & ERROR ANALYSIS — v3")
    print("=" * 60)

    print("\n[1/3] Loading model and data...")
    model = joblib.load(os.path.join(ML_DIR, "model_v3.pkl"))
    X_test = np.load(os.path.join(ML_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(ML_DIR, "y_test.npy"))

    with open(os.path.join(ML_DIR, "metadata.json")) as f:
        full_meta = json.load(f)
    dead_idx = [i for i, n in enumerate(full_meta["feature_names"]) if n == "home_advantage"]
    if dead_idx:
        X_test = np.delete(X_test, dead_idx, axis=1)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>15} {'Away':>8} {'Home':>8} {'Draw':>8}")
    for i in range(3):
        print(f"{CLASS_NAMES[i]:>15} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")

    print("\n[2/3] Generating heatmaps...")
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        title="Confusion Matrix — v3 (absolute counts)",
        normalize=False,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        title="Confusion Matrix — v3 (normalized per actual class)",
        normalize=True,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"),
    )

    print("\n[3/3] Error analysis...")
    analysis = analyze_errors(cm, y_test, y_pred)

    print(f"\nRecall per class:")
    for cls, r in analysis["recall_per_class"].items():
        print(f"  {cls:<6}: {r*100:.2f}% ({analysis['n_per_class'][cls]} samples)")

    print(f"\nTop confusion patterns (sorted by count):")
    for i, e in enumerate(analysis["top_errors"][:6], 1):
        print(f"  {i}. Actual={e['actual']:<5} -> Predicted={e['predicted']:<5}: "
              f"{e['count']:>4} cases ({e['pct_of_actual']:.1f}% of actual {e['actual']})")

    # Insights pentru dizertatie
    draw_misclassif = [e for e in analysis["top_errors"] if e["actual"] == "Draw"]
    draw_lost_to = sum(e["count"] for e in draw_misclassif)
    draw_total = analysis["n_per_class"]["Draw"]
    print(f"\nInsights pentru dizertatie:")
    print(f"  - Egaluri prezise corect: {analysis['recall_per_class']['Draw']*100:.1f}%")
    print(f"  - Egaluri ratate: {draw_lost_to}/{draw_total} ({draw_lost_to/draw_total*100:.1f}%)")
    print(f"  - Egalurile sunt cel mai greu de prezis (dezechilibru natural in fotbal)")

    # Salvam JSON
    out = {
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "analysis": analysis,
    }
    out_path = os.path.join(OUTPUT_DIR, "error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("  CONFUSION MATRIX ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
