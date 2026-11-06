#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: plot_confusion_matrix.py

Purpose:
    Plot the post-gate confusion matrix of the final fusion decision.

Inputs:
    - Merged evaluation CSV containing at least:
      status, blocked, and optionally hard_fail
      (e.g. results/test/merged/merged_postgate.csv)

Outputs:
    - fig_confusion_matrix.png

Run:
    python plots/plot_confusion_matrix.py \
        --merged results/test/merged/merged_cases.csv \
        --out_dir results/test/figs

Generates:
    - Figure: fig_confusion_matrix.png
    - Confusion matrix of fusion decision for post-gate samples.

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.4.2 Konfusionsmatrix und Gesamtleistungskennzahlen
    - Figure: Abbildung 18
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out_dir", default="results/test/figs")
    args = ap.parse_args()

    df = pd.read_csv(args.merged)

    if "hard_fail" in df.columns:
        df = df[df["hard_fail"] == False].copy()

    y_true = (df["status"] != "CORRECT").astype(int).values
    y_pred = (df["blocked"] == True).astype(int).values

    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))

    cm = np.array([[TN, FP],
                   [FN, TP]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Pass", "Pred: Block"])
    ax.set_yticklabels(["True: Correct", "True: Incorrect"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix of Fusion Decision (post-gate)")

    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            text_color = "white" if value > cm.max() * 0.6 else "black"
            ax.text(j, i, f"{value}",
                    ha="center", va="center",
                    color=text_color, fontsize=14)

    fig.colorbar(im)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig_confusion_matrix.png", dpi=300)
    plt.close()

    print("Saved confusion matrix.")


if __name__ == "__main__":
    main()