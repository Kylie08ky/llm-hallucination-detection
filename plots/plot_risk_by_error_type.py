#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: plot_risk_by_error_type.py

Purpose:
    Plot the response patterns of the four risk signals
    (uncertainty, unit, self-consistency, and proxy)
    across different hallucination / error types.

Inputs:
    - Detection result CSV from compare_pred_gt.py
      (e.g. results/test/det/gt_compare_frozen.csv)
    - Risk output CSV from eval_cases.py
      (e.g. results/test/det/results_frozen.csv)

Outputs:
    - fig_risk_by_error_type.png

Run:
    python plots/plot_risk_by_error_type.py \
        --det_csv results/test/det/gt_compare_frozen.csv \
        --risk_csv results/test/det/results_frozen.csv \
        --out_dir results/test/figs

Generates:
    - Figure: fig_risk_by_error_type.png
    - Boxplots + scatter plots showing how each risk signal
      reacts to different error categories.

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.3.2 Analyse des Reaktionsmusters der Risikosignalschicht
    - Figure: Abbildung 15
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RISK_COLS = [
    ("uncertainty_risk", "Uncertainty Risk"),
    ("unit_risk", "Unit Risk"),
    ("sc_risk", "Self-Consistency Risk"),
    ("proxy_risk", "Proxy Risk"),
]

STATUS_MAP = {
    "CORRECT": "Correct",
    "FUNCTION_MISMATCH": "Function mismatch",
    "PARAM_MISMATCH": "Parameter mismatch",
    "CALL_COUNT_MISMATCH": "Call-count mismatch",
    "PRED_EMPTY": "Prediction empty",
    "NO_GT": "No GT",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_csv", required=True)
    parser.add_argument("--risk_csv", required=True)
    parser.add_argument("--out_dir", default="results/test/figs")
    args = parser.parse_args()

    det_df = pd.read_csv(args.det_csv, sep="\t")
    risk_df = pd.read_csv(args.risk_csv)

    # merge on case_id
    df = risk_df.merge(det_df[["case_id", "status"]], on="case_id", how="left")

    # map status names
    df["group"] = df["status"].map(STATUS_MAP)

    # Hard-fail override
    if "hard_fail" in df.columns:
        df.loc[df["hard_fail"] == True, "group"] = "Hard-Fail"

    # keep only valid groups
    df = df[df["group"].notna()]

    group_order = [
        "Correct",
        "Function mismatch",
        "Parameter mismatch",
        "Call-count mismatch",
        "Prediction empty",
        "Hard-Fail",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, RISK_COLS):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        data = []
        labels = []
        for g in group_order:
            vals = df[df["group"] == g][col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(g)

        ax.boxplot(data, labels=labels, showfliers=False)
        
        # jitter
        for i, vals in enumerate(data, start=1):
            if len(vals) > 500:
                vals = np.random.choice(vals, 500, replace=False)
            x = np.random.normal(i, 0.05, size=len(vals))
            ax.scatter(x, vals, alpha=0.25, s=10)

        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle("Fig Risk Signal Layer Response by Error Type", fontsize=14)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "fig_risk_by_error_type.png"
    fig.savefig(png_path, dpi=250)
    print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()