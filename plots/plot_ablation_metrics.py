#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: plot_ablation_metrics.py

Purpose:
    Plot ablation-study figures for different system variants,
    including Accuracy, Precision/Recall/F1, and risk_final_mean.

Inputs:
    - Ablation metrics CSV
      (e.g. ablation_metrics.csv)

Outputs:
    - fig_ablation_prf.png
    - fig_ablation_accuracy.png
    - fig_ablation_riskmean_llmcall.png

Run:
    python plots/plot_ablation_metrics.py \
        --csv results/dev/metrics/ablation_metrics.csv \
        --out_dir results/dev/figs \
        --title_prefix "Ablation (dev)"

Generates:
    - Accuracy comparison by system variant
    - Precision / Recall / F1 comparison by system variant
    - risk_final_mean comparison by system variant
      (and llm_call_rate if available)

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.6.1 Vergleich der Gesamtgenauigkeit
    - Figure: Ablation Accuracy figure
    - Section: 7.6.2 Analyse von Precision / Recall / F1
    - Figure: Ablation PRF figure
    - Section: 7.6.4 Analyse der Risikoverteilung und des Systemverhaltens
    - Figure: risk_final_mean figure
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ablation_fusion_metrics.csv")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--title_prefix", default="Ablation", help="prefix for figure titles")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # keep the given order in CSV
    systems = df["system"].astype(str).tolist()
    x = np.arange(len(systems))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Fig 1: Precision / Recall / F1 (grouped bars)
    # -----------------------------
    metrics = ["Precision", "Recall", "F1"]
    bar_w = 0.25

    plt.figure(figsize=(12, 5))
    for i, m in enumerate(metrics):
        vals = df[m].astype(float).values
        plt.bar(x + (i - 1) * bar_w, vals, width=bar_w, label=m)

        # value labels
        for xi, yi in zip(x + (i - 1) * bar_w, vals):
            plt.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, systems, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"{args.title_prefix}: Precision / Recall / F1 by System")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ablation_prf.png", dpi=300)
    plt.close()

    # -----------------------------
    # Fig 2: Accuracy
    # -----------------------------
    plt.figure(figsize=(12, 4.8))
    acc = df["Accuracy"].astype(float).values
    plt.bar(x, acc)
    for xi, yi in zip(x, acc):
        plt.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, systems, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(f"{args.title_prefix}: Accuracy by System")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ablation_accuracy.png", dpi=300)
    plt.close()

    # -----------------------------
    # Fig 3: risk_final_mean & llm_call_rate (optional, dual-axis)
    # -----------------------------
    if "risk_final_mean" in df.columns:
        rmean = df["risk_final_mean"].astype(float).values
    else:
        rmean = None

    llm_ok = ("llm_call_rate" in df.columns) and (df["llm_call_rate"].notna().any())
    llm = df["llm_call_rate"].astype(float).values if llm_ok else None

    if rmean is not None:
        fig, ax1 = plt.subplots(figsize=(12, 4.8))
        ax1.bar(x, rmean)
        ax1.set_ylabel("risk_final_mean")
        ax1.set_xticks(x)
        ax1.set_xticklabels(systems, rotation=25, ha="right")
        ax1.set_title(f"{args.title_prefix}: risk_final_mean (and LLM call rate if available)")

        # labels for risk mean
        for xi, yi in zip(x, rmean):
            ax1.text(xi, yi + (max(rmean) * 0.02 if max(rmean) > 0 else 0.01),
                     f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

        if llm is not None:
            ax2 = ax1.twinx()
            ax2.plot(x, llm, marker="o")
            ax2.set_ylabel("llm_call_rate")
            for xi, yi in zip(x, llm):
                ax2.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "fig_ablation_riskmean_llmcall.png", dpi=300)
        plt.close(fig)

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()