#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: plot_stability_figures.py

Purpose:
    Plot stability and reproducibility figures from
    stability_per_case.csv and stability_per_run.csv.

Inputs:
    - stability_per_case.csv
    - stability_per_run.csv (optional but recommended)

Outputs:
    - fig_stab_true_1_consistency_hist.png
    - fig_stab_true_2_risk_std_hist.png
    - fig_stab_true_3_disagreement_hist.png
    - fig_stab_true_4_topk_lines.png

Run:
    python plots/plot_stability_figures.py \
        --case_csv results/test/stability/stability_per_case.csv \
        --run_csv results/test/stability/stability_per_run.csv \
        --out_dir results/test/figs \
        --topk 10

Generates:
    - Decision consistency distribution
    - risk_final standard deviation distribution
    - disagreement rate distribution
    - Top-K most unstable cases across runs

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.5.2 Analyse der Entscheidungs-Konsistenzverteilung
    - Figure: Abbildung 20
    - Section: 7.5.3 Analyse der Risikoschwankungsverteilung
    - Figure: Abbildung 21
    - Section: 7.5.4 Zusammenhang zwischen struktureller Divergenz und Risikoschwankung
    - Figures: Abbildung 14, Abbildung 22
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_csv", required=True, help="stability_per_case.csv")
    ap.add_argument("--run_csv", default=None, help="stability_per_run.csv (optional but recommended)")
    ap.add_argument("--out_dir", required=True, help="output dir for figures")
    ap.add_argument("--topk", type=int, default=10, help="top K most unstable cases for line plot")
    ap.add_argument("--bins", type=int, default=12, help="hist bins")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_case = pd.read_csv(args.case_csv)

    # ------------------------------
    # Fig 1: decision_consistency distribution
    # ------------------------------
    plt.figure()
    plt.hist(df_case["decision_consistency"].dropna(), bins=args.bins)
    plt.title("Decision Consistency Distribution (stability)")
    plt.xlabel("Decision Consistency (0-1)")
    plt.ylabel("Number of Cases")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_stab_true_1_consistency_hist.png", dpi=300)
    plt.close()

    # ------------------------------
    # Fig 2: risk_std distribution
    # ------------------------------
    plt.figure()
    plt.hist(df_case["risk_std"].dropna(), bins=args.bins)
    plt.title("risk_final Std Distribution (stability_true)")
    plt.xlabel("risk_final standard deviation")
    plt.ylabel("Number of Cases")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_stab_true_2_risk_std_hist.png", dpi=300)
    plt.close()

    # ------------------------------
    # Fig 3: disagreement_rate distribution (if exists)
    # ------------------------------
    if "disagreement_rate" in df_case.columns:
        plt.figure()
        plt.hist(df_case["disagreement_rate"].dropna(), bins=args.bins)
        plt.title("Run-Structure Disagreement Rate Distribution")
        plt.xlabel("disagreement_rate (0-1)")
        plt.ylabel("Number of Cases")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_stab_true_3_disagreement_hist.png", dpi=300)
        plt.close()

    # ------------------------------
    # Fig 4: Top-K most unstable cases (line plot across runs)
    # ------------------------------
    if args.run_csv:
        df_run = pd.read_csv(args.run_csv)

        # pick topK by risk_range (preferred), else by risk_std
        if "risk_range" in df_case.columns:
            top_ids = (
                df_case.sort_values("risk_range", ascending=False)
                      .head(args.topk)["case_id"].tolist()
            )
        else:
            top_ids = (
                df_case.sort_values("risk_std", ascending=False)
                      .head(args.topk)["case_id"].tolist()
            )

        # plot one figure: each case a line of 5 points
        plt.figure()
        for cid in top_ids:
            sub = df_run[df_run["case_id"] == cid].sort_values("run_index")
            if len(sub) == 0:
                continue
            plt.plot(sub["run_index"], sub["risk_final"], marker="o", label=cid)

        plt.title(f"Top-{args.topk} Most Unstable Cases: risk_final across runs")
        plt.xlabel("run_index")
        plt.ylabel("risk_final")
        plt.xticks(sorted(df_run["run_index"].unique()))
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_stab_true_4_topk_lines.png", dpi=300)
        plt.close()

    print("figures saved to:", out_dir)


if __name__ == "__main__":
    main()