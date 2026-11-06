#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: plot_error_types_multilabel.py

Purpose:
    Plot hallucination label statistics from gt_compare*.csv,
    including multi-label frequency and single-vs-multi-label distribution.

Inputs:
    - Tab-separated detection comparison CSV produced by compare_pred_gt.py
      (e.g. results/test/det/gt_compare_frozen.csv)

Outputs:
    - error_type_multilabel.png
    - error_type_multilabel_single_vs_multi.png
    - optional: error_type_multilabel_multi_cases.csv

Run:
    python plots/plot_error_types_multilabel.py \
        --in_csv results/test/det/gt_compare_frozen.csv \
        --out_png results/test/figs/error_type_multilabel.png \
        --export_multi_csv

Generates:
    - Hallucination label frequency chart
    - Single-label vs multi-label distribution chart
    - Optional CSV export of multi-label cases

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.2.2 Verteilung der Halluzinationstypen
    - Figure: Abbildung 10
    - Section: 7.2.3 Einzelne und kombinierte Halluzinationsstrukturen
    - Figure: Abbildung 11
"""

import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt


FN_PAT = re.compile(r"\bfn@\d+:", re.IGNORECASE)
PARAM_PAT = re.compile(r"\bparam@\d+:", re.IGNORECASE)


def ensure_out_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def plot_bar_with_labels(
    names,
    counts,
    total,
    title,
    ylabel,
    out_png,
    rotate=15,
    small_bar_ratio=0.12,
):
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, counts)

    if max(counts) > 0:
        plt.ylim(0, max(counts) * 1.20)

    for bar, v in zip(bars, counts):
        pct = (v / total * 100) if total else 0.0

        if max(counts) > 0 and v < small_bar_ratio * max(counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.5,
                f"{v} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            y = v * 0.95 if v > 0 else 0.5
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{v}\n({pct:.1f}%)",
                ha="center",
                va="top",
                fontsize=10,
                color="white" if v > 0 else "black",
            )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotate, ha="right")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    ensure_out_dir(out_png)
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"[OK] Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Path to gt_compare*.csv (tab-separated)")
    ap.add_argument("--out_png", default="error_type_multilabel.png")
    ap.add_argument(
        "--title",
        default="Frequency of Hallucination Labels (Multi-label; labels may overlap)",
    )
    ap.add_argument(
        "--export_multi_csv",
        action="store_true",
        help="Export multi-label cases to CSV",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, sep="\t")

    if "status" not in df.columns:
        raise ValueError("Input CSV must contain a 'status' column.")

    if "details" not in df.columns:
        df["details"] = ""

    df = df[df["status"].astype(str) != "NO_GT"].copy()
    total = len(df)

    status = df["status"].astype(str)
    details = df["details"].fillna("").astype(str)

    df["has_callcount"] = status.eq("CALL_COUNT_MISMATCH")
    df["has_pred_empty"] = status.eq("PRED_EMPTY")

    df["has_fn"] = details.apply(lambda s: bool(FN_PAT.search(s)))
    df["has_param"] = details.apply(lambda s: bool(PARAM_PAT.search(s)))

    df.loc[status.eq("FUNCTION_MISMATCH"), "has_fn"] = True
    df.loc[status.eq("PARAM_MISMATCH"), "has_param"] = True

    label_cols = ["has_callcount", "has_fn", "has_param"]

    # ---- Plot 1: label frequency ----
    names = [
        "Call-count mismatch",
        "Function mismatch",
        "Parameter mismatch",
    ]
    counts = [int(df[c].sum()) for c in label_cols]

    plot_bar_with_labels(
        names=names,
        counts=counts,
        total=total,
        title=args.title,
        ylabel="Number of Cases (labels can overlap)",
        out_png=args.out_png,
        rotate=15,
    )

    # ---- Plot 2: single vs multi ----
    df["n_labels"] = df[label_cols].sum(axis=1).astype(int)
    single = int((df["n_labels"] == 1).sum())
    multi = int((df["n_labels"] >= 2).sum())
    none = int((df["n_labels"] == 0).sum())

    out2 = args.out_png.replace(".png", "_single_vs_multi.png")

    plot_bar_with_labels(
        names=["Single-label", "Multi-label (>=2)", "Correct (no hallucination)"],
        counts=[single, multi, none],
        total=total,
        title="Single vs Multi-label ErrorHallucination Cases",
        ylabel="Number of Cases",
        out_png=out2,
        rotate=0,
        small_bar_ratio=0.20,
    )

    print(f"[INFO] Total counted (excluding NO_GT): {total}")
    print(f"[INFO] Single-label cases: {single} ({single/total*100:.1f}%)")
    print(f"[INFO] Multi-label cases: {multi} ({multi/total*100:.1f}%)")
    print(f"[INFO] No-label cases: {none} ({none/total*100:.1f}%)")

    if args.export_multi_csv:
        out_csv = args.out_png.replace(".png", "_multi_cases.csv")
        ensure_out_dir(out_csv)
        df[df["n_labels"] >= 2].to_csv(out_csv, index=False)
        print(f"Exported multi-label cases to: {out_csv}")


if __name__ == "__main__":
    main()