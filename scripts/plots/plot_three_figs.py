#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusion-aware plots (3 figs) for the current risk-fusion pipeline.

FIG1: Distribution of risk_final by label (CORRECT vs INCORRECT) with decision threshold line.
FIG2: Baseline vs Routed metrics bar chart (Precision/Recall/F1) from summary_overall.csv (if exists).
FIG3: Contribution distributions (contrib_*) by label to explain fusion (INCORRECT vs CORRECT).

Run:
  PYTHONPATH=src python scripts/eval/plots/plot_three_figs.py \
    --merged results/dev/merged/merged_cases.csv \
    --out_dir results/dev/figs

Optionally if you have routed + summary:
  --merged_routed results/dev/merged/merged_cases_routed.csv \
  --summary results/dev/metrics/summary_overall.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _post_gate(df: pd.DataFrame) -> pd.DataFrame:
    if "hard_fail" in df.columns:
        return df[df["hard_fail"] == False].copy()
    return df.copy()


def _label_series(df: pd.DataFrame) -> pd.Series:
    if "status" not in df.columns:
        raise ValueError("merged csv missing 'status' column.")
    # INCORRECT=1 (hallucination), CORRECT=0
    return (df["status"] != "CORRECT").astype(int)


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _get_final_threshold(df: pd.DataFrame, override: float | None) -> float:
    if override is not None:
        return float(override)
    if "final_threshold" in df.columns:
        v = pd.to_numeric(df["final_threshold"], errors="coerce").dropna()
        if len(v) > 0:
            return float(v.iloc[0])
    # fallback default used in your runs
    return 0.20


def fig1_risk_final_distribution(df: pd.DataFrame, out_path: Path, final_thr: float):
    y = _label_series(df)
    risk = _numeric(df, "risk_final", default=np.nan)

    corr = risk[y == 0].dropna().values
    inc = risk[y == 1].dropna().values

    plt.figure(figsize=(10, 5))
    bins = 30

    plt.hist(corr, bins=bins, alpha=0.6, label=f"CORRECT (n={len(corr)})")
    plt.hist(inc, bins=bins, alpha=0.6, label=f"INCORRECT (n={len(inc)})")

    plt.axvline(final_thr, linewidth=2, linestyle="--", label=f"final_threshold={final_thr:.3f}")
    plt.title("Fusion decision signal: risk_final distribution (post-gate)")
    plt.xlabel("risk_final")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig2_baseline_vs_routed(summary_csv: Path, out_path: Path):
    if not summary_csv.exists():
        return

    s = pd.read_csv(summary_csv)
    if "system" not in s.columns:
        return

    def get_row(name: str):
        r = s[s["system"].astype(str).str.lower() == name]
        return r.iloc[0] if len(r) else None

    b = get_row("baseline")
    r = get_row("routed")

    if b is None or r is None:
        return

    # ---- performance metrics ----
    metrics = ["Precision", "Recall", "F1"]
    b_perf = [float(b[m]) for m in metrics]
    r_perf = [float(r[m]) for m in metrics]

    # ---- cost / behavior metrics ----
    cost_metrics = ["llm_call_rate_post_gate", "blocked_rate"]
    b_cost = [float(b[m]) for m in cost_metrics]
    r_cost = [float(r[m]) for m in cost_metrics]

    x1 = np.arange(len(metrics))
    x2 = np.arange(len(cost_metrics)) + len(metrics) + 1
    w = 0.35

    plt.figure(figsize=(10, 5))

    # performance bars
    plt.bar(x1 - w/2, b_perf, width=w, label="baseline (perf)")
    plt.bar(x1 + w/2, r_perf, width=w, label="routed (perf)")

    # cost bars
    plt.bar(x2 - w/2, b_cost, width=w, label="baseline (cost)")
    plt.bar(x2 + w/2, r_cost, width=w, label="routed (cost)")

    # annotate
    for i, v in enumerate(b_perf):
        plt.text(x1[i] - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(r_perf):
        plt.text(x1[i] + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    for i, v in enumerate(b_cost):
        plt.text(x2[i] - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(r_cost):
        plt.text(x2[i] + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    plt.ylim(0, 1.0)
    plt.xticks(
        list(x1) + list(x2),
        metrics + ["CallRate", "BlockedRate"],
        rotation=30
    )

    plt.title("Baseline vs Routed: Performance and Cost")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def fig3_contrib_distributions(df: pd.DataFrame, out_path: Path):
    y = _label_series(df)

    # Prefer contrib_* columns (already weighted contributions)
    contrib_cols = [c for c in df.columns if c.startswith("contrib_")]
    # Keep common ordering if exists
    preferred = ["contrib_unc", "contrib_unit", "contrib_sc", "contrib_llm", "contrib_proxy"]
    ordered = [c for c in preferred if c in contrib_cols] + [c for c in contrib_cols if c not in preferred]

    # If no contrib_* exist, try raw risks as fallback
    if len(ordered) == 0:
        fallback = ["uncertainty_risk", "unit_risk", "sc_risk", "llm_risk", "proxy_risk"]
        ordered = [c for c in fallback if c in df.columns]
        if len(ordered) == 0:
            return

    # Build boxplot data: for each signal, show CORRECT vs INCORRECT
    labels = []
    data = []

    for c in ordered:
        vals = _numeric(df, c, default=np.nan)
        corr = vals[y == 0].dropna().values
        inc = vals[y == 1].dropna().values
        data.append(corr)
        data.append(inc)
        labels.append(f"{c}\nCOR")
        labels.append(f"{c}\nINC")

    plt.figure(figsize=(max(10, len(labels) * 0.9), 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Fusion explanation: contribution distributions (post-gate)")
    plt.ylabel("value (contrib_* preferred; else raw risk)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged_cases.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--merged_routed", default="", help="optional merged_cases_routed.csv (not required)")
    ap.add_argument("--summary", default="", help="optional summary_overall.csv")
    ap.add_argument("--final_threshold", type=float, default=None, help="override final_threshold for fig1 line")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _post_gate(_read_csv(args.merged))
    final_thr = _get_final_threshold(df, args.final_threshold)

    fig1_risk_final_distribution(
        df,
        out_dir / "fig1_risk_final_distribution.png",
        final_thr
    )

    # Use summary_overall.csv if given; else try to infer common path
    summary_csv = Path(args.summary) if args.summary else None
    if summary_csv is None or not summary_csv.exists():
        # try same tree default
        maybe = Path(args.merged).parent.parent / "metrics" / "summary_overall.csv"
        summary_csv = maybe if maybe.exists() else None

    if summary_csv is not None and summary_csv.exists():
        fig2_baseline_vs_routed(summary_csv, out_dir / "fig2_baseline_vs_routed.png")

    fig3_contrib_distributions(df, out_dir / "fig3_contrib_distributions.png")

    print("[OK] wrote 3 figs to:", str(out_dir))


if __name__ == "__main__":
    main()
