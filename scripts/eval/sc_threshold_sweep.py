#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep SC threshold on the subset that passed Proxy (post-gate) and report SC-only metrics.

Why: on DEV you often want to choose sc_threshold that improves F1 *conditional on proxy pass*.

Expected columns in merged CSV:
- hard_fail (bool)
- status (CORRECT vs others)
- proxy_block (bool)
- sc_risk (float)

Usage:
  PYTHONPATH=src python scripts/eval/sc_threshold_sweep.py \
    --merged results/dev/merged/merged_cases.csv \
    --out_csv results/dev/metrics/sc_threshold_sweep.csv \
    --thr_min 0.0 --thr_max 0.8 --thr_step 0.02
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def confusion(y_true, y_pred):
    tp = int(((y_true==1) & (y_pred==1)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    return tp, fp, fn, tn

def metrics_from_cm(tp, fp, fn, tn):
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return precision, recall, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--thr_min", type=float, default=0.0)
    ap.add_argument("--thr_max", type=float, default=0.8)
    ap.add_argument("--thr_step", type=float, default=0.02)
    args = ap.parse_args()

    df = pd.read_csv(args.merged)
    if "hard_fail" in df.columns:
        df = df[df["hard_fail"] == False].copy()
    # conditional on proxy pass
    df = df[df["proxy_block"] == False].copy()

    df["y_true"] = (df["status"] != "CORRECT").astype(int)
    df["sc_risk"] = pd.to_numeric(df["sc_risk"], errors="coerce").fillna(0.0)

    thrs = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)

    rows = []
    for thr in thrs:
        y_pred = (df["sc_risk"] > float(thr)).astype(int)
        tp, fp, fn, tn = confusion(df["y_true"], y_pred)
        precision, recall, f1 = metrics_from_cm(tp, fp, fn, tn)
        rows.append({
            "thr_sc": float(thr),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1,
            "n": len(df),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    best = max(rows, key=lambda r: r["F1"])
    print("[OK] wrote", out_path)
    print(f"[BEST] thr_sc={best['thr_sc']:.3f} F1={best['F1']:.4f} P={best['Precision']:.4f} R={best['Recall']:.4f} (proxy-pass subset, n={best['n']})")

if __name__ == "__main__":
    main()
