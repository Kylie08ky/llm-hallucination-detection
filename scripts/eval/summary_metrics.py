#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline merged vs routed merged and output a compact summary CSV.

Expected columns (baseline and routed):
- hard_fail (bool)
- status (CORRECT vs others)
- final_valid (bool) OR final_blocked (bool) OR (proxy_block/sc_block/llm_block)

Optional (for efficiency stats):
- route_llm (bool): whether LLM would be called under routing
- proxy_block (bool): used to define eligible set (post-gate & proxy pass)

This script computes post-gate confusion and metrics, plus LLM call rates.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def metrics_from_cm(tp, fp, fn, tn):
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return acc, precision, recall, f1


def derive_blocked(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit final_valid if present
    if "final_valid" in df.columns:
        return (~df["final_valid"].astype(bool))
    if "final_blocked" in df.columns:
        return df["final_blocked"].astype(bool)
    # fall back to OR of layer blocks
    cols = [c for c in ["proxy_block", "sc_block", "llm_block"] if c in df.columns]
    if cols:
        out = df[cols[0]].astype(bool)
        for c in cols[1:]:
            out = out | df[c].astype(bool)
        return out
    raise ValueError("Cannot derive blocked: missing final_valid/final_blocked and layer block columns.")


def compute_efficiency(post: pd.DataFrame) -> dict:
    """
    Efficiency stats:
    - eligible_n: post-gate and proxy pass count (if proxy_block exists; else equals post_gate_n)
    - llm_calls: number of samples where route_llm==True (if route_llm exists; else 0)
    - call rates: normalized by post_gate_n and eligible_n
    """
    post_gate_n = len(post)

    if "proxy_block" in post.columns:
        elig = post[post["proxy_block"].astype(bool) == False]
        eligible_n = int(len(elig))
    else:
        eligible_n = int(post_gate_n)

    if "route_llm" in post.columns:
        llm_calls = int(post["route_llm"].fillna(False).astype(bool).sum())
    else:
        # baseline (full LLM) often lacks route_llm; we keep 0 to avoid implying a rate we can't prove
        llm_calls = 0

    llm_call_rate_post_gate = (llm_calls / post_gate_n) if post_gate_n else 0.0
    llm_call_rate_eligible = (llm_calls / eligible_n) if eligible_n else 0.0

    return {
        "eligible_n": eligible_n,
        "llm_calls": llm_calls,
        "llm_call_rate_post_gate": llm_call_rate_post_gate,
        "llm_call_rate_eligible": llm_call_rate_eligible,
    }


def compute_one(df: pd.DataFrame) -> dict:
    if "hard_fail" in df.columns:
        post = df[df["hard_fail"] == False].copy()
    else:
        post = df.copy()

    y_true = (post["status"] != "CORRECT").astype(int)
    y_pred = derive_blocked(post).astype(int)

    tp, fp, fn, tn = confusion(y_true, y_pred)
    acc, precision, recall, f1 = metrics_from_cm(tp, fp, fn, tn)

    eff = compute_efficiency(post)

    return {
        "post_gate_n": len(post),
        **eff,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1,
        "blocked_rate": float(y_pred.mean()) if len(post) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--merged_routed", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    base_df = pd.read_csv(args.merged)
    routed_df = pd.read_csv(args.merged_routed)

    base = compute_one(base_df)
    routed = compute_one(routed_df)

    out = pd.DataFrame([
        {"system": "baseline", **base},
        {"system": "routed", **routed},
    ])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[OK] wrote", out_path)


if __name__ == "__main__":
    main()
