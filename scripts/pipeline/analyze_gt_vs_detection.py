#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk-fusion analysis: merge detection results with GT, emit contribution-centric report.

Inputs
------
1) Detection CSV from eval_cases (e.g., results/dev/det/results_single.csv)
2) GT compare TSV/CSV from compare_pred_gt.py (e.g., results/dev/det/gt_compare_single.csv)

Outputs (under --out_dir)
-------------------------
- merged_cases.csv
- report_by_type.csv        (risk-fusion contribution report)
- correct_but_blocked.csv

Key idea
--------
Your system is risk-fusion (risk_final >= final_threshold), so per-layer "block_rate" is not meaningful.
We report:
- blocked_rate by GT status
- routed_rate / llm_ran_rate
- mean weighted contribution per layer (unc/sc/llm/unit/proxy)
- dominant contributor shares among BLOCKED samples per status
"""

import argparse
import os
import pandas as pd
from collections import Counter

def _read_gt_compare(path: str) -> pd.DataFrame:
    # compare_pred_gt writes a tab-separated file but uses .csv extension.
    try:
        df = pd.read_csv(path, sep="\\t", engine="python")
        if "case_id" in df.columns and "status" in df.columns:
            return df
    except Exception:
        pass
    return pd.read_csv(path)

def _resolve_blocked_column(df: pd.DataFrame) -> pd.Series:
    if "blocked" in df.columns:
        return df["blocked"].astype(bool)
    if "final_blocked" in df.columns:
        return df["final_blocked"].astype(bool)
    if "final_valid" in df.columns:
        return ~df["final_valid"].astype(bool)
    if "final_decision" in df.columns:
        return df["final_decision"].astype(str).str.lower().isin(["block","blocked","fail","false"])
    raise RuntimeError("Cannot find final decision column (blocked / final_blocked / final_valid / final_decision)")

def _get_weights(det: pd.DataFrame, args) -> dict:
    # Prefer per-row weight columns if you ever choose to write them.
    # Otherwise use CLI args (recommended for reproducibility).
    w = {
        "unc": float(args.w_unc),
        "sc": float(args.w_sc),
        "llm": float(args.w_llm),
        "unit": float(args.w_unit),
        "proxy": float(args.w_proxy),
    }
    # Normalize to sum=1 for consistent "contribution" interpretation
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-12:
        # fallback
        w = {"unc":0.35,"sc":0.20,"llm":0.10,"unit":0.25,"proxy":0.10}
        s = sum(w.values())
    for k in w:
        w[k] = max(0.0, w[k]) / s
    return w

def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)

def _compute_contrib_cols(df: pd.DataFrame, w: dict) -> pd.DataFrame:
    # Required risk cols: uncertainty_risk, sc_risk, llm_risk, unit_risk, proxy_risk
    # If missing, create as 0.
    for col in ["uncertainty_risk","sc_risk","llm_risk","unit_risk","proxy_risk"]:
        if col not in df.columns:
            df[col] = 0.0

    unc = _to_float(df["uncertainty_risk"])
    sc  = _to_float(df["sc_risk"])
    llm = _to_float(df["llm_risk"])
    unit = _to_float(df["unit_risk"])
    proxy = _to_float(df["proxy_risk"])

    routed = df["route_llm"].astype(bool) if "route_llm" in df.columns else False

    df["contrib_unc"] = w["unc"] * unc
    df["contrib_sc"]  = w["sc"]  * sc
    df["contrib_llm"]= w["llm"] * llm * routed.astype(float)  # only count if routed
    df["contrib_unit"]= w["unit"] * unit
    df["contrib_proxy"]= w["proxy"] * proxy

    # Dominant contributor among the five (for each row)
    contrib_cols = ["contrib_unc","contrib_sc","contrib_llm","contrib_unit","contrib_proxy"]
    df["dominant"] = df[contrib_cols].idxmax(axis=1).map({
        "contrib_unc":"unc",
        "contrib_sc":"sc",
        "contrib_llm":"llm",
        "contrib_unit":"unit",
        "contrib_proxy":"proxy",
    })

    return df

def _report_one_group(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    blocked_rate = float(g["blocked"].mean()) if n else 0.0
    routed_rate = float(g["route_llm"].mean()) if ("route_llm" in g.columns and n) else 0.0
    llm_ran_rate = float(g["llm_ran"].mean()) if ("llm_ran" in g.columns and n) else 0.0

    mean_contrib = {
        "mean_contrib_unc": float(g["contrib_unc"].mean()) if n else 0.0,
        "mean_contrib_sc": float(g["contrib_sc"].mean()) if n else 0.0,
        "mean_contrib_llm": float(g["contrib_llm"].mean()) if n else 0.0,
        "mean_contrib_unit": float(g["contrib_unit"].mean()) if n else 0.0,
        "mean_contrib_proxy": float(g["contrib_proxy"].mean()) if n else 0.0,
    }

    # Dominant shares among BLOCKED only
    gb = g[g["blocked"] == True]
    nb = len(gb)
    dom = Counter(gb["dominant"].tolist())
    def pct(x): return 0.0 if nb == 0 else float(x) / nb

    dom_share = {
        "dom_unc": pct(dom.get("unc", 0)),
        "dom_sc": pct(dom.get("sc", 0)),
        "dom_llm": pct(dom.get("llm", 0)),
        "dom_unit": pct(dom.get("unit", 0)),
        "dom_proxy": pct(dom.get("proxy", 0)),
    }

    return pd.Series({
        "n": n,
        "blocked_rate": blocked_rate,
        "routed_rate": routed_rate,
        "llm_ran_rate": llm_ran_rate,
        **mean_contrib,
        **dom_share,
        "blocked_n": int(nb),
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_csv", required=True)
    ap.add_argument("--gt_compare", required=True)
    ap.add_argument("--out_dir", default="results")

    # Weights used in contribution report (MUST match eval_cases run for paper consistency)
    ap.add_argument("--w_unc", type=float, default=0.35)
    ap.add_argument("--w_sc", type=float, default=0.20)
    ap.add_argument("--w_llm", type=float, default=0.10)
    ap.add_argument("--w_unit", type=float, default=0.25)
    ap.add_argument("--w_proxy", type=float, default=0.10)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    det = pd.read_csv(args.det_csv)
    gt = _read_gt_compare(args.gt_compare)

    if "case_id" not in det.columns:
        raise RuntimeError("det_csv missing case_id")
    if "case_id" not in gt.columns or "status" not in gt.columns:
        raise RuntimeError("gt_compare must contain: case_id, status")

    merged = det.merge(gt[["case_id","status"]], on="case_id", how="left")
    merged["blocked"] = _resolve_blocked_column(merged)

    # Ensure routing columns exist
    for col in ["route_llm","llm_ran"]:
        if col not in merged.columns:
            merged[col] = False

    # Contribution columns
    w = _get_weights(det, args)
    merged = _compute_contrib_cols(merged, w)

    merged_path = os.path.join(args.out_dir, "merged_cases.csv")
    merged.to_csv(merged_path, index=False)

    report = (
        merged.groupby("status", dropna=False)
              .apply(_report_one_group)
              .reset_index()
              .sort_values("n", ascending=False)
    )

    # Add an overall row for convenience
    overall = _report_one_group(merged)
    overall["status"] = "OVERALL"
    report = pd.concat([pd.DataFrame([overall]), report], ignore_index=True)

    report_path = os.path.join(args.out_dir, "report_by_type.csv")
    report.to_csv(report_path, index=False)

    # CORRECT but blocked (FP analysis)
    cols = [
        "case_id","prompt","status","blocked",
        "risk_final","final_threshold","final_reason",
        "uncertainty_risk","sc_risk","llm_risk","unit_risk","proxy_risk",
        "contrib_unc","contrib_sc","contrib_llm","contrib_unit","contrib_proxy",
        "dominant","first_back_blocker",
        "route_llm","llm_ran","llm_verdict",
        "hard_fail","hard_reason",
    ]
    keep_cols = [c for c in cols if c in merged.columns]
    cbb = merged[(merged["status"] == "CORRECT") & (merged["blocked"] == True)][keep_cols].copy()
    cbb_path = os.path.join(args.out_dir, "correct_but_blocked.csv")
    cbb.to_csv(cbb_path, index=False)

    print("\n[OK] Wrote:")
    print(" -", merged_path)
    print(" -", report_path)
    print(" -", cbb_path)
    print("\n[INFO] Contribution weights (normalized):", w)
    print("\n[HEAD] report_by_type.csv")
    print(report.to_string(index=False))

if __name__ == "__main__":
    main()
