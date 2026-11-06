#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep routing threshold thr_u for deciding when to CALL LLM under RISK-FUSION.

New policy:
  should_call_llm = (u_col >= thr_u)

Fusion-aware routed prediction:
  - For non-routed cases, LLM contribution is removed from fused risk:
        risk_final_routed = risk_final - contrib_llm   (if not routed)
        risk_final_routed = risk_final                 (if routed)
  - Then:
        blocked_routed = (risk_final_routed >= final_threshold)

This tool estimates:
- LLM call rate / cost proxy
- Resulting metrics under fusion logic if LLM only contributes on routed subset
- Recommended thr_u (max_f1 / min_calls_at_f1 / f1_at_callrate)

Expected columns in merged_cases.csv (post-gate rows):
- status (CORRECT vs others)
- hard_fail (optional; if exists, hard_fail==False are used)
- risk_final (float)          fused risk from your pipeline (with LLM included when it ran)
- final_threshold (float)     per-run threshold (optional; can override via arg)
- contrib_llm (float)        weighted LLM contribution included in risk_final
- u_score OR uncertainty_risk (routing score column)

If contrib_llm is missing:
- fallback to w_llm * llm_risk if both columns exist,
- otherwise treat contrib_llm as 0 (routing sweep will have no effect).

Usage:
  PYTHONPATH=src python scripts/eval/route_unc_sc_sweep.py \
    --merged results/dev/merged/merged_cases.csv \
    --out_csv results/dev/metrics/route_unc_sweep.csv \
    --u_col u_score \
    --thr_min 0.05 --thr_max 0.60 --thr_step 0.01 \
    --out_yaml config/routing.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def confusion(y_true: pd.Series, y_pred: pd.Series):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def metrics_from_cm(tp, fp, fn, tn):
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    return acc, precision, recall, f1


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def write_routing_yaml(path: Path, thr_u: float, u_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = f"""routing:
  thr_u: {thr_u:.6f}
  u_col: {u_col}
"""
    path.write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--u_col", default="", help="Routing score column. Prefer u_score; fallback uncertainty_risk.")
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.60)
    ap.add_argument("--thr_step", type=float, default=0.01)

    ap.add_argument("--final_threshold", type=float, default=None,
                    help="Override final_threshold. If not set, reads from merged csv column final_threshold.")
    ap.add_argument("--out_yaml", default="", help="Optional: write chosen thr_u into routing.yaml")

    ap.add_argument("--select", default="f1_at_30",
                    choices=["max_f1", "f1_at_30", "min_calls_at_f1"],
                    help="How to pick the recommended thr_u.")
    ap.add_argument("--target_f1", type=float, default=0.70, help="Used by min_calls_at_f1.")
    ap.add_argument("--call_rate_ref", type=float, default=0.30, help="Used by f1_at_30.")
    ap.add_argument("--call_rate_tol", type=float, default=0.08)

    args = ap.parse_args()

    df = pd.read_csv(args.merged)

    # drop hard_fail rows (post-gate evaluation)
    if "hard_fail" in df.columns:
        df = df[df["hard_fail"] == False].copy()

    if "status" not in df.columns:
        raise SystemExit("[ERROR] merged csv missing 'status' column.")
    if "risk_final" not in df.columns:
        raise SystemExit("[ERROR] merged csv missing 'risk_final' column (required for fusion sweep).")

    df["y_true"] = (df["status"] != "CORRECT").astype(int)

    # routing score column
    u_col = args.u_col.strip()
    if not u_col:
        u_col = pick_first_existing(df, ["u_score", "u", "uncertainty_risk"]) or ""
    if not u_col or u_col not in df.columns:
        raise SystemExit("[ERROR] cannot find routing score column. Provide --u_col (e.g., u_score).")

    df[u_col] = pd.to_numeric(df[u_col], errors="coerce").fillna(0.0)

    # final threshold
    if args.final_threshold is not None:
        final_thr = float(args.final_threshold)
    else:
        if "final_threshold" not in df.columns:
            raise SystemExit("[ERROR] final_threshold not provided and not found in merged csv.")
        final_thr = float(pd.to_numeric(df["final_threshold"], errors="coerce").iloc[0])

    # LLM contribution term that can be removed for non-routed cases
    if "contrib_llm" in df.columns:
        contrib_llm = pd.to_numeric(df["contrib_llm"], errors="coerce").fillna(0.0)
    else:
        # fallback: w_llm * llm_risk
        if ("w_llm" in df.columns) and ("llm_risk" in df.columns):
            contrib_llm = (
                pd.to_numeric(df["w_llm"], errors="coerce").fillna(0.0)
                * pd.to_numeric(df["llm_risk"], errors="coerce").fillna(0.0)
            )
        else:
            contrib_llm = pd.Series(0.0, index=df.index)

    risk_final = pd.to_numeric(df["risk_final"], errors="coerce").fillna(0.0)

    thrs = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    rows = []
    n = len(df)

    for thr_u in thrs:
        thr_u = float(thr_u)
        should_call = (df[u_col] >= thr_u)

        # fusion-aware routed risk: remove LLM contribution where not routed
        risk_routed = risk_final.copy()
        risk_routed.loc[~should_call] = (risk_final - contrib_llm).loc[~should_call]

        blocked_routed = (risk_routed >= final_thr)

        tp, fp, fn, tn = confusion(df["y_true"], blocked_routed.astype(int))
        acc, precision, recall, f1 = metrics_from_cm(tp, fp, fn, tn)
        call_rate = float(should_call.mean()) if n else 0.0

        rows.append({
            "thr_u": thr_u,
            "llm_call_rate": call_rate,
            "llm_calls": int(should_call.sum()),
            "n": n,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1,
            "final_threshold": final_thr,
            "u_col": u_col,
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("[OK] wrote", out_path)

    # pick recommended thr_u
    if args.select == "max_f1":
        best = max(rows, key=lambda r: r["F1"])
    elif args.select == "min_calls_at_f1":
        cand = [r for r in rows if r["F1"] >= args.target_f1]
        best = min(cand, key=lambda r: r["llm_calls"]) if cand else max(rows, key=lambda r: r["F1"])
    else:
        cand = [r for r in rows if abs(r["llm_call_rate"] - args.call_rate_ref) <= args.call_rate_tol]
        best = max(cand, key=lambda r: r["F1"]) if cand else max(rows, key=lambda r: r["F1"])

    print(
        f"[RECOMMEND] thr_u={best['thr_u']:.3f} "
        f"call_rate={best['llm_call_rate']:.3f} "
        f"F1={best['F1']:.4f} P={best['Precision']:.4f} R={best['Recall']:.4f} "
        f"(final_thr={final_thr:.3f}, u_col={u_col})"
    )

    if args.out_yaml:
        write_routing_yaml(Path(args.out_yaml), best["thr_u"], u_col)
        print("[OK] wrote", args.out_yaml)


if __name__ == "__main__":
    main()
