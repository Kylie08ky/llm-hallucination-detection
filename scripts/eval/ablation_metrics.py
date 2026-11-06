#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion-style ablation metrics for hallucination detection.

This script evaluates multiple "systems" under the SAME fusion decision rule:
  risk_final = w_unc*unc + w_sc*sc + w_llm*llm + w_unit*unit + w_proxy*proxy
  blocked   = (risk_final >= final_threshold)

It supports LLM selective routing:
- If route_to_llm exists, LLM risk contributes only when route_to_llm==1.
- Otherwise LLM risk is used as-is.

Expected (default) risk columns in merged CSV:
- uncertainty_risk, sc_risk, llm_risk, unit_risk, proxy_risk
Expected label column:
- status (CORRECT means y_true=0; others mean y_true=1)

Usage:
PYTHONPATH=src python scripts/eval/ablation_metrics.py \
  --merged results/dev/merged/merged_cases.csv \
  --out_csv results/dev/metrics/ablation_metrics.csv \
  --final_threshold 0.12 \
  --w_unc 0.30 \
  --w_sc 0.20 \
  --w_llm 0.20 \
  --w_unit 0.25 \
  --w_proxy 0.05

If you pass merged_cases_routed.csv that already has route_to_llm applied,
the script will still be correct (it will apply routing again only if route_col exists).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def confusion(y_true: pd.Series, y_pred: pd.Series) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def metrics_from_cm(tp: int, fp: int, fn: int, tn: int) -> Tuple[float, float, float, float]:
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    return precision, recall, f1, acc


@dataclass(frozen=True)
class Weights:
    w_unc: float
    w_sc: float
    w_llm: float
    w_unit: float
    w_proxy: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "w_unc": self.w_unc,
            "w_sc": self.w_sc,
            "w_llm": self.w_llm,
            "w_unit": self.w_unit,
            "w_proxy": self.w_proxy,
        }

    def with_overrides(self, **kwargs) -> "Weights":
        d = self.as_dict()
        for k, v in kwargs.items():
            if k not in d:
                raise KeyError(f"Unknown weight key: {k}")
            d[k] = float(v)
        return Weights(**d)  # type: ignore


def safe_col(df: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    if col not in df.columns:
        # Missing column -> treat as zeros
        return pd.Series([fill] * len(df), index=df.index, dtype="float64")
    s = df[col]
    # Coerce numeric if possible
    s = pd.to_numeric(s, errors="coerce").fillna(fill)
    return s


def compute_risk_final(
    df: pd.DataFrame,
    weights: Weights,
    unc_col: str,
    sc_col: str,
    llm_col: str,
    unit_col: str,
    proxy_col: str,
    route_col: Optional[str] = "route_to_llm",
) -> Tuple[pd.Series, Optional[float]]:
    unc = safe_col(df, unc_col, 0.0)
    sc = safe_col(df, sc_col, 0.0)
    llm = safe_col(df, llm_col, 0.0)
    unit = safe_col(df, unit_col, 0.0)
    proxy = safe_col(df, proxy_col, 0.0)

    call_rate = None
    if route_col and route_col in df.columns:
        route = df[route_col].astype(int)
        # Apply routing gate to LLM contribution
        llm = llm * route
        call_rate = float(route.mean())

    risk_final = (
        weights.w_unc * unc
        + weights.w_sc * sc
        + weights.w_llm * llm
        + weights.w_unit * unit
        + weights.w_proxy * proxy
    )
    return risk_final, call_rate


def build_systems(base: Weights) -> List[Tuple[str, Weights]]:
    """
    Fusion-style ablations that match your current narrative:
    - Unc is main signal
    - Unit is structural error signal
    - SC is auxiliary
    - LLM is enhancer for routed subset
    - Proxy is small
    """
    return [
        ("FULL", base),
        ("UNC_ONLY", base.with_overrides(w_sc=0, w_llm=0, w_unit=0, w_proxy=0)),
        ("UNIT_ONLY", base.with_overrides(w_unc=0, w_sc=0, w_llm=0, w_proxy=0)),
        ("UNC+UNIT", base.with_overrides(w_sc=0, w_llm=0, w_proxy=0)),
        ("UNC+UNIT+SC", base.with_overrides(w_llm=0, w_proxy=0)),
        ("UNC+UNIT+LLM", base.with_overrides(w_sc=0, w_proxy=0)),
        ("UNC+UNIT+SC+LLM", base.with_overrides(w_proxy=0)),
        ("NO_LLM", base.with_overrides(w_llm=0)),     # Full minus LLM
        ("NO_PROXY", base.with_overrides(w_proxy=0)),   # Full minus Proxy
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="Input merged_cases.csv or merged_cases_routed.csv")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--final_threshold", type=float, required=True)

    # Base weights (for FULL)
    ap.add_argument("--w_unc", type=float, required=True)
    ap.add_argument("--w_sc", type=float, required=True)
    ap.add_argument("--w_llm", type=float, required=True)
    ap.add_argument("--w_unit", type=float, required=True)
    ap.add_argument("--w_proxy", type=float, required=True)

    # Column names (override if your merged schema differs)
    ap.add_argument("--status_col", default="status")
    ap.add_argument("--hard_fail_col", default="hard_fail")
    ap.add_argument("--route_col", default="route_to_llm")

    ap.add_argument("--unc_col", default="uncertainty_risk")
    ap.add_argument("--sc_col", default="sc_risk")
    ap.add_argument("--llm_col", default="llm_risk")
    ap.add_argument("--unit_col", default="unit_risk")
    ap.add_argument("--proxy_col", default="proxy_risk")

    args = ap.parse_args()

    merged_path = Path(args.merged)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_path)

    # Post-gate filter (keep behavior consistent with your current pipeline)
    if args.hard_fail_col in df.columns:
        df = df[df[args.hard_fail_col] == False].copy()

    # y_true: non-CORRECT means hallucination/incorrect => positive(1)
    if args.status_col not in df.columns:
        raise KeyError(f"Missing label column '{args.status_col}' in {merged_path}")
    y_true = (df[args.status_col] != "CORRECT").astype(int)

    base = Weights(
        w_unc=args.w_unc,
        w_sc=args.w_sc,
        w_llm=args.w_llm,
        w_unit=args.w_unit,
        w_proxy=args.w_proxy,
    )
    systems = build_systems(base)

    rows: List[Dict[str, object]] = []
    for name, w in systems:
        risk_final, call_rate = compute_risk_final(
            df=df,
            weights=w,
            unc_col=args.unc_col,
            sc_col=args.sc_col,
            llm_col=args.llm_col,
            unit_col=args.unit_col,
            proxy_col=args.proxy_col,
            route_col=args.route_col if args.route_col else None,
        )
        y_pred = (risk_final >= float(args.final_threshold)).astype(int)

        tp, fp, fn, tn = confusion(y_true, y_pred)
        precision, recall, f1, acc = metrics_from_cm(tp, fp, fn, tn)

        row: Dict[str, object] = {
            "system": name,
            "final_threshold": float(args.final_threshold),
            **w.as_dict(),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "risk_final_mean": float(pd.to_numeric(risk_final, errors="coerce").fillna(0.0).mean()),
        }
        # Only meaningful when route_col exists and system actually uses LLM weight
        if call_rate is not None:
            row["llm_call_rate"] = call_rate
        else:
            row["llm_call_rate"] = None

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print("[OK] wrote", out_path)


if __name__ == "__main__":
    main()
