#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: eval_stability_from_sc_runs.py

Purpose:
    Re-evaluate multiple SC runs per case under fixed fusion parameters
    and export per-run and per-case stability statistics.

Inputs:
    - Base case JSON files
      (e.g. data/test/cases_json/case_*.json)
    - Corresponding self-consistency run files
      (e.g. data/test/cases_json/case_*_sc_runs.json)

Outputs:
    - stability_per_run.csv
    - stability_per_case.csv

Run:
    python plots/eval_stability_from_sc_runs.py \
        --split test \
        --final_threshold 0.12 \
        --out_dir results/test/stability

Generates:
    - Per-run stability records
    - Per-case stability statistics
      (decision consistency, risk_std, risk_range, disagreement_rate)

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.5 Stabilitäts- und Reproduzierbarkeitsvalidierung
    - Used as input for figures in Sections 7.5.2, 7.5.3, and 7.5.4
"""
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Utils
# -----------------------------
def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def extract_first_number(prompt: str) -> Optional[float]:
    """
    Extract the first number in prompt (supports "4", "4.5").
    """
    if not prompt:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", prompt)
    if not m:
        return None
    return safe_float(m.group(1))


def has_ambiguity_marker(prompt: str) -> bool:
    p = (prompt or "").lower()
    markers = [
        "a bit", "slightly", "somehow", "as usual", "do the usual", "around",
        "roughly", "approximately", "near", "close", "some", "few"
    ]
    return any(w in p for w in markers)


# -----------------------------
# Parse calls from sc_runs.json
# -----------------------------
def get_runs_list(sc_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports common layouts:
      sc_obj["runs"] -> list[run]
      sc_obj["run_outputs"] -> list[run]
    """
    if isinstance(sc_obj.get("runs"), list):
        return sc_obj["runs"]
    if isinstance(sc_obj.get("run_outputs"), list):
        return sc_obj["run_outputs"]
    return []


def extract_call_list(run_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Try to locate the tool call list in a run object.
    Supports patterns like:
      run_obj["call"] -> list
      run_obj["upstream_output"]["call"] -> list
      run_obj["output"]["call"] -> list
    """
    if isinstance(run_obj.get("call"), list):
        return run_obj["call"]

    for k in ["upstream_output", "output", "assistant_output", "result", "action_json", "llm_output"]:
        v = run_obj.get(k)
        if isinstance(v, dict) and isinstance(v.get("call"), list):
            return v["call"]

    # sometimes run_obj itself is a dict with {"type":"action","call":[...]}
    if isinstance(run_obj.get("type"), str) and isinstance(run_obj.get("call"), list):
        return run_obj["call"]

    return []


def normalize_signature(call_list: List[Dict[str, Any]]) -> str:
    """
    Build a stable signature describing the *structure* of calls:
      function_name + sorted parameter keys (not values)
    This is used to detect self-consistency disagreement across runs.
    """
    parts = []
    for c in call_list:
        fn = c.get("function_name")
        params = c.get("parameters")
        if not isinstance(fn, str):
            fn = "UNKNOWN"
        if not isinstance(params, dict):
            params = {}
        keys = ",".join(sorted([str(k) for k in params.keys()]))
        parts.append(f"{fn}({keys})")
    return " | ".join(parts) if parts else "EMPTY"


def infer_distance_from_call(call_list: List[Dict[str, Any]]) -> Optional[float]:
    """
    Try to infer a "distance-like" value from common navigation tools:
      - drive_direction(distance=...)
      - drive_distance(distance=...)
      - drive_coordinates(x=..., y=...)  -> use sqrt(x^2+y^2)
    If multiple calls exist, we take the first meaningful one.
    """
    for c in call_list:
        fn = c.get("function_name")
        params = c.get("parameters", {})
        if not isinstance(params, dict):
            params = {}

        if fn in ("drive_direction", "drive_distance"):
            d = safe_float(params.get("distance"))
            if d is not None:
                return d

        if fn == "drive_coordinates":
            x = safe_float(params.get("x"))
            y = safe_float(params.get("y"))
            if x is not None and y is not None:
                return math.sqrt(x * x + y * y)

    return None


# -----------------------------
# Risks (true-content-based)
# -----------------------------
def compute_uncertainty_risk(base_case: Dict[str, Any]) -> float:
    v = safe_float(base_case.get("gen_uncertainty", 0.0))
    return clip01(v if v is not None else 0.0)


def compute_sc_risk(signature: str, majority_signature: str) -> float:
    """
    sc_risk reflects whether this run deviates from the majority structure.
    """
    return 0.0 if signature == majority_signature else 1.0


def compute_unit_risk(prompt_num: Optional[float], inferred_dist: Optional[float]) -> float:
    """
    unit_risk: if prompt includes a numeric target and the run's inferred distance
    deviates strongly (relative error > 25%) or sign mismatch, raise risk.
    """
    if prompt_num is None or inferred_dist is None:
        return 0.0

    # sign mismatch (e.g., expects +4 but got -4 encoded)
    if prompt_num >= 0 and inferred_dist < 0:
        return 1.0

    denom = max(1.0, abs(prompt_num))
    rel_err = abs(inferred_dist - prompt_num) / denom
    return 1.0 if rel_err > 0.25 else 0.0


def compute_proxy_risk(prompt: str, disagreement_rate: float, unc_risk: float) -> float:
    """
    proxy_risk = blend(semantic ambiguity, multi-run structural disagreement, unc_risk)
    - ambiguity marker -> higher
    - disagreement_rate = 1 - maxfreq/num_runs
    """
    amb = 0.65 if has_ambiguity_marker(prompt) else 0.20
    proxy = 0.40 * amb + 0.40 * disagreement_rate + 0.20 * unc_risk
    return clip01(proxy)


def fuse_risk(unc: float, sc: float, unit: float, proxy: float,
              w_unc: float, w_sc: float, w_unit: float, w_proxy: float) -> float:
    return clip01(w_unc * unc + w_sc * sc + w_unit * unit + w_proxy * proxy)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--cases_dir", default=None, help="override cases_json dir")
    ap.add_argument("--out_dir", default=None, help="override output dir")
    ap.add_argument("--final_threshold", type=float, default=0.12)

    # frozen weights (adjust if your paper uses different ones)
    ap.add_argument("--w_unc", type=float, default=0.30)
    ap.add_argument("--w_sc", type=float, default=0.20)
    ap.add_argument("--w_unit", type=float, default=0.20)
    ap.add_argument("--w_proxy", type=float, default=0.30)

    args = ap.parse_args()

    cases_dir = Path(args.cases_dir) if args.cases_dir else Path(f"data/{args.split}/cases_json")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"results/{args.split}/stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows: List[Dict[str, Any]] = []
    per_case_rows: List[Dict[str, Any]] = []

    case_files = sorted([p for p in cases_dir.glob("case_*.json") if not p.name.endswith("_sc_runs.json")])

    for case_path in case_files:
        case_id = case_path.stem
        sc_path = cases_dir / f"{case_id}_sc_runs.json"
        if not sc_path.exists():
            continue

        base_case = read_json(case_path)
        sc_obj = read_json(sc_path)

        prompt = base_case.get("prompt", "") or ""
        prompt_num = extract_first_number(prompt)
        unc_risk = compute_uncertainty_risk(base_case)

        runs = get_runs_list(sc_obj)
        if not runs:
            continue

        # Build signatures and majority signature (self-consistency reference)
        run_calls: List[List[Dict[str, Any]]] = []
        run_sigs: List[str] = []
        for r in runs:
            calls = extract_call_list(r if isinstance(r, dict) else {})
            run_calls.append(calls)
            run_sigs.append(normalize_signature(calls))

        sig_counts = Counter(run_sigs)
        majority_sig, majority_freq = sig_counts.most_common(1)[0]
        disagreement_rate = 1.0 - (majority_freq / max(1, len(run_sigs)))

        # case-level proxy risk (same for all runs, depends on disagreement + ambiguity)
        proxy_risk = compute_proxy_risk(prompt, disagreement_rate, unc_risk)

        # Evaluate each run
        risk_values: List[float] = []
        blocked_values: List[bool] = []

        for idx, (calls, sig) in enumerate(zip(run_calls, run_sigs)):
            inferred_dist = infer_distance_from_call(calls)

            sc_risk = compute_sc_risk(sig, majority_sig)
            unit_risk = compute_unit_risk(prompt_num, inferred_dist)

            risk_final = fuse_risk(
                unc=unc_risk,
                sc=sc_risk,
                unit=unit_risk,
                proxy=proxy_risk,
                w_unc=args.w_unc,
                w_sc=args.w_sc,
                w_unit=args.w_unit,
                w_proxy=args.w_proxy,
            )
            blocked = (risk_final >= args.final_threshold)

            risk_values.append(risk_final)
            blocked_values.append(blocked)

            per_run_rows.append({
                "case_id": case_id,
                "run_index": idx,
                "signature": sig,
                "majority_signature": majority_sig,
                "unc_risk": unc_risk,
                "proxy_risk": proxy_risk,
                "sc_risk": sc_risk,
                "unit_risk": unit_risk,
                "risk_final": risk_final,
                "blocked": int(blocked),
            })

        # Per-case stability stats
        num_runs = len(risk_values)
        if num_runs == 0:
            continue

        # decision_consistency: majority proportion of blocked/pass
        n_block = sum(1 for b in blocked_values if b)
        n_pass = num_runs - n_block
        decision_consistency = max(n_block, n_pass) / num_runs

        risk_std = statistics.pstdev(risk_values) if num_runs > 1 else 0.0
        risk_range = (max(risk_values) - min(risk_values)) if num_runs > 1 else 0.0

        per_case_rows.append({
            "case_id": case_id,
            "num_runs": num_runs,
            "majority_signature_freq": majority_freq,
            "disagreement_rate": disagreement_rate,
            "decision_consistency": decision_consistency,
            "risk_mean": sum(risk_values) / num_runs,
            "risk_std": risk_std,
            "risk_range": risk_range,
            "threshold": args.final_threshold,
        })

    # Write CSVs
    run_csv = out_dir / "stability_per_run.csv"
    case_csv = out_dir / "stability_per_case.csv"

    if per_run_rows:
        with run_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
            w.writeheader()
            w.writerows(per_run_rows)

    if per_case_rows:
        with case_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
            w.writeheader()
            w.writerows(per_case_rows)

    print("wrote:", run_csv)
    print("wrote:", case_csv)
    print("cases evaluated:", len(per_case_rows))


if __name__ == "__main__":
    main()