#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: export_gate_fail_cases.py

Purpose:
    Export typical hard-fail / gate-fail cases from results_frozen.csv
    into CSV and Markdown format for manual inspection and figure selection.

Inputs:
    - Full evaluation result CSV
      (e.g. results/test/det/results_frozen.csv)

Outputs:
    - gate_fail_cases.csv
    - gate_fail_cases.md

Run:
    python plots/export_gate_fail_cases.py \
        --results_csv results/test/det/results_frozen.csv \
        --out_dir results/test/analysis_gate \
        --max_cases 50

Generates:
    - CSV file of gate-fail samples
    - Markdown file for selecting and screenshotting representative examples

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.3.1 Analyse des strukturellen Abfangverhaltens der Gate-Layer
    - Used to prepare example cases for:
      JSON parsing failure, invalid type, and out-of-range parameter examples
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_json_loads(x: str) -> Optional[Any]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def extract_raw_from_parameters_list(params_obj: Any) -> Optional[str]:
    if isinstance(params_obj, list):
        for item in params_obj:
            if isinstance(item, dict) and "raw" in item:
                raw = item.get("raw")
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--out_dir", default="results/test/analysis_gate")
    ap.add_argument("--max_cases", type=int, default=50)
    args = ap.parse_args()

    results_csv = Path(args.results_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(results_csv)

    if "hard_fail" not in df.columns:
        raise SystemExit("cannot find hard_fail row. Make sure uploading results_frozen.csv.")

    gate_df = df[df["hard_fail"] == True].copy()
    if gate_df.empty:
        raise SystemExit("There is no case that hard_fail==True. Please check there exist Gate Fail cases during this generation.")

    gate_df = gate_df.head(args.max_cases)

    need_cols = [
        "case_id",
        "prompt",
        "hard_reason",
        "functions",
        "parameters_list",
        "source_file",
    ]
    for c in need_cols:
        if c not in gate_df.columns:
            gate_df[c] = ""

    rows: List[Dict[str, Any]] = []

    for _, r in gate_df.iterrows():
        params_obj = safe_json_loads(r.get("parameters_list", ""))

        raw_text = extract_raw_from_parameters_list(params_obj)

        rows.append(
            {
                "case_id": str(r.get("case_id", "")),
                "prompt": str(r.get("prompt", "")),
                "hard_reason": str(r.get("hard_reason", "")),
                "functions": str(r.get("functions", "")),
                "source_file": str(r.get("source_file", "")),
                "parameters_list_pretty": pretty_json(params_obj) if params_obj is not None else str(r.get("parameters_list", "")),
                "raw_output": raw_text or "",
            }
        )

    out_csv = out_dir / "gate_fail_cases.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    out_md = out_dir / "gate_fail_cases.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Gate Fail Cases (hard_fail == True)\n\n")
        f.write(f"- Source: `{results_csv}`\n")
        f.write(f"- Count: {len(rows)}\n\n")

        for i, x in enumerate(rows, 1):
            f.write(f"## {i}. {x['case_id']}\n\n")

            f.write("**Prompt**:\n\n")
            f.write(x["prompt"].strip() + "\n\n")

            f.write("**Hard fail reason**:\n\n")
            f.write(f"`{x['hard_reason']}`\n\n")

            f.write("**Functions (parsed)**:\n\n")
            f.write(f"`{x['functions']}`\n\n")

            if x["raw_output"]:
                f.write("**Raw generated output (from parameters_list.raw)**:\n\n")
                f.write("```text\n")
                f.write(x["raw_output"] + "\n")
                f.write("```\n\n")
            else:
                f.write("**Parsed parameters_list**:\n\n")
                f.write("```json\n")
                f.write(x["parameters_list_pretty"] + "\n")
                f.write("```\n\n")

            if x["source_file"] and x["source_file"].lower() != "nan":
                f.write(f"**Source case file**: `{x['source_file']}`\n\n")

    print(f"Gate fail cases: {len(rows)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()