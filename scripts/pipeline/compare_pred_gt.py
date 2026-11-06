#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

IGNORE_EXTRA_KEYS_BY_FUNCTION = {
    "add_obstacle": {"note", "notes", "remark", "remarks", "comment", "comments", "memo"},
}

def extract_case_id(p: Path) -> str:
    m = re.search(r"(case_\d+)", p.stem)
    return m.group(1) if m else p.stem

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_gt_jsonl(gt_path: Path) -> Dict[str, Dict[str, Any]]:
    gt = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("case_id") or obj.get("id") or obj.get("case")
            if not cid:
                continue
            gt[cid] = obj
    return gt

def get_pred_calls(case_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    llm = case_obj.get("llm_output", case_obj)
    if isinstance(llm, dict) and "call" in llm:
        calls = llm["call"]
        if isinstance(calls, list):
            return calls
    
    if isinstance(llm, dict) and llm.get("type") == "list" and "call" in llm and isinstance(llm["call"], list):
        return llm["call"]
    return []

def normalize_call(c: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    fn = c.get("function_name") or c.get("function") or ""
    params = c.get("parameters") or c.get("arguments") or {}
    if params is None:
        params = {}
    if not isinstance(params, dict):
        params = {"_raw": params}
    return fn, params

def compare_params(fn: str, pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ignore = IGNORE_EXTRA_KEYS_BY_FUNCTION.get(fn, set())

    diffs = []
    for k, gv in gt.items():
        if k in ignore:
            continue
        if k not in pred:
            diffs.append(f"missing:{k}")
            continue
        pv = pred[k]

        if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
            if float(pv) != float(gv):
                diffs.append(f"diff:{k} pred={pv} gt={gv}")
        else:
            if pv != gv:
                diffs.append(f"diff:{k} pred={pv} gt={gv}")
    return (len(diffs) == 0), diffs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_dir", required=True)
    ap.add_argument("--gt_jsonl", required=True)
    ap.add_argument("--out_csv", default="results/gt_compare.csv")
    args = ap.parse_args()

    cases_dir = Path(args.cases_dir)
    gt_map = load_gt_jsonl(Path(args.gt_jsonl))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    correct = 0
    counts = {"missing_gt":0, "pred_missing":0, "call_count_mismatch":0, "function_mismatch":0, "param_mismatch":0}

    files = sorted([p for p in cases_dir.glob("case_*.json") if not p.name.endswith("_sc_runs.json")])
    for p in files:
        cid = extract_case_id(p)
        case_obj = load_json(p)
        pred_calls_raw = get_pred_calls(case_obj)
        pred_calls = [normalize_call(c) for c in pred_calls_raw]

        gt_obj = gt_map.get(cid)
        if not gt_obj:
            counts["missing_gt"] += 1
            rows.append([cid, "NO_GT", "", "", "", ""])
            continue

        gt_calls_raw = gt_obj.get("expected_calls") or gt_obj.get("calls") or []
        gt_calls = [normalize_call(c) for c in gt_calls_raw]

        if not pred_calls:
            counts["pred_missing"] += 1
            rows.append([cid, "PRED_EMPTY", "", "", "", ""])
            continue

        if len(pred_calls) != len(gt_calls):
            counts["call_count_mismatch"] += 1
            rows.append([cid, "CALL_COUNT_MISMATCH", len(pred_calls), len(gt_calls), "", ""])
            continue

        ok_all = True
        why = []
        for i, ((pfn, ppar), (gfn, gpar)) in enumerate(zip(pred_calls, gt_calls), start=1):
            if pfn != gfn:
                ok_all = False
                why.append(f"fn@{i}:{pfn}!={gfn}")
                continue
            okp, diffs = compare_params(gfn, ppar, gpar)
            if not okp:
                ok_all = False
                why.append(f"param@{i}:" + ";".join(diffs))

        if ok_all:
            correct += 1
            rows.append([cid, "CORRECT", len(pred_calls), len(gt_calls), "", ""])
        else:
            if any(w.startswith("fn@") for w in why):
                counts["function_mismatch"] += 1
                etype = "FUNCTION_MISMATCH"
            else:
                counts["param_mismatch"] += 1
                etype = "PARAM_MISMATCH"
            rows.append([cid, etype, len(pred_calls), len(gt_calls), " | ".join(why), ""])

    total = len(files)
    incorrect = total - correct - counts["missing_gt"]
    header = "case_id\tstatus\tpred_n\tgt_n\tdetails\tnote\n"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            f.write("\t".join(map(lambda x: str(x).replace("\t"," "), r)) + "\n")

    print("\n[GT COMPARE SUMMARY]")
    print(f"- Total cases: {total}")
    print(f"- GT missing: {counts['missing_gt']}")
    print(f"- Correct: {correct}")
    print(f"- Incorrect (excluding GT missing): {incorrect}")
    print("[Breakdown]")
    for k,v in counts.items():
        if k != "missing_gt":
            print(f"- {k}: {v}")
    print(f"[OK] Wrote: {out_csv}")

if __name__ == "__main__":
    main()
