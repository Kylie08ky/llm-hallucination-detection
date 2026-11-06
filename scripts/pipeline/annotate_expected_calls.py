#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

IGNORE_EXTRA_KEYS_BY_FUNCTION = {
    "add_obstacle": {"note", "notes", "remark", "remarks", "comment", "comments", "memo"},
}

def extract_case_id(path: Path) -> str:
    m = re.search(r"(case_\d+)", path.stem)
    return m.group(1) if m else path.stem

def sort_key_case_id(p: Path) -> Tuple[int, str]:
    cid = extract_case_id(p)
    m = re.search(r"case_(\d+)", cid)
    return (int(m.group(1)) if m else 10**9, cid)

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def load_existing_jsonl(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("case_id")
                if cid:
                    done.add(cid)
            except Exception:
                continue
    return done

def prompt_yes_no(msg: str) -> bool:
    while True:
        s = input(msg).strip().lower()
        if s in ("y","yes"): return True
        if s in ("n","no"): return False
        print("  Please enter y/yes or n/no")

def prompt_int(msg: str, lo: int, hi: int) -> int:
    while True:
        s = input(msg).strip()
        try:
            v = int(s)
        except:
            print("  Please enter an integer.")
            continue
        if lo <= v <= hi:
            return v
        print(f"  Please enter an integer in [{lo}, {hi}].")

def coerce_value(raw: str, schema: Dict[str, Any]) -> Any:
    raw = raw.strip()
    if raw == "":
        return None
    t = schema.get("type")
    if t == "integer":
        return int(float(raw))
    if t == "number":
        return float(raw)
    if t == "boolean":
        return raw.lower() in ("1","true","yes","y")
    return raw  # string/default

def load_tools(tools_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load tools from util/tools.
    Supports BOTH:
    1) { "name": "...", "parameters": {...} }
    2) { "type": "function", "function": { "name": "...", "parameters": {...} } }
    """
    tools: Dict[str, Dict[str, Any]] = {}

    if not tools_dir.exists():
        return tools

    for p in tools_dir.glob("*.json"):
        try:
            obj = read_json(p)
        except Exception:
            continue

        # ---- Case 1: OpenAI-style wrapper ----
        if (
            isinstance(obj, dict)
            and obj.get("type") == "function"
            and isinstance(obj.get("function"), dict)
            and "name" in obj["function"]
        ):
            fn = obj["function"]["name"]
            tools[fn] = obj["function"]
            continue

        # ---- Case 2: Flat single function ----
        if isinstance(obj, dict) and "name" in obj:
            tools[obj["name"]] = obj
            continue

        # ---- Case 3: list of functions ----
        if isinstance(obj, list):
            for t in obj:
                if (
                    isinstance(t, dict)
                    and t.get("type") == "function"
                    and isinstance(t.get("function"), dict)
                    and "name" in t["function"]
                ):
                    tools[t["function"]["name"]] = t["function"]
                elif isinstance(t, dict) and "name" in t:
                    tools[t["name"]] = t

    return tools


def pick_function(tools: Dict[str, Dict[str, Any]]) -> str:
    names = sorted(tools.keys())
    hint = "(Enter function name; '?' to list all; '/keyword' to filter; or enter index.)"
    while True:
        s = input(f"function_name {hint}: ").strip()
        if s == "?":
            print("Available functions:")
            for i, n in enumerate(names, 1):
                print(f"  {i:>3}. {n}")
            continue
        if s.startswith("/"):
            kw = s[1:].lower()
            filt = [n for n in names if kw in n.lower()]
            if not filt:
                print("  No matching function name found.")
                continue
            print("Matched functions:")
            for i, n in enumerate(filt, 1):
                print(f"  {i:>3}. {n}")
            c = input("Enter index to select (or press Enter to keep typing): ").strip()
            if c.isdigit():
                idx = int(c)
                if 1 <= idx <= len(filt):
                    return filt[idx-1]
            continue
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(names):
                return names[idx-1]
            print("  Index out of range.")
            continue
        if s in tools:
            return s
        print("  Function not found in registry. Use '?' to list or '/keyword' to filter.")

def ask_params_by_schema(fn: str, tool: Dict[str, Any]) -> Dict[str, Any]:
    params_schema = tool.get("parameters", {})
    if not isinstance(params_schema, dict):
        return {}

    props = params_schema.get("properties", {}) if isinstance(params_schema.get("properties"), dict) else {}
    required = params_schema.get("required", []) if isinstance(params_schema.get("required"), list) else []
    ignore = IGNORE_EXTRA_KEYS_BY_FUNCTION.get(fn, set())

    out: Dict[str, Any] = {}

    # required
    for k in required:
        sch = props.get(k, {})
        enum = sch.get("enum")
        if isinstance(enum, list) and enum:
            enum_str = [str(x) for x in enum]
            while True:
                v = input(f"  - {k} {enum_str}: ").strip()
                if v in enum_str:
                    out[k] = v
                    break
                print("    Please enter one value from the enum list.")
        else:
            while True:
                v = input(f"  - {k}: ").strip()
                val = coerce_value(v, sch)
                if val is None:
                    print("    Required field cannot be empty.")
                    continue
                out[k] = val
                break

    # optional (enter to skip)
    for k, sch in props.items():
        if k in out or k in required or k in ignore:
            continue
        enum = sch.get("enum")
        if isinstance(enum, list) and enum:
            enum_str = [str(x) for x in enum]
            v = input(f"  - {k} optional {enum_str} (Enter to skip): ").strip()
            if v == "":
                continue
            if v not in enum_str:
                print("    Not in enum list; skipped.")
                continue
            out[k] = v
        else:
            v = input(f"  - {k} (Enter to skip): ").strip()
            if v == "":
                continue
            out[k] = coerce_value(v, sch)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_dir", required=True)
    ap.add_argument("--tools_dir", default="util/tools")
    ap.add_argument("--out_jsonl", default="data/test/gt/gt.jsonl")
    args = ap.parse_args()

    cases_dir = Path(args.cases_dir)
    tools_dir = Path(args.tools_dir)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tools = load_tools(tools_dir)
    done = load_existing_jsonl(out_jsonl)

    files = [p for p in cases_dir.glob("case_*.json") if not p.name.endswith("_sc_runs.json")]
    files.sort(key=sort_key_case_id)

    print(f"[INFO] Found {len(files)} cases in {cases_dir}")
    print(f"[INFO] Tools loaded: {len(tools)} from {tools_dir}")
    print(f"[INFO] Output JSONL: {out_jsonl}")
    print(f"[INFO] Already labeled: {len(done)} (will skip)")

    for p in files:
        cid = extract_case_id(p)
        if cid in done:
            continue

        obj = read_json(p)
        prompt = obj.get("prompt", "")
        pred = obj.get("llm_output", {}) if isinstance(obj.get("llm_output", {}), dict) else {}

        print("\n" + "="*90)
        print(f"[CASE] {cid} ({p.name})")
        print("- NL prompt:")
        print(prompt)
        print("- Predicted action JSON:")
        print(pretty(pred))

        hallu = prompt_yes_no("[LABEL] Is there any hallucination? (y/n): ")

        # If no hallucination: accept the prediction directly (no extra input)
        if not hallu:
            rec = {
                "case_id": cid,
                "prompt": prompt,
                "hallucinated": False,
                "expected_calls": pred.get("call", []) if isinstance(pred, dict) else [],
            }
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(cid)
            print(f"[SAVED] {cid} (accepted prediction)")
            continue

        # If hallucination: manually label the correct function sequence + params
        print("\n[GROUND TRUTH] Please enter the correct function call sequence.")
        n = prompt_int("How many functions are expected for this prompt (0-20): ", 0, 20)

        expected_calls: List[Dict[str, Any]] = []
        for i in range(1, n+1):
            print(f"\n# {i}")
            fn = pick_function(tools)
            print("  -> Enter parameters field-by-field (required fields must be filled; optional: Enter to skip).")
            params = ask_params_by_schema(fn, tools[fn])
            expected_calls.append({"function_name": fn, "parameters": params})

        rec = {"case_id": cid, "prompt": prompt, "hallucinated": True, "expected_calls": expected_calls}
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done.add(cid)
        print(f"[SAVED] {cid}")

    print("\n[DONE] All cases processed.")

if __name__ == "__main__":
    main()
