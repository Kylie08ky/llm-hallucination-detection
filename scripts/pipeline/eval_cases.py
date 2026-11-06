#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_cases.py (LLM-dominant + Param(Unit) evidence + risk-fusion summary)

FIX (2026-02):
- LLM verdict consistency: verdict=pass ONLY IF all checks are supported.
- Any missing/ambiguous/unsupported check => verdict=fail.
- If LLM verdict=fail, lift llm_risk to at least a floor (default 0.60) to avoid "fail but tiny risk".
- Do NOT treat llm_raw_error (summary) as an error indicator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import RequestException


# -----------------------------
# IO helpers
# -----------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "case_id","prompt","decision_mode",
        "final_valid","blocked","risk_final","final_threshold","first_blocker","first_back_blocker",
        "hard_fail","hard_reason",
        "pipeline",
        "functions","parameters_list",

        "uncertainty_risk","uncertainty_details",

        "proxy_ran","proxy_risk","proxy_threshold","proxy_block","proxy_details",

        "unit_risk","unit_details",

        "sc_enabled","sc_metric","sc_ran","sc_missing","sc_n","sc_agreement",
        "sc_risk","sc_threshold","sc_hard_threshold","sc_uncertainty_threshold","sc_block",
        "sc_key_agreement","sc_value_risk",

        "u_score","thr_u","route_llm",

        "llm_ran","llm_verdict","llm_risk","llm_risk_threshold","llm_block","llm_failed_checks",
        "llm_error","llm_raw_error",

        "final_reason","model","source_file",
    ]
    all_keys = set().union(*(r.keys() for r in rows))
    fieldnames = [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Tool schema loading
# -----------------------------

@dataclass
class ToolSpec:
    name: str
    required: List[str]
    properties: Dict[str, Any]
    description: str = ""

def load_tool_specs_from_dir(tools_dir: Path) -> Dict[str, ToolSpec]:
    specs: Dict[str, ToolSpec] = {}
    if not tools_dir.exists():
        return specs

    for p in sorted(tools_dir.glob("*.json")):
        try:
            obj = read_json(p)
        except Exception:
            continue

        fn = obj.get("function") if isinstance(obj, dict) else None
        if not isinstance(fn, dict):
            continue

        name = fn.get("name")
        params = fn.get("parameters")
        if not isinstance(name, str) or not isinstance(params, dict):
            continue

        props = params.get("properties", {})
        req = params.get("required", [])
        if not isinstance(props, dict):
            props = {}
        if not isinstance(req, list):
            req = []

        specs[name] = ToolSpec(
            name=name,
            required=[str(x) for x in req],
            properties=props,
            description=str(fn.get("description", "")),
        )
    return specs

def tools_to_prompt_text(specs: Dict[str, ToolSpec], max_distance_m: float, max_radius_m: float) -> str:
    lines = ["Available tools (MUST obey exactly):"]
    for name, spec in specs.items():
        lines.append(f"- {name}")
        if spec.required:
            lines.append(f"  required: {spec.required}")
        if spec.properties:
            lines.append(f"  properties: {list(spec.properties.keys())}")
    lines.append("")
    lines.append("Hard constraints (must be enforced):")
    lines.append("- drive_direction.direction ∈ {left,right,forward,backward}")
    lines.append("- drive_direction.distance must be >= 0")
    lines.append(f"- drive_direction.distance must be <= {max_distance_m} (meters)")
    lines.append("- set_radius.radius must be > 0")
    lines.append(f"- set_radius.radius must be <= {max_radius_m} (meters)")
    lines.append("- drive_coordinates/calculate_path/add_obstacle require numeric x,y")
    lines.append("- drive_direction/calculate_path/add_obstacle/set_radius params must match schema types")
    return "\n".join(lines)


# -----------------------------
# Case filtering
# -----------------------------

_CASE_ID_RE = re.compile(r"(\d{3})")

def case_id_to_int(case_stem: str) -> Optional[int]:
    m = _CASE_ID_RE.search(case_stem or "")
    return int(m.group(1)) if m else None

def parse_case_range(case_range: str) -> Optional[Tuple[int, int]]:
    if not case_range:
        return None
    try:
        a, b = case_range.split("-")
        lo, hi = int(a), int(b)
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    except Exception:
        return None

def parse_case_list(case_list: str) -> Optional[set]:
    if not case_list:
        return None
    out = set()
    for part in case_list.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            m = _CASE_ID_RE.search(part)
            if m:
                out.add(int(m.group(1)))
    return out if out else None

def in_filter(case_stem: str, lo_hi: Optional[Tuple[int, int]], include_set: Optional[set]) -> bool:
    cid = case_id_to_int(case_stem)
    if cid is None:
        return False
    if include_set is not None:
        return cid in include_set
    if lo_hi is None:
        return True
    lo, hi = lo_hi
    return lo <= cid <= hi


# -----------------------------
# Robust action extraction
# -----------------------------

def extract_json_from_text(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    t = re.sub(r"```json", "```", text, flags=re.IGNORECASE)
    t = t.replace("```", "").strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None

def extract_action_container(case_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = ["upstream_output","output","llm_output","action_json","result","assistant_output"]
    for k in candidates:
        v = case_obj.get(k)
        if isinstance(v, dict) and "call" in v:
            return v

    v = case_obj.get("call")
    if isinstance(v, dict) and "call" in v:
        return v
    if isinstance(v, list):
        return {"type": "list", "call": v}

    raw = case_obj.get("raw_output") or case_obj.get("raw") or case_obj.get("content")
    if isinstance(raw, str):
        obj = extract_json_from_text(raw)
        if isinstance(obj, dict) and "call" in obj:
            return obj
    return None

@dataclass
class ParsedCall:
    ok: bool
    function_name: str
    parameters: Dict[str, Any]
    error: str = ""

def parse_calls(action_container: Optional[Dict[str, Any]]) -> Tuple[bool, List[ParsedCall], str]:
    if not isinstance(action_container, dict):
        return False, [], "missing_action_container"
    calls = action_container.get("call")
    if not isinstance(calls, list) or not calls:
        return False, [], "missing_or_empty_call_list"
    out: List[ParsedCall] = []
    for i, c in enumerate(calls):
        if not isinstance(c, dict):
            out.append(ParsedCall(False, "ERROR", {}, error=f"call_{i}_not_dict"))
            continue
        fn = c.get("function_name")
        params = c.get("parameters", {})
        if not isinstance(fn, str) or not fn:
            out.append(ParsedCall(False, "ERROR", {}, error=f"call_{i}_missing_function_name"))
            continue
        if not isinstance(params, dict):
            params = {"_raw": params}
        if fn.upper() == "ERROR":
            out.append(ParsedCall(False, "ERROR", params, error=str(params.get("error","upstream_error"))))
            continue
        out.append(ParsedCall(True, fn, params, error=""))
    return True, out, ""


# -----------------------------
# Gate checks (hard)
# -----------------------------

def is_number(x: Any) -> bool:
    return safe_float(x) is not None

def gate_check_call(
    call: ParsedCall,
    tool_specs: Dict[str, ToolSpec],
    max_distance_m: float,
    max_radius_m: float,
) -> Tuple[bool, str]:
    if not call.ok:
        return False, f"parse_fail:{call.error}"
    if call.function_name not in tool_specs:
        return False, f"unknown_function:{call.function_name}"

    spec = tool_specs[call.function_name]
    params = call.parameters or {}

    missing = [k for k in spec.required if k not in params]
    if missing:
        return False, f"missing_required_params:{missing}"

    for key, prop in spec.properties.items():
        if key not in params:
            continue
        declared_type = str(prop.get("type", "")).lower()
        val = params.get(key)
        if declared_type in ("float","number","int","integer"):
            if not is_number(val):
                return False, f"invalid_type:{key}:expected_number"
        if declared_type == "string":
            if not isinstance(val, str):
                return False, f"invalid_type:{key}:expected_string"

    if call.function_name == "drive_direction":
        direction = params.get("direction")
        if not isinstance(direction, str) or direction not in ["left","right","forward","backward"]:
            return False, "invalid_parameter:direction_enum"
        if "distance" in params:
            dist = safe_float(params.get("distance"))
            if dist is None:
                return False, "invalid_parameter:distance_not_number"
            if dist < 0:
                return False, "invalid_parameter:distance_must_be_ge_0"
            if dist > float(max_distance_m):
                return False, f"invalid_parameter:distance_too_large>{max_distance_m}"

    if call.function_name in ("drive_coordinates","calculate_path","add_obstacle"):
        x = safe_float(params.get("x"))
        y = safe_float(params.get("y"))
        if x is None or y is None:
            return False, "invalid_type:x_or_y_expected_number"

    if call.function_name == "set_radius":
        r = safe_float(params.get("radius"))
        if r is None:
            return False, "invalid_parameter:radius_not_number"
        if r <= 0:
            return False, "invalid_parameter:radius_must_be_gt_0"
        if r > float(max_radius_m):
            return False, f"invalid_parameter:radius_too_large>{max_radius_m}"

    return True, ""


# -----------------------------
# Proxy (soft evidence)
# -----------------------------

def proxy_uncertainty(case_obj: Dict[str, Any], unc_risk: Optional[float] = None) -> Tuple[float, str]:
    prompt = str(case_obj.get("prompt") or "")
    low = prompt.lower()
    ambiguous_markers = [
        "a bit","slightly","somehow","do the usual","as usual",
        "handle the situation","there","it","usual spot",
        "safe path","avoid the obstacle","proceed","a little",
        "around","roughly","approximately","close to","near",
    ]
    amb = 0.65 if any(w in low for w in ambiguous_markers) else 0.20

    if unc_risk is None:
        return amb, "heuristic_only"
    proxy = 0.35 * float(amb) + 0.65 * float(unc_risk)
    proxy = max(0.0, min(1.0, proxy))
    return proxy, f"amb={amb:.2f}+unc={float(unc_risk):.3f}"


# -----------------------------
# Param(Unit) evidence
# -----------------------------

_UNIT_FACTORS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "millimetre": 0.001, "millimetres": 0.001,
    "dm": 0.1, "decimeter": 0.1, "decimeters": 0.1, "decimetre": 0.1, "decimetres": 0.1,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
}

_RE_NUM_UNIT = re.compile(
    r"(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|dm|km|m|meter|meters|metre|metres|ft|feet|foot)\b",
    flags=re.IGNORECASE
)

_DISTANCE_KEYS = {
    "distance", "radius", "r", "range", "length", "height", "width",
    "x", "y", "z", "dx", "dy", "dz"
}

def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(1e-6, abs(b))

def unit_consistency_risk(prompt: str, calls: List[ParsedCall]) -> Tuple[float, str]:
    if not prompt:
        return 0.0, "no_prompt"

    low = prompt.lower()
    matches = list(_RE_NUM_UNIT.finditer(low))
    if not matches:
        return 0.0, "no_unit_pattern"

    mentions: List[Tuple[float, str, float]] = []
    for m in matches:
        num = safe_float(m.group("num"))
        unit = (m.group("unit") or "").lower()
        if num is None:
            continue
        factor = _UNIT_FACTORS.get(unit)
        if factor is None:
            continue
        mentions.append((float(num), unit, float(factor)))

    if not mentions:
        return 0.0, "no_valid_mentions"

    cand_params: List[Tuple[str, str, float]] = []
    for c in calls:
        if not c.ok:
            continue
        if not isinstance(c.parameters, dict):
            continue
        for k, v in c.parameters.items():
            if str(k).lower() not in _DISTANCE_KEYS:
                continue
            fv = safe_float(v)
            if fv is None:
                continue
            cand_params.append((c.function_name, str(k), float(fv)))

    if not cand_params:
        return 0.0, "no_candidate_params"

    best_risk = 0.0
    best_detail = "no_issue"

    for num, unit, factor in mentions:
        if abs(factor - 1.0) < 1e-9:
            continue

        converted = num * factor
        for fn, key, v in cand_params:
            if _rel_err(v, num) <= 0.05:
                if _rel_err(v, converted) >= 0.30:
                    scale_gap = abs(math.log10(max(1e-9, abs(num))) - math.log10(max(1e-9, abs(converted))))
                    strength = min(1.0, 0.7 + 0.3 * min(1.0, scale_gap / 2.0))
                    risk = max(0.85, strength)
                    if risk > best_risk:
                        best_risk = risk
                        best_detail = (
                            f"unit_mismatch: prompt '{num:g}{unit}' -> {converted:g}m, "
                            f"but {fn}.{key}={v:g} (≈raw {num:g})"
                        )

    return float(best_risk), best_detail


# -----------------------------
# SC helpers
# -----------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def load_sc_runs(sc_path: Path) -> List[Dict[str, Any]]:
    try:
        obj = read_json(sc_path)
    except Exception:
        return []
    if isinstance(obj, dict) and isinstance(obj.get("runs"), list):
        return [r for r in obj["runs"] if isinstance(r, dict)]
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    return []

def _mean_pairwise_jaccard(sets: List[set]) -> float:
    n = len(sets)
    if n <= 1:
        return 1.0
    s = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            inter = len(a & b)
            union = len(a | b)
            s += (inter / union) if union else 1.0
            cnt += 1
    return s / max(cnt, 1)

def _extract_call_list_from_action_container(ac: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    call = ac.get("call")
    if not isinstance(call, list) or not call:
        return None
    out: List[Dict[str, Any]] = []
    for item in call:
        if not isinstance(item, dict):
            continue
        fn = item.get("function_name")
        params = item.get("parameters", {})
        if not isinstance(fn, str) or not fn:
            continue
        if not isinstance(params, dict):
            params = {"_raw": params}
        out.append({"function_name": fn, "parameters": params})
    return out if out else None

def _keyset_for_call_list(call_list: List[Dict[str, Any]]) -> set:
    ks = set()
    ks.add(f"n_calls={len(call_list)}")
    for i, call in enumerate(call_list):
        fn = str(call.get("function_name", ""))
        ks.add(f"call[{i}].fn={fn}")
        params = call.get("parameters") or {}
        if isinstance(params, dict):
            for k in params.keys():
                ks.add(f"call[{i}].{fn}.{k}")
        else:
            ks.add(f"call[{i}].{fn}._raw")
    return ks

def _value_stability_risk(call_lists: List[List[Dict[str, Any]]]) -> float:
    vals_num: Dict[str, List[float]] = {}
    vals_cat: Dict[str, List[str]] = {}

    def _add_num(key: str, v: float):
        vals_num.setdefault(key, []).append(v)

    def _add_cat(key: str, v: str):
        vals_cat.setdefault(key, []).append(v)

    for calls in call_lists:
        for i, call in enumerate(calls):
            fn = str(call.get("function_name", ""))
            params = call.get("parameters") or {}
            if not isinstance(params, dict):
                continue
            for k, v in params.items():
                field = f"call[{i}].{fn}.{k}"
                fv = safe_float(v)
                if fv is not None:
                    _add_num(field, float(fv))
                else:
                    _add_cat(field, str(v))

    risks: List[float] = []

    for xs in vals_num.values():
        if len(xs) <= 1:
            continue
        med = statistics.median(xs)
        mad = statistics.median([abs(x - med) for x in xs])
        rel = mad / (abs(med) + 1e-6)
        risks.append(max(0.0, min(1.0, rel / 0.20)))

    for xs in vals_cat.values():
        if len(xs) <= 1:
            continue
        c = Counter(xs)
        mode_cnt = c.most_common(1)[0][1]
        mode_frac = mode_cnt / max(1, len(xs))
        risks.append(1.0 - mode_frac)

    return max(risks) if risks else 0.0

def sc_metrics(
    case_file: Path,
    sc_metric: str = "exact",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not isinstance(case_file, Path):
        return 0, None, None, None, None
    sc_path = case_file.with_name(case_file.stem + "_sc_runs.json")
    if not sc_path.exists():
        return 0, None, None, None, None

    runs = load_sc_runs(sc_path)

    call_lists: List[List[Dict[str, Any]]] = []
    texts: List[str] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        ac = extract_action_container(r) if isinstance(r, dict) else None
        if ac is None and isinstance(r, dict) and "call" in r:
            ac = r
        if not isinstance(ac, dict):
            continue

        cl = _extract_call_list_from_action_container(ac)
        if cl is None:
            continue

        call_lists.append(cl)
        texts.append(canonical_json(ac.get("call")))

    n = len(call_lists)
    if n == 0:
        return 0, None, None, None, None

    if sc_metric == "exact":
        counts: Dict[str, int] = {}
        for t in texts:
            counts[t] = counts.get(t, 0) + 1
        maj = max(counts.values())
        base_agreement = maj / n
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(sbert_model)
        emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        import numpy as np
        X = np.asarray(emb, dtype=float)
        if X.shape[0] <= 1:
            base_agreement = 1.0
        else:
            norms = (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            Xn = X / norms
            C = Xn @ Xn.T
            s = 0.0
            cnt = 0
            for i in range(Xn.shape[0]):
                for j in range(i + 1, Xn.shape[0]):
                    s += float(C[i, j]); cnt += 1
            base_agreement = s / max(cnt, 1)
        base_agreement = max(0.0, min(1.0, float(base_agreement)))

    base_risk = 1.0 - float(base_agreement)

    key_sets = [_keyset_for_call_list(cl) for cl in call_lists]
    key_agreement = float(_mean_pairwise_jaccard(key_sets))
    key_risk = 1.0 - key_agreement

    value_risk = float(_value_stability_risk(call_lists))

    risk_new = 0.50 * base_risk + 0.30 * key_risk + 0.20 * value_risk
    risk_new = max(0.0, min(1.0, float(risk_new)))
    return n, float(base_agreement), float(risk_new), float(key_agreement), float(value_risk)


# -----------------------------
# LLM (Ollama OpenAI-compatible)
# -----------------------------

def _normalize_ollama_root(ollama_url: str) -> str:
    base = (ollama_url or "").strip().strip('"').strip("'")
    if base.startswith("[") and base.endswith("]"):
        base = base[1:-1].strip()
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base

def ollama_chat_completions(root_url: str, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    url = root_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
        "response_format": {"type": "json_object"},
        "format": "json",
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def build_llm_messages(prompt: str, action_container: Dict[str, Any], tools_text: str) -> List[Dict[str, str]]:
    action_text = json.dumps(action_container, ensure_ascii=False, indent=2)

    system = (
        "You are a STRICT verifier for a tool-calling system.\n"
        "Your job is to check whether the ACTION JSON is supported by the USER PROMPT.\n"
        "IMPORTANT:\n"
        "- You MUST NOT assume missing details.\n"
        "- If the prompt is underspecified but the JSON invents details, mark those fields unsupported.\n"
        "- If you are unsure, mark ambiguous.\n"
        "- Return ONLY valid JSON (no markdown, no extra text).\n\n"
        "Return JSON with EXACT schema:\n"
        "{\n"
        '  "verdict": "pass" | "fail",\n'
        '  "checks": [\n'
        "    {\n"
        '      "path": "call[0].function_name" | "call[0].parameters.<key>" | ...,\n'
        '      "status": "supported" | "unsupported" | "ambiguous",\n'
        '      "evidence": ["<quote or short pointer>", "..."],\n'
        '      "reason": "short reason"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "one-line summary"\n'
        "}\n\n"
        "You must evaluate ALL calls and ALL parameter fields present in the JSON.\n\n"
        f"{tools_text}\n"
    )

    user = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"ACTION JSON:\n{action_text}\n\n"
        "Instruction:\n"
        "- For each call[i], create a check for function_name.\n"
        "- For each parameters key, create a check for that key.\n"
        "- Verdict = pass ONLY IF all checks are supported.\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def extract_json_obj_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    t = re.sub(r"```json", "```", text, flags=re.IGNORECASE)
    t = t.replace("```", "").strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def extract_llm_payload(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if not isinstance(content, str):
            return None
        return extract_json_obj_from_text(content)
    except Exception:
        return None

def _normalize_check_item(x: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(x, dict):
        return None
    path = x.get("path")
    status = x.get("status")
    evidence = x.get("evidence", [])
    reason = x.get("reason", "")
    if not isinstance(path, str) or not path:
        return None
    if status not in ("supported", "unsupported", "ambiguous"):
        status = "ambiguous"
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(e) for e in evidence if isinstance(e, (str, int, float))][:5]
    if not isinstance(reason, str):
        reason = str(reason)
    return {"path": path, "status": status, "evidence": evidence, "reason": reason[:200]}

def _score_llm_checks(
    action_container: Dict[str, Any],
    parsed: Dict[str, Any],
    tool_specs: Dict[str, ToolSpec],
) -> Tuple[str, float, List[str]]:
    expected_items: List[Tuple[str, str, Optional[str]]] = []
    calls = action_container.get("call")
    if isinstance(calls, list):
        for i, c in enumerate(calls):
            if not isinstance(c, dict):
                continue
            fn = str(c.get("function_name", ""))
            if not fn:
                continue
            expected_items.append((f"call[{i}].function_name", fn, None))
            params = c.get("parameters", {})
            if isinstance(params, dict):
                for k in params.keys():
                    expected_items.append((f"call[{i}].parameters.{k}", fn, str(k)))

    if not expected_items:
        return "fail", 1.0, ["missing_action_paths"]

    checks_raw = parsed.get("checks", [])
    if not isinstance(checks_raw, list):
        return "fail", 1.0, ["missing_checks_list"]

    by_path: Dict[str, Dict[str, Any]] = {}
    for item in checks_raw:
        ci = _normalize_check_item(item)
        if ci is None:
            continue
        if ci["path"] not in by_path:
            by_path[ci["path"]] = ci

    def _schema_allows_param(fn: str, key: str) -> bool:
        spec = tool_specs.get(fn)
        if spec is None:
            return False
        return key in (spec.properties or {}) or key in (spec.required or [])

    def _penalty(pth: str, status: str) -> float:
        if status == "missing_check":
            return 0.20
        if status == "ambiguous":
            return 0.40
        if status == "unsupported":
            return 1.0 if pth.endswith(".function_name") else 0.80
        return 0.0

    failed: List[str] = []
    penalty = 0.0
    total = 0

    for pth, fn, key in expected_items:
        total += 1
        ci = by_path.get(pth)
        if ci is None:
            penalty += _penalty(pth, "missing_check")
            failed.append(f"{pth}:missing_check")
            continue

        status = str(ci.get("status", "ambiguous"))

        if status == "unsupported" and key is not None and _schema_allows_param(fn, key):
            status = "ambiguous"

        if status in ("unsupported", "ambiguous"):
            penalty += _penalty(pth, status)
            failed.append(f"{pth}:{status}")

    risk = penalty / max(1, total)
    verdict = "pass" if len(failed) == 0 else "fail"
    return verdict, float(max(0.0, min(1.0, risk))), failed[:50]

def run_llm(
    prompt: str,
    action_container: Dict[str, Any],
    model: str,
    ollama_url: str,
    tools_text: str,
    tool_specs: Dict[str, ToolSpec],
) -> Dict[str, Any]:
    root = _normalize_ollama_root(ollama_url)
    try:
        resp = ollama_chat_completions(root, model, build_llm_messages(prompt, action_container, tools_text))
    except (RequestException, ValueError, OSError) as e:
        return {"error": True, "verdict": "error", "failed_checks": ["transport_error"], "risk": 1.0, "raw_error": str(e)}

    parsed = extract_llm_payload(resp)
    if not isinstance(parsed, dict):
        return {"error": False, "verdict": "fail", "failed_checks": ["llm_unparseable"], "risk": 1.0, "raw_error": "llm_unparseable"}

    verdict, risk, failed_paths = _score_llm_checks(action_container, parsed, tool_specs)

    summary = parsed.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    return {
        "error": False,
        "verdict": verdict,
        "failed_checks": failed_paths,
        "risk": float(risk),
        "raw_error": summary[:200],  # summary only, NOT an error flag
    }


# -----------------------------
# Pipeline string
# -----------------------------

def build_pipeline_string(
    gate_ok: bool,
    proxy_ran: bool,
    proxy_block: bool,
    sc_enabled: bool,
    sc_ran: bool,
    sc_missing: bool,
    sc_block: bool,
    route_llm: bool,
    llm_ran: bool,
    llm_block: bool,
    llm_error: bool,
) -> str:
    s = "GATE✓" if gate_ok else "GATE✗"
    if not gate_ok:
        return s

    s += " > PROXY" + ("✗" if proxy_block else "✓") if proxy_ran else " > PROXY(SKIP)"

    if not sc_enabled:
        s += " > SC(OFF)"
    else:
        if sc_missing:
            s += " > SC(MISSING)"
        elif not sc_ran:
            s += " > SC(SKIP)"
        else:
            s += " > SC✗" if sc_block else " > SC✓"

    s += " > ROUTE" + ("✓" if route_llm else "✗")

    if not llm_ran:
        s += " > LLM(SKIP)"
    else:
        if llm_error:
            s += " > LLM(ERR)"
        else:
            s += " > LLM✗" if llm_block else " > LLM✓"

    return s


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tools_dir", type=str, default="util/tools")

    ap.add_argument("--case_range", type=str, default="", help='run only range: "1-15" or "001-015"')
    ap.add_argument("--case_list", type=str, default="", help='run only set: "1,2,5,10" (overrides --case_range)')

    ap.add_argument("--decision_mode", type=str, default="back_only", choices=["deployment","hybrid","back_only"])

    ap.add_argument("--max_distance_m", type=float, default=50.0)
    ap.add_argument("--max_radius_m", type=float, default=10.0)

    # Proxy
    ap.add_argument("--proxy_threshold", type=float, default=0.6)
    ap.add_argument("--proxy_metric", type=str, default="heuristic", choices=["heuristic","local","upstream_eq"])
    ap.add_argument("--proxy_local_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    # SC
    ap.add_argument("--sc", action="store_true")
    ap.add_argument("--sc_threshold", type=float, default=0.5)
    ap.add_argument("--sc_hard_threshold", type=float, default=0.90)
    ap.add_argument("--sc_metric", type=str, default="exact", choices=["exact","sbert"])
    ap.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sc_uncertainty_threshold", type=float, default=0.05)

    # Routing & final decision
    ap.add_argument("--thr_u", type=float, default=0.30, help="Routing threshold on u_score to call LLM.")
    ap.add_argument("--final_threshold", type=float, default=0.25, help="Final blocking threshold on risk_final.")
    ap.add_argument("--w_proxy", type=float, default=0.10)
    ap.add_argument("--w_unit", type=float, default=0.25)
    ap.add_argument("--w_unc", type=float, default=0.35)
    ap.add_argument("--w_sc", type=float, default=0.20)
    ap.add_argument("--w_llm", type=float, default=0.10)

    # How much unit evidence affects routing (NEW)
    ap.add_argument("--unit_route_weight", type=float, default=0.20, help="Adds unit_risk into u_score for routing.")

    # LLM
    ap.add_argument("--disable_llm", action="store_true")
    ap.add_argument("--llm_model", type=str, default="qwen2.5:3b")
    ap.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--llm_risk_threshold", type=float, default=0.3)

    #lift LLM risk when verdict==fail to ensure it contributes
    ap.add_argument("--llm_fail_risk_floor", type=float, default=0.60)

    args = ap.parse_args()

    cases_dir = Path(args.cases_dir)
    out_csv = Path(args.out_csv)
    tools_dir = Path(args.tools_dir)

    if not cases_dir.exists():
        print(f"[ERROR] cases_dir not found: {cases_dir}")
        sys.exit(1)

    tool_specs = load_tool_specs_from_dir(tools_dir)
    if not tool_specs:
        print(f"[ERROR] No tool specs loaded from: {tools_dir}")
        sys.exit(1)

    tools_text = tools_to_prompt_text(tool_specs, float(args.max_distance_m), float(args.max_radius_m))

    lo_hi = parse_case_range(args.case_range)
    include_set = parse_case_list(args.case_list)

    case_files = sorted(cases_dir.glob("case_*.json"))
    case_files = [p for p in case_files if not p.name.endswith("_sc_runs.json")]
    case_files = [p for p in case_files if in_filter(p.stem, lo_hi, include_set)]

    # -----------------------------
    # Uncertainty normalization bounds (robust per-batch percentiles)
    # -----------------------------
    def _percentile(xs: List[float], q: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        if len(ys) == 1:
            return float(ys[0])
        q = max(0.0, min(1.0, float(q)))
        pos = q * (len(ys) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ys[lo])
        w = pos - lo
        return float(ys[lo] * (1.0 - w) + ys[hi] * w)

    nll_vals: List[float] = []
    ent_vals: List[float] = []
    for cf in case_files:
        try:
            obj = read_json(cf)
        except Exception:
            continue
        u = obj.get("gen_uncertainty", {})
        if not isinstance(u, dict):
            continue
        nll = safe_float(u.get("avg_nll"))
        ent = safe_float(u.get("avg_entropy"))
        if nll is not None:
            nll_vals.append(float(nll))
        if ent is not None:
            ent_vals.append(float(ent))

    if nll_vals and ent_vals:
        nll_min, nll_max = _percentile(nll_vals, 0.05), _percentile(nll_vals, 0.95)
        ent_min, ent_max = _percentile(ent_vals, 0.05), _percentile(ent_vals, 0.95)
        if nll_max <= nll_min:
            nll_min, nll_max = min(nll_vals), max(nll_vals)
        if ent_max <= ent_min:
            ent_min, ent_max = min(ent_vals), max(ent_vals)
    else:
        nll_min, nll_max = 0.0, 5.0
        ent_min, ent_max = 0.0, 5.0

    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def _minmax(v: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin:
            return 0.0
        return _clip01((float(v) - float(vmin)) / (float(vmax) - float(vmin)))

    # normalize weights
    def _norm_weights() -> Tuple[float, float, float, float, float]:
        ws = [max(0.0, float(args.w_proxy)), max(0.0, float(args.w_unit)), max(0.0, float(args.w_unc)),
              max(0.0, float(args.w_sc)), max(0.0, float(args.w_llm))]
        s = sum(ws)
        if s <= 1e-9:
            return 0.1, 0.2, 0.35, 0.25, 0.1
        return (ws[0]/s, ws[1]/s, ws[2]/s, ws[3]/s, ws[4]/s)

    w_proxy, w_unit, w_unc, w_sc, w_llm = _norm_weights()

    def _risk_final(proxy_r: float, unit_r: float, unc_r: float, sc_r: float, llm_r: float, routed: bool) -> float:
        c = float(llm_r) if routed else 0.0
        rf = (
            w_proxy * float(proxy_r) +
            w_unit  * float(unit_r) +
            w_unc   * float(unc_r) +
            w_sc    * float(sc_r) +
            w_llm  * float(c)
        )
        return _clip01(rf)

    rows: List[Dict[str, Any]] = []

    for cf in case_files:
        row: Dict[str, Any] = {
            "case_id": cf.stem,
            "prompt": "",
            "decision_mode": args.decision_mode,

            "final_valid": False,
            "blocked": True,
            "risk_final": "",
            "final_threshold": float(args.final_threshold),
            "first_blocker": "gate",
            "first_back_blocker": "none",

            "hard_fail": False,
            "hard_reason": "",
            "pipeline": "",

            "functions": "",
            "parameters_list": "",

            "uncertainty_risk": "",
            "uncertainty_details": "",

            "proxy_ran": False,
            "proxy_risk": "",
            "proxy_threshold": float(args.proxy_threshold),
            "proxy_block": False,
            "proxy_details": "",

            "unit_risk": "",
            "unit_details": "",

            "sc_enabled": bool(args.sc),
            "sc_metric": args.sc_metric if args.sc else "",
            "sc_ran": False,
            "sc_missing": False,
            "sc_n": "",
            "sc_agreement": "",
            "sc_key_agreement": "",
            "sc_value_risk": "",
            "sc_risk": "",
            "sc_threshold": float(args.sc_threshold),
            "sc_hard_threshold": float(args.sc_hard_threshold),
            "sc_uncertainty_threshold": float(args.sc_uncertainty_threshold),
            "sc_block": False,

            "u_score": "",
            "thr_u": float(args.thr_u),
            "route_llm": False,

            "llm_ran": False,
            "llm_verdict": "",
            "llm_risk": "",
            "llm_risk_threshold": float(args.llm_risk_threshold),
            "llm_block": False,
            "llm_failed_checks": "",
            "llm_error": False,
            "llm_raw_error": "",

            "final_reason": "",
            "model": "",
            "source_file": str(cf),
        }

        try:
            case_obj = read_json(cf)
        except Exception as e:
            row["hard_fail"] = True
            row["hard_reason"] = f"case_json_read_error:{e}"
            row["final_reason"] = row["hard_reason"]
            row["pipeline"] = "GATE✗"
            rows.append(row)
            continue

        prompt = str(case_obj.get("prompt") or "")
        model = str(case_obj.get("model") or "")
        row["prompt"] = prompt
        row["model"] = model

        action_container = extract_action_container(case_obj)
        ok_container, calls, container_reason = parse_calls(action_container)
        if not ok_container:
            row["hard_fail"] = True
            row["hard_reason"] = f"gate:{container_reason}"
            row["final_reason"] = row["hard_reason"]
            row["pipeline"] = "GATE✗"
            rows.append(row)
            continue

        row["functions"] = "|".join(c.function_name for c in calls)
        row["parameters_list"] = json.dumps([c.parameters for c in calls], ensure_ascii=False)

        gate_ok = True
        gate_reason = ""
        for idx, c in enumerate(calls):
            ok, reason = gate_check_call(c, tool_specs, float(args.max_distance_m), float(args.max_radius_m))
            if not ok:
                gate_ok = False
                gate_reason = f"call_{idx}:{reason}"
                break

        if not gate_ok:
            row["hard_fail"] = True
            row["hard_reason"] = f"gate:{gate_reason}"
            row["final_reason"] = row["hard_reason"]
            row["pipeline"] = "GATE✗"
            rows.append(row)
            continue

        # -----------------------------
        # Uncertainty risk (0..1)
        # -----------------------------
        unc_risk = 0.0
        unc_details = ""
        u0 = case_obj.get("gen_uncertainty", {})
        avg_nll0 = safe_float(u0.get("avg_nll")) if isinstance(u0, dict) else None
        avg_ent0 = safe_float(u0.get("avg_entropy")) if isinstance(u0, dict) else None
        gen_tokens0 = u0.get("gen_tokens", "") if isinstance(u0, dict) else ""
        if avg_nll0 is not None and avg_ent0 is not None:
            norm_nll0 = _minmax(avg_nll0, nll_min, nll_max)
            norm_ent0 = _minmax(avg_ent0, ent_min, ent_max)
            unc_risk = _clip01(0.5 * norm_nll0 + 0.5 * norm_ent0)
            unc_details = (
                f"upstream_eq nll={avg_nll0:.3f} ent={avg_ent0:.3f} "
                f"norm_nll={norm_nll0:.3f} norm_ent={norm_ent0:.3f} "
                f"bounds(nll:{nll_min:.3f}-{nll_max:.3f},ent:{ent_min:.3f}-{ent_max:.3f}) tokens={gen_tokens0}"
            )
        else:
            unc_risk = 0.0
            unc_details = "missing_gen_uncertainty->unc=0"

        row["uncertainty_risk"] = f"{float(unc_risk):.3f}"
        row["uncertainty_details"] = unc_details

        # -----------------------------
        # Proxy (soft evidence)
        # -----------------------------
        proxy_risk = 0.0
        proxy_details = ""
        if args.proxy_metric == "local":
            from fa_eval.uncertainty_hf import score_action_uncertainty
            action_text = json.dumps(action_container, ensure_ascii=False)
            r = score_action_uncertainty(
                prompt=prompt, tools_text=tools_text, action_json_text=action_text,
                model_name=str(args.proxy_local_model)
            )
            proxy_risk = _clip01(float(r.risk))
            proxy_details = str(r.details)
        elif args.proxy_metric == "upstream_eq":
            u = case_obj.get("gen_uncertainty", {})
            avg_nll = safe_float(u.get("avg_nll")) if isinstance(u, dict) else None
            avg_ent = safe_float(u.get("avg_entropy")) if isinstance(u, dict) else None
            gen_tokens = u.get("gen_tokens", "") if isinstance(u, dict) else ""
            if avg_nll is None or avg_ent is None:
                proxy_risk, proxy_details = proxy_uncertainty(case_obj, unc_risk)
                proxy_risk = _clip01(float(proxy_risk))
            else:
                norm_nll = _minmax(avg_nll, nll_min, nll_max)
                norm_ent = _minmax(avg_ent, ent_min, ent_max)
                proxy_risk = _clip01(0.5 * norm_nll + 0.5 * norm_ent)
                proxy_details = (
                    f"upstream_eq nll={avg_nll:.3f} ent={avg_ent:.3f} "
                    f"norm_nll={norm_nll:.3f} norm_ent={norm_ent:.3f} tokens={gen_tokens}"
                )
        else:
            proxy_risk, proxy_details = proxy_uncertainty(case_obj, unc_risk)
            proxy_risk = _clip01(float(proxy_risk))

        row["proxy_ran"] = True
        row["proxy_risk"] = f"{float(proxy_risk):.3f}"
        row["proxy_details"] = proxy_details

        proxy_block = float(proxy_risk) > float(args.proxy_threshold)
        row["proxy_block"] = bool(proxy_block)

        # -----------------------------
        # Unit (param) evidence
        # -----------------------------
        unit_risk, unit_details = unit_consistency_risk(prompt, calls)
        unit_risk = _clip01(float(unit_risk))
        row["unit_risk"] = f"{float(unit_risk):.3f}"
        row["unit_details"] = unit_details[:300]

        # -----------------------------
        # SC
        # -----------------------------
        sc_enabled = bool(args.sc)
        sc_ran = False
        sc_missing = False
        sc_n = None
        sc_agreement = None
        sc_risk = 0.0
        sc_block = False
        sc_key_agreement = None
        sc_value_risk = None

        if sc_enabled:
            sc_n, sc_agreement, sc_risk_tmp, sc_key_agreement, sc_value_risk = sc_metrics(
                cf, sc_metric=args.sc_metric, sbert_model=args.sbert_model
            )
            row["sc_key_agreement"] = "" if sc_key_agreement is None else f"{float(sc_key_agreement):.3f}"
            row["sc_value_risk"] = "" if sc_value_risk is None else f"{float(sc_value_risk):.3f}"

            if sc_n == 0 and sc_agreement is None and sc_risk_tmp is None:
                sc_missing = True
                sc_risk = 0.0
            else:
                sc_ran = True
                sc_risk = _clip01(float(sc_risk_tmp)) if sc_risk_tmp is not None else 0.0
                if float(sc_risk) >= float(args.sc_hard_threshold):
                    sc_block = True
                elif float(sc_risk) > float(args.sc_threshold):
                    if float(unc_risk) >= float(args.sc_uncertainty_threshold):
                        sc_block = True

        row["sc_ran"] = sc_ran
        row["sc_missing"] = sc_missing
        row["sc_n"] = "" if sc_n is None else sc_n
        row["sc_agreement"] = "" if sc_agreement is None else f"{float(sc_agreement):.3f}"
        row["sc_risk"] = f"{float(sc_risk):.3f}"
        row["sc_block"] = bool(sc_block)

        # -----------------------------
        # Routing u_score includes unit evidence
        # -----------------------------
        u_score = _clip01(
            w_unc * float(unc_risk) +
            w_sc  * float(sc_risk) +
            w_proxy * float(proxy_risk) +
            float(args.unit_route_weight) * float(unit_risk)
        )
        route_llm = (u_score >= float(args.thr_u)) and (not args.disable_llm)
        row["u_score"] = f"{float(u_score):.3f}"
        row["route_llm"] = bool(route_llm)

        # -----------------------------
        # LLM (only if routed)
        # -----------------------------
        llm_ran = False
        llm_error = False
        llm_verdict = ""
        llm_risk = 0.0
        llm_failed_checks: List[str] = []
        llm_raw_error = ""
        llm_block = False

        if route_llm:
            llm = run_llm(
                prompt, action_container,
                str(args.llm_model), str(args.ollama_url),
                tools_text, tool_specs
            )
            llm_ran = True
            llm_error = bool(llm.get("error", False))
            llm_verdict = str(llm.get("verdict", ""))
            cr = safe_float(llm.get("risk"))
            llm_risk = _clip01(float(cr)) if cr is not None else 1.0
            fc = llm.get("failed_checks", [])
            if isinstance(fc, list):
                llm_failed_checks = [str(x) for x in fc]
            llm_raw_error = str(llm.get("raw_error", ""))

            if llm_error:
                llm_verdict = "error"
                llm_risk = 1.0
            elif llm_verdict.lower() != "pass":
                llm_risk = max(float(llm_risk), float(args.llm_fail_risk_floor))

            if llm_error:
                llm_block = True
            else:
                llm_block = float(llm_risk) > float(args.llm_risk_threshold)

        row["llm_ran"] = bool(llm_ran)
        row["llm_error"] = bool(llm_error)
        row["llm_verdict"] = llm_verdict
        row["llm_risk"] = f"{float(llm_risk):.3f}"
        row["llm_failed_checks"] = "|".join(llm_failed_checks)
        row["llm_raw_error"] = llm_raw_error[:200]
        row["llm_block"] = bool(llm_block)

        # -----------------------------
        # FINAL decision (risk-fusion)
        # -----------------------------
        risk_final = _risk_final(
            proxy_r=proxy_risk,
            unit_r=unit_risk,
            unc_r=unc_risk,
            sc_r=sc_risk,
            llm_r=llm_risk,
            routed=bool(route_llm),
        )
        row["risk_final"] = f"{float(risk_final):.3f}"

        blocked = (float(risk_final) >= float(args.final_threshold))
        row["blocked"] = bool(blocked)
        row["final_valid"] = not bool(blocked)

        contrib = {
            "unc":  w_unc   * float(unc_risk),
            "sc":   w_sc    * float(sc_risk),
            "llm": (w_llm * float(llm_risk)) if bool(route_llm) else 0.0,
            "proxy": w_proxy * float(proxy_risk),
            "unit": w_unit  * float(unit_risk),
        }
        first_back = max(contrib.items(), key=lambda kv: kv[1])[0] if contrib else "none"
        row["first_back_blocker"] = first_back if blocked else "none"

        if blocked:
            row["first_blocker"] = first_back
            row["final_reason"] = f"blocked:risk_final={risk_final:.3f}>=thr={float(args.final_threshold):.2f}"
        else:
            row["first_blocker"] = "none"
            row["final_reason"] = f"pass:risk_final={risk_final:.3f}<thr={float(args.final_threshold):.2f}"

        row["pipeline"] = build_pipeline_string(
            gate_ok=True,
            proxy_ran=True,
            proxy_block=bool(proxy_block),
            sc_enabled=sc_enabled,
            sc_ran=sc_ran,
            sc_missing=sc_missing,
            sc_block=bool(sc_block),
            route_llm=bool(route_llm),
            llm_ran=bool(llm_ran),
            llm_block=bool(llm_block),
            llm_error=bool(llm_error),
        )

        rows.append(row)

    write_csv(out_csv, rows)
    print(f"[OK] Wrote: {out_csv} ({len(rows)} rows)")

    # -----------------------------
    # Summary
    # -----------------------------
    def _as_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    total = len(rows)
    hard_fail_n = sum(1 for r in rows if _as_bool(r.get("hard_fail")))
    eligible = [r for r in rows if not _as_bool(r.get("hard_fail"))]
    eligible_n = len(eligible)

    blocked_rows = [r for r in eligible if _as_bool(r.get("blocked"))]
    blocked_n = len(blocked_rows)

    routed_n = sum(1 for r in eligible if _as_bool(r.get("route_llm")))
    llm_ran_n = sum(1 for r in eligible if _as_bool(r.get("llm_ran")))

    dom_counts = Counter()
    for r in blocked_rows:
        dom_counts[str(r.get("first_back_blocker","none"))] += 1

    def _mean_contrib(subset: List[Dict[str, Any]]) -> Dict[str, float]:
        if not subset:
            return {k: 0.0 for k in ["unc","sc","llm","proxy","unit"]}
        out = {}
        for k in ["unc","sc","llm","proxy","unit"]:
            vals = []
            for r in subset:
                unc = _as_float(r.get("uncertainty_risk"))
                sc  = _as_float(r.get("sc_risk"))
                llm= _as_float(r.get("llm_risk"))
                pr  = _as_float(r.get("proxy_risk"))
                ur  = _as_float(r.get("unit_risk"))
                routed = _as_bool(r.get("route_llm"))
                if k == "unc":   vals.append(w_unc * unc)
                if k == "sc":    vals.append(w_sc * sc)
                if k == "llm":  vals.append((w_llm * llm) if routed else 0.0)
                if k == "proxy": vals.append(w_proxy * pr)
                if k == "unit":  vals.append(w_unit * ur)
            out[k] = float(sum(vals) / max(1, len(vals)))
        return out

    mean_all = _mean_contrib(eligible)
    mean_blk = _mean_contrib(blocked_rows)

    def _pct(n: int, d: int) -> str:
        return "0.0%" if d <= 0 else f"{(100.0 * n / d):.1f}%"

    print("\n" + "=" * 80)
    print("[SUMMARY (risk-fusion)]")
    print(f"- Total cases:              {total}")
    print(f"- Hard fails (Gate):        {hard_fail_n}  ({_pct(hard_fail_n, total)})")
    print(f"- Eligible (post-gate):     {eligible_n}  ({_pct(eligible_n, total)})")
    print(f"- Blocked (risk_final):     {blocked_n}  ({_pct(blocked_n, eligible_n)})")
    print(f"- Routed to LLM:           {routed_n}  ({_pct(routed_n, eligible_n)})")
    print(f"- LLM ran:                 {llm_ran_n}  ({_pct(llm_ran_n, eligible_n)})")
    print("")
    print("- Dominant contributor among BLOCKED:")
    for k in ["unc","sc","llm","unit","proxy"]:
        print(f"  * {k:5s}: {dom_counts.get(k,0):3d}  ({_pct(dom_counts.get(k,0), blocked_n)})")
    other = blocked_n - sum(dom_counts.get(k,0) for k in ["unc","sc","llm","unit","proxy"])
    if other > 0:
        print(f"  * other: {other:3d}  ({_pct(other, blocked_n)})")

    print("")
    print("- Mean weighted contribution (ELIGIBLE):")
    print(f"  unc={mean_all['unc']:.3f}  sc={mean_all['sc']:.3f}  llm={mean_all['llm']:.3f}  unit={mean_all['unit']:.3f}  proxy={mean_all['proxy']:.3f}")
    print("- Mean weighted contribution (BLOCKED):")
    print(f"  unc={mean_blk['unc']:.3f}  sc={mean_blk['sc']:.3f}  llm={mean_blk['llm']:.3f}  unit={mean_blk['unit']:.3f}  proxy={mean_blk['proxy']:.3f}")
    print("")
    print(f"- Weights (normalized): w_unc={w_unc:.3f} w_sc={w_sc:.3f} w_llm={w_llm:.3f} w_unit={w_unit:.3f} w_proxy={w_proxy:.3f}")
    print(f"- Routing: thr_u={float(args.thr_u):.3f}  unit_route_weight={float(args.unit_route_weight):.3f}  final_threshold={float(args.final_threshold):.3f}")
    print(f"- LLM: llm_fail_risk_floor={float(args.llm_fail_risk_floor):.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
