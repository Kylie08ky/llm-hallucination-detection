from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional


def canonical(obj: Any) -> str:
    """Canonical string for exact-match agreement."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def majority_vote(outputs: List[Any]) -> Tuple[float, Any]:
    """Return (agreement_ratio, majority_object) using exact-match over canonical JSON."""
    if not outputs:
        return 0.0, {}
    c = Counter(canonical(o) for o in outputs)
    maj_str, maj_cnt = c.most_common(1)[0]
    try:
        maj_obj = json.loads(maj_str)
    except Exception:
        maj_obj = outputs[0]
    return maj_cnt / max(1, len(outputs)), maj_obj


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _extract_call_list(run: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Expect run like {'call':[{'function_name':..,'parameters':..}, ...]}."""
    call = run.get("call")
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


def _keyset_for_run(call_list: List[Dict[str, Any]]) -> set:
    """
    Schema-agnostic keyset capturing call length, order, tool names, and parameter keys.
    This stays model-/tool-independent: it only measures the model's *self* stability.
    """
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


def _value_stability_risk(call_lists: List[List[Dict[str, Any]]]) -> float:
    """
    Schema-agnostic value stability:
      - numeric: robust relative MAD (median absolute deviation)
      - non-numeric scalar: 1 - mode_fraction
    Aggregate risk as MAX across all observed fields (recall-oriented).
    """
    vals_num: Dict[str, List[float]] = defaultdict(list)
    vals_cat: Dict[str, List[str]] = defaultdict(list)

    for calls in call_lists:
        for i, call in enumerate(calls):
            fn = str(call.get("function_name", ""))
            params = call.get("parameters") or {}
            if not isinstance(params, dict):
                continue
            for k, v in params.items():
                field = f"call[{i}].{fn}.{k}"
                fv = _safe_float(v)
                if fv is not None:
                    vals_num[field].append(fv)
                else:
                    vals_cat[field].append(str(v))

    risks: List[float] = []

    # numeric dispersion -> risk
    for _, xs in vals_num.items():
        if len(xs) <= 1:
            continue
        med = statistics.median(xs)
        mad = statistics.median([abs(x - med) for x in xs])
        rel = mad / (abs(med) + 1e-6)
        # scale: rel MAD 0.20 -> risk 1.0 (tunable, reasonable default)
        risks.append(max(0.0, min(1.0, rel / 0.20)))

    # categorical disagreement -> risk
    for _, xs in vals_cat.items():
        if len(xs) <= 1:
            continue
        c = Counter(xs)
        mode_cnt = c.most_common(1)[0][1]
        mode_frac = mode_cnt / max(1, len(xs))
        risks.append(1.0 - mode_frac)

    return max(risks) if risks else 0.0


def sc_risk_from_runs(runs: List[Dict[str, Any]], base_metric: str = "exact") -> Dict[str, Any]:
    """
    SC risk computed purely from multiple generations (no external knowledge, no domain rules).

    Backward compatible keys:
      - n, agreement, risk_sc, majority

    Additional debug keys:
      - key_agreement, key_risk
      - value_risk
      - base_risk, base_metric
    """
    agree_base, maj = majority_vote(runs)
    base_risk = 1.0 - float(agree_base)

    call_lists: List[List[Dict[str, Any]]] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        cl = _extract_call_list(r)
        if cl is not None:
            call_lists.append(cl)

    if not call_lists:
        return {
            "n": 0,
            "agreement": None,
            "risk_sc": None,
            "majority": {},
            "key_agreement": None,
            "key_risk": None,
            "value_risk": None,
            "base_risk": None,
            "base_metric": base_metric,
        }

    key_sets = [_keyset_for_run(cl) for cl in call_lists]
    key_agree = float(_mean_pairwise_jaccard(key_sets))
    key_risk = 1.0 - key_agree

    value_risk = float(_value_stability_risk(call_lists))

    # recall-oriented: any instability increases risk
    risk_sc = max(base_risk, key_risk, value_risk)

    return {
        "n": len(call_lists),
        "agreement": float(agree_base),
        "risk_sc": float(risk_sc),
        "majority": maj,
        "key_agreement": key_agree,
        "key_risk": key_risk,
        "value_risk": value_risk,
        "base_risk": base_risk,
        "base_metric": base_metric,
    }
