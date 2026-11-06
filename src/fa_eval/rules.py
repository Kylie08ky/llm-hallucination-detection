from __future__ import annotations
from typing import Any, Dict, List, Tuple

ALLOWED_FUNCTIONS = {
    "drive_direction",
    "drive_coordinates",
    "add_obstacle",
    "calculate_path",
    "set_radius",
}

MAX_DISTANCE_M = 50.0
MAX_RADIUS_M = 10.0


def evaluate_rules(llm_output: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Minimal model-agnostic evaluator.
    Returns:
      (is_valid, reasons, scores)
    scores are simple 0~1 risk proxies.
    """
    reasons: List[str] = []
    scores: Dict[str, float] = {
        "schema_risk": 0.0,
        "tool_risk": 0.0,
        "param_risk": 0.0,
        "conflict_risk": 0.0,
    }

    # --- basic structure ---
    out_type = llm_output.get("type")
    call = llm_output.get("call")

    if out_type != "list":
        reasons.append("llm_output_type_not_list")
        scores["schema_risk"] = max(scores["schema_risk"], 0.6)

    if not isinstance(call, list):
        reasons.append("call_not_list")
        scores["schema_risk"] = 1.0
        return False, reasons, scores

    if len(call) == 0:
        reasons.append("empty_call_list")
        scores["schema_risk"] = max(scores["schema_risk"], 0.8)
        return False, reasons, scores

    seen_dirs: List[str] = []

    for idx, item in enumerate(call):
        if not isinstance(item, dict):
            reasons.append(f"call_item_{idx}_not_object")
            scores["schema_risk"] = 1.0
            continue

        fn = item.get("function_name")
        params = item.get("parameters")

        if fn not in ALLOWED_FUNCTIONS:
            reasons.append(f"unknown_function:{fn}")
            scores["tool_risk"] = 1.0

        if not isinstance(params, dict):
            reasons.append(f"parameters_not_object:{fn}")
            scores["schema_risk"] = max(scores["schema_risk"], 0.8)
            continue

        # --- function-specific checks ---
        if fn == "drive_direction":
            direction = params.get("direction")
            if direction not in {"left", "right", "forward", "backward"}:
                reasons.append("invalid_direction")
                scores["param_risk"] = max(scores["param_risk"], 0.7)
            else:
                seen_dirs.append(direction)

            if "distance" in params:
                dist = params.get("distance")
                if not isinstance(dist, (int, float)):
                    reasons.append("distance_not_number")
                    scores["param_risk"] = max(scores["param_risk"], 0.7)
                else:
                    if dist < 0:
                        reasons.append("distance_negative")
                        scores["param_risk"] = 1.0
                    if dist > MAX_DISTANCE_M:
                        reasons.append("distance_too_large")
                        scores["param_risk"] = max(scores["param_risk"], 0.9)

        elif fn == "drive_coordinates":
            x = params.get("x")
            y = params.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                reasons.append("coords_not_number")
                scores["param_risk"] = max(scores["param_risk"], 0.8)

        elif fn == "set_radius":
            r = params.get("radius")
            if not isinstance(r, (int, float)):
                reasons.append("radius_not_number")
                scores["param_risk"] = max(scores["param_risk"], 0.7)
            else:
                if r <= 0:
                    reasons.append("radius_non_positive")
                    scores["param_risk"] = 1.0
                if r > MAX_RADIUS_M:
                    reasons.append("radius_too_large")
                    scores["param_risk"] = max(scores["param_risk"], 0.8)

        # add_obstacle / calculate_path

    # --- very simple conflict check ---
    if "forward" in seen_dirs and "backward" in seen_dirs:
        reasons.append("direction_conflict_forward_backward")
        scores["conflict_risk"] = max(scores["conflict_risk"], 0.8)

    # --- MVP decision ---
    if any(r.startswith("unknown_function:") for r in reasons):
        return False, reasons, scores

    if (
        "radius_non_positive" in reasons
        or "distance_negative" in reasons
        or "distance_too_large" in reasons
        or "radius_too_large" in reasons
    ):
        return False, reasons, scores

    if "call_not_list" in reasons or "call_item_0_not_object" in reasons:
        return False, reasons, scores


    return True, reasons, scores
