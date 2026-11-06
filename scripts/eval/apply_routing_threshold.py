#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply routing threshold (thr_u) to decide route_llm and recompute blocked under risk-fusion.

Fixes vs previous version:
- DO NOT merge back (avoids route_llm_x / route_llm_y suffix problems).
- Always write stable columns: route_llm, risk_final_routed, blocked_routed.
- Compatible with legacy column names (e.g., "route llm").
- If route_llm is missing, compute from u_col >= thr_u.
- Fusion-aware routed risk: remove LLM contribution for non-routed cases:
      risk_final_routed = risk_final - contrib_llm  (if not routed)
      risk_final_routed = risk_final                (if routed)

Expected columns in merged_cases.csv (post-gate rows):
- case_id
- status (CORRECT vs others)
- risk_final (float)
- final_threshold (float)  (unless provided by args)
- contrib_llm (float) OR (w_llm and llm_risk)
- u_score OR uncertainty_risk (routing score)
- hard_fail (optional; if exists, post-gate is hard_fail==False)
"""

import argparse
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def read_routing_yaml(path: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if not path or not os.path.exists(path):
        return cfg
    current_section: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and not line.startswith("-"):
                current_section = line[:-1].strip()
                cfg[current_section] = {}
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                target = cfg.get(current_section, cfg) if current_section else cfg
                if v.lower() in ("true", "false"):
                    target[k] = (v.lower() == "true")
                else:
                    try:
                        target[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                    except ValueError:
                        target[k] = v
                if current_section:
                    cfg[current_section] = target
    return cfg


def pick_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--config", default="config/routing.yaml")
    ap.add_argument("--thr_u", type=float, default=None)
    ap.add_argument("--final_threshold", type=float, default=None)
    ap.add_argument("--u_col", type=str, default=None, help="prefer u_score; fallback uncertainty_risk")
    args = ap.parse_args()

    df = pd.read_csv(args.merged_csv)
    if "case_id" not in df.columns:
        raise SystemExit("[ERROR] merged csv missing 'case_id' column.")

    # Work on post-gate subset
    if "hard_fail" in df.columns:
        post = df[df["hard_fail"] == False].copy()
    else:
        post = df.copy()

    cfg = read_routing_yaml(args.config)
    routing_cfg = cfg.get("routing", cfg)

    thr_u = args.thr_u
    if thr_u is None:
        thr_u = float(routing_cfg.get("thr_u", 0.30))

    # final_threshold
    final_thr = args.final_threshold
    if final_thr is None:
        if "final_threshold" not in post.columns:
            raise SystemExit("[ERROR] final_threshold not provided and not found in merged csv.")
        final_thr = float(pd.to_numeric(post["final_threshold"], errors="coerce").iloc[0])

    # choose u_col
    u_col = args.u_col or routing_cfg.get("u_col", None)
    if u_col is None:
        u_col = pick_col(post, ["u_score", "u", "uncertainty_risk"])
    if u_col is None or u_col not in post.columns:
        raise SystemExit("[ERROR] cannot find u_col. Tried u_score/u/uncertainty_risk.")
    u_vals = pd.to_numeric(post[u_col], errors="coerce").fillna(0.0)

    # route_llm input compatibility
    # If dataset already has route_llm/route llm, we ignore it and recompute (routing threshold is what we apply)
    post["route_llm"] = (u_vals >= thr_u)

    # compute removable LLM contribution
    if "contrib_llm" in post.columns:
        contrib_llm = pd.to_numeric(post["contrib_llm"], errors="coerce").fillna(0.0)
    else:
        if ("w_llm" in post.columns) and ("llm_risk" in post.columns):
            contrib_llm = (
                pd.to_numeric(post["w_llm"], errors="coerce").fillna(0.0)
                * pd.to_numeric(post["llm_risk"], errors="coerce").fillna(0.0)
            )
        else:
            contrib_llm = pd.Series(0.0, index=post.index)

    if "risk_final" not in post.columns:
        raise SystemExit("[ERROR] merged csv missing risk_final (needed for fusion routing).")
    risk_final = pd.to_numeric(post["risk_final"], errors="coerce").fillna(0.0)

    # fusion-aware routed risk
    risk_final_routed = risk_final.copy()
    not_routed = ~post["route_llm"]
    risk_final_routed.loc[not_routed] = (risk_final - contrib_llm).loc[not_routed]
    post["risk_final_routed"] = risk_final_routed

    # decision
    post["blocked_routed"] = (post["risk_final_routed"] >= final_thr)

    # Write back WITHOUT merge: map by case_id
    out = df.copy()
    if "blocked" in out.columns:
        out["blocked_full"] = out["blocked"]

    post_idx = post.set_index("case_id")
    out["route_llm"] = out["case_id"].map(post_idx["route_llm"])
    out["risk_final_routed"] = out["case_id"].map(post_idx["risk_final_routed"])
    out["blocked_routed"] = out["case_id"].map(post_idx["blocked_routed"])

    # Use routed decision where available; else keep original blocked if exists
    if "blocked" in out.columns:
        out["blocked"] = out["blocked_routed"].where(out["blocked_routed"].notna(), out["blocked"]).fillna(False).astype(bool)
    else:
        out["blocked"] = out["blocked_routed"].fillna(False).astype(bool)

    out.to_csv(args.out_csv, index=False)

    # safe prints
    route_calls = int(out["route_llm"].fillna(False).sum())
    blocked_cnt = int(out["blocked"].fillna(False).sum())
    print("[OK] wrote", args.out_csv)
    print(f"thr_u: {thr_u} | final_threshold: {final_thr} | u_col: {u_col}")
    print("route_llm calls:", route_calls)
    print("blocked (after routing):", blocked_cnt)


if __name__ == "__main__":
    main()
