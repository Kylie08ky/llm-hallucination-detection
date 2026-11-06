#!/usr/bin/env bash
set -euo pipefail

SPLIT="test"
CASES_DIR="data/${SPLIT}/cases_json"
GT_JSONL="data/${SPLIT}/gt/gt.jsonl"
OUT_DIR="results/${SPLIT}"
CFG="config/frozen.yaml"

mkdir -p "${OUT_DIR}"/{det,merged,metrics,figs}

# --- backup freeze config for reproducibility ---
cp "${CFG}" "${OUT_DIR}/run_config_frozen.yaml"
git rev-parse HEAD > "${OUT_DIR}/git_commit.txt" 2>/dev/null || true
date -Iseconds > "${OUT_DIR}/run_time.txt"

# --- read frozen.yaml (no yq needed) ---
eval "$(
python - <<'PY'
import yaml, sys
cfg = yaml.safe_load(open("config/frozen.yaml","r"))

def g(path, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict): return default
        cur = cur.get(k, default)
        if cur is default: return default
    return cur

def req(path):
    v = g(path, None)
    if v is None:
        raise SystemExit(f"Missing key in frozen.yaml: {path}")
    return v

print(f'export W_UNC="{req("fusion.w_unc")}"')
print(f'export W_SC="{req("fusion.w_sc")}"')
print(f'export W_LLM="{req("fusion.w_llm")}"')
print(f'export W_UNIT="{req("fusion.w_unit")}"')
print(f'export W_PROXY="{req("fusion.w_proxy")}"')
print(f'export FINAL_THR="{req("fusion.final_threshold")}"')
print(f'export THR_U="{req("routing.thr_u")}"')

print(f'export PROXY_METRIC="{req("modules.proxy_metric")}"')
print(f'export PROXY_THR="{req("modules.proxy_threshold")}"')

sc_enabled = g("modules.sc_enabled", True)
print(f'export SC_ENABLED="{sc_enabled}"')
print(f'export SC_METRIC="{g("modules.sc_metric","sbert")}"')
print(f'export SC_THR="{g("modules.sc_threshold",0.35)}"')
print(f'export SC_UNC_THR="{g("modules.sc_uncertainty_threshold",0.05)}"')

print(f'export LLM_MODEL="{req("modules.llm_model")}"')
print(f'export OLLAMA_URL="{req("modules.ollama_url")}"')
print(f'export LLM_RISK_THR="{g("modules.llm_risk_threshold",0.30)}"')
print(f'export LLM_FAIL_FLOOR="{g("modules.llm_fail_risk_floor",0.20)}"')

print(f'export UNIT_ROUTE_W="{req("unit.unit_route_weight")}"')
PY
)"

SC_ARGS=""
if [[ "${SC_ENABLED}" == "True" || "${SC_ENABLED}" == "true" ]]; then
  SC_ARGS="--sc --sc_metric ${SC_METRIC} --sc_threshold ${SC_THR} --sc_uncertainty_threshold ${SC_UNC_THR}"
fi

# --- (1) eval_cases ---
PYTHONPATH=src python scripts/pipeline/eval_cases.py \
  --cases_dir "${CASES_DIR}" \
  --out_csv "${OUT_DIR}/det/results_frozen.csv" \
  --tools_dir util/tools \
  --decision_mode back_only \
  --proxy_metric "${PROXY_METRIC}" \
  --proxy_threshold "${PROXY_THR}" \
  ${SC_ARGS} \
  --thr_u "${THR_U}" \
  --final_threshold "${FINAL_THR}" \
  --w_unc "${W_UNC}" --w_sc "${W_SC}" --w_llm "${W_LLM}" --w_unit "${W_UNIT}" --w_proxy "${W_PROXY}" \
  --unit_route_weight "${UNIT_ROUTE_W}" \
  --llm_model "${LLM_MODEL}" --ollama_url "${OLLAMA_URL}" \
  --llm_risk_threshold "${LLM_RISK_THR}" \
  --llm_fail_risk_floor "${LLM_FAIL_FLOOR}"

# --- (2) compare_pred_gt ---
PYTHONPATH=src python scripts/pipeline/compare_pred_gt.py \
  --cases_dir "${CASES_DIR}" \
  --gt_jsonl "${GT_JSONL}" \
  --out_csv "${OUT_DIR}/det/gt_compare_frozen.csv"

# --- (3) merged ---
PYTHONPATH=src python scripts/pipeline/analyze_gt_vs_detection.py \
  --det_csv "${OUT_DIR}/det/results_frozen.csv" \
  --gt_compare "${OUT_DIR}/det/gt_compare_frozen.csv" \
  --out_dir "${OUT_DIR}/merged" \
  --w_unc "${W_UNC}" --w_sc "${W_SC}" --w_llm "${W_LLM}" --w_unit "${W_UNIT}" --w_proxy "${W_PROXY}"

# --- (4) apply routing using frozen thr_u (avoid config/routing.yaml conflicts) ---
cat > "${OUT_DIR}/routing_frozen.yaml" <<EOF
thr_u: ${THR_U}
u_col: u_score
final_threshold: ${FINAL_THR}
EOF

PYTHONPATH=src python scripts/eval/apply_routing_threshold.py \
  --merged_csv "${OUT_DIR}/merged/merged_cases.csv" \
  --config "${OUT_DIR}/routing_frozen.yaml" \
  --out_csv "${OUT_DIR}/merged/merged_cases_routed.csv"

# --- (5) metrics ---
PYTHONPATH=src python scripts/eval/metrics_from_merged.py \
  --merged "${OUT_DIR}/merged/merged_cases.csv" \
  --out_dir "${OUT_DIR}/metrics"

# --- (6) summary (baseline vs routed) ---
PYTHONPATH=src python scripts/eval/summary_metrics.py \
  --merged "${OUT_DIR}/merged/merged_cases.csv" \
  --merged_routed "${OUT_DIR}/merged/merged_cases_routed.csv" \
  --out_csv "${OUT_DIR}/metrics/summary_overall.csv"

# --- (7) plots ---
PYTHONPATH=src python scripts/plots/plot_three_figs.py \
  --merged "${OUT_DIR}/merged/merged_cases.csv" \
  --summary "${OUT_DIR}/metrics/summary_overall.csv" \
  --out_dir "${OUT_DIR}/figs"

echo "[OK] Test frozen pipeline done -> ${OUT_DIR}"
