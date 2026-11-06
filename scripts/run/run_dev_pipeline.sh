#!/usr/bin/env bash
set -euo pipefail

SPLIT="dev"
CASES_DIR="data/${SPLIT}/cases_json"
GT_JSONL="data/${SPLIT}/gt/gt_dev.jsonl"
OUT_DIR="results/${SPLIT}"

mkdir -p "${OUT_DIR}"/{det,merged,metrics,figs}

########################################
# (1) eval_cases
########################################

PYTHONPATH=src python scripts/pipeline/eval_cases.py \
  --cases_dir "${CASES_DIR}" \
  --out_csv "${OUT_DIR}/det/results_dev.csv" \
  --tools_dir util/tools \
  --decision_mode back_only \
  --proxy_metric upstream_eq \
  --proxy_threshold 0.32 \
  --sc --sc_metric sbert --sc_threshold 0.35 \
  --sc_uncertainty_threshold 0.05 \
  --thr_u 0.24 \
  --final_threshold 0.12 \
  --w_unc 0.30 --w_sc 0.20 --w_llm 0.20 --w_unit 0.25 --w_proxy 0.05 \
  --unit_route_weight 0.30 \
  --llm_model qwen2.5:3b --ollama_url http://127.0.0.1:11434 \
  --llm_risk_threshold 0.30 \
  --llm_fail_risk_floor 0.20

########################################
# (2) compare_pred_gt
########################################

PYTHONPATH=src python scripts/pipeline/compare_pred_gt.py \
  --cases_dir "${CASES_DIR}" \
  --gt_jsonl "${GT_JSONL}" \
  --out_csv "${OUT_DIR}/det/gt_compare_dev.csv"

########################################
# (3) merged_cases
########################################

PYTHONPATH=src python scripts/pipeline/analyze_gt_vs_detection.py \
  --det_csv "${OUT_DIR}/det/results_dev.csv" \
  --gt_compare "${OUT_DIR}/det/gt_compare_dev.csv" \
  --out_dir "${OUT_DIR}/merged" \
  --w_unc 0.30 --w_sc 0.20 --w_llm 0.20 --w_unit 0.25 --w_proxy 0.05

########################################
# (4) baseline metrics
########################################

PYTHONPATH=src python scripts/eval/metrics_from_merged.py \
  --merged "${OUT_DIR}/merged/merged_cases.csv" \
  --out_dir "${OUT_DIR}/metrics"

########################################
# (5) routing sweep
########################################

PYTHONPATH=src python scripts/eval/route_unc_sc_sweep.py \
  --merged "${OUT_DIR}/merged/merged_cases.csv" \
  --out_csv "${OUT_DIR}/metrics/route_unc_sweep.csv" \
  --u_col u_score \
  --thr_min 0.05 --thr_max 0.60 --thr_step 0.01 \
  --out_yaml config/routing.yaml

echo ">>> routing.yaml updated as dev recommended thr_u"

########################################
# (6) apply routing
########################################

PYTHONPATH=src python scripts/eval/apply_routing_threshold.py \
  --merged_csv "${OUT_DIR}/merged/merged_cases.csv" \
  --config config/routing.yaml \
  --out_csv "${OUT_DIR}/merged/merged_cases_routed.csv"

########################################
# (7) summary metrics
########################################

PYTHONPATH=src python scripts/eval/summary_metrics.py \
  --merged "${OUT_DIR}/merged/merged_cases.csv" \
  --merged_routed "${OUT_DIR}/merged/merged_cases_routed.csv" \
  --out_csv "${OUT_DIR}/metrics/summary_overall.csv"

########################################
# (8) plots
########################################

PYTHONPATH=src python scripts/plots/plot_three_figs.py \
  --merged "${OUT_DIR}/merged/merged_cases.csv" \
  --summary "${OUT_DIR}/metrics/summary_overall.csv" \
  --out_dir "${OUT_DIR}/figs"

echo "[OK] DEV pipeline finished."
