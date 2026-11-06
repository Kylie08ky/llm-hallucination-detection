# FA3940 Hallucination Detection

**Risk-Fusion + Selective LLM-as-a-judge Routing**

FA3940 research project

------------------------------------------------------------------------

# 1. Overview

This repository implements a hallucination detection system for structured tool/function calls.

The system does NOT fine-tune any model.\
Instead, it performs post-hoc risk fusion over multiple detection signals and applies a calibrated decision rule.

Final decision rule:

    risk_final = w_unc * uncertainty
               + w_sc  * self_consistency
               + w_llm * llm_risk
               + w_unit * unit_mismatch
               + w_proxy * proxy_risk

    blocked = (risk_final >= final_threshold)

Routing rule:

    route_to_llm = (u_score >= thr_u)

All final hyperparameters are frozen in:

    config/frozen.yaml

------------------------------------------------------------------------

# 2. Repository Structure

    FA3940_Hallucinations/
    ├── config/
    │   ├── frozen.yaml
    │   ├── routing.yaml
    │   └── thresholds_sbert.json
    │
    ├── data/
    │   ├── dev/
    │   │   ├── prompts/
    │   │   ├── cases_json/
    │   │   └── gt/
    │   ├── test/
    │   │   ├── prompts/
    │   │   ├── cases_json/
    │   │   └── gt/
    │   └── archive_dev/
    │
    ├── results/
    │   ├── dev/
    │   └── test/
    │
    ├── scripts/
    │   ├── pipeline/
    │   ├── eval/
    │   ├── plots/
    │   └── run/
    │
    ├── src/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# 3. Environment Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH=src

------------------------------------------------------------------------

# 4. LLM Backend (LLM-as-a-judge)

The system assumes a local Ollama backend:

    ollama serve
    ollama list

Example model:

    qwen2.5:3b

------------------------------------------------------------------------

# 5. Upstream: Case Generation & GT Annotation

## 5.1 Generate Cases (TEST)

    PYTHONPATH=src python scripts/pipeline/generate_cases_sc.py --prompts data/test/prompts/prompts_mixed.txt --out data/test/cases_json --gen_backend hf --model hf:Qwen/Qwen2.5-1.5B-Instruct

## 5.2 Annotate Expected Calls (TEST)

    PYTHONPATH=src python scripts/pipeline/annotate_expected_calls.py --cases_dir data/test/cases_json --out_jsonl data/test/gt/gt.jsonl

------------------------------------------------------------------------

# 6. DEV Pipeline (Calibration)

Run:

    bash scripts/run/run_dev_pipeline.sh

This performs:

1.  Detection
2.  GT alignment
3.  Merge
4.  Baseline metrics
5.  thr_u sweep
6.  Apply routing
7.  Summary metrics
8.  Plot generation

After sweep, update `config/frozen.yaml` with selected parameters.

------------------------------------------------------------------------

# 7. TEST Pipeline (Frozen Evaluation)

Run:

    bash scripts/run/run_test_frozen.sh

This:

-   Reads ONLY `config/frozen.yaml`
-   Applies frozen routing threshold
-   Computes final metrics
-   Generates plots

No sweep is performed on TEST.

------------------------------------------------------------------------

# 8. Reproducibility Notes

-   `config/frozen.yaml` is the single source of truth.
-   `results/` is reproducible and may be gitignored.
-   `data/*/cases_json/` can be regenerated.
-   Prompts and GT should be version-controlled if allowed.

------------------------------------------------------------------------
