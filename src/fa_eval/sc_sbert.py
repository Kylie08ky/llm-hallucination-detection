# src/fa_eval/sc_sbert.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def action_to_text(function_name: str, parameters: Dict[str, Any]) -> str:
    """
    Stable serialization: function_name | k1=v1 | k2=v2 ...
    - Sort keys for determinism
    - Round floats to reduce tiny formatting differences
    """
    parts = [function_name.strip()]
    if isinstance(parameters, dict):
        for k in sorted(parameters.keys()):
            v = parameters.get(k)
            fv = _safe_float(v)
            if fv is not None:
                v_str = f"{fv:.3f}".rstrip("0").rstrip(".")  # 1.000->1, 0.500->0.5
            else:
                v_str = str(v)
            parts.append(f"{k}={v_str}")
    return " | ".join(parts)

def mean_pairwise_cosine(embeddings) -> float:
    """
    Mean cosine similarity over all pairs (upper triangle).
    Returns 1.0 if N<=1.
    """
    import numpy as np

    X = np.asarray(embeddings, dtype=float)
    n = X.shape[0]
    if n <= 1:
        return 1.0

    # normalize
    norms = (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Xn = X / norms

    # cosine matrix
    C = Xn @ Xn.T

    # upper triangle mean excluding diagonal
    s = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += float(C[i, j])
            cnt += 1
    return s / max(cnt, 1)

def sc_sbert_metrics_from_runs(
    runs: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Input: runs list where each run has {"call":[{"function_name":..,"parameters":..}, ...]}
    Output: (n, agreement, risk)
      agreement = mean pairwise SBERT cosine similarity
      risk = 1 - agreement
    """
    # Lazy import to keep baseline runnable without SBERT installed
    from sentence_transformers import SentenceTransformer

    texts: List[str] = []
    for r in runs:
        call = r.get("call")
        if not isinstance(call, list) or not call:
            continue
        first = call[0]
        if not isinstance(first, dict):
            continue
        fn = first.get("function_name", "")
        params = first.get("parameters", {})
        if not isinstance(fn, str) or not fn:
            continue
        if not isinstance(params, dict):
            params = {"_raw": params}
        texts.append(action_to_text(fn, params))

    n = len(texts)
    if n == 0:
        return 0, None, None

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)

    agreement = float(mean_pairwise_cosine(emb))
    # clamp to [0,1] to be safe
    agreement = max(0.0, min(1.0, agreement))
    risk = 1.0 - agreement
    return n, agreement, risk
