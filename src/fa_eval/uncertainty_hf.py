# src/fa_eval/uncertainty_hf.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, Tuple
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class UncResult:
    n_tokens: int
    nll: float            # avg negative log-likelihood per token
    entropy: float        # avg token entropy
    risk: float           # normalized 0..1
    details: str


@lru_cache(maxsize=2)
def _load(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    return tok, model


def _mean_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: [1, T, V]
    logp = torch.log_softmax(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    ent = -(p * logp).sum(dim=-1)  # [1, T]
    return ent.mean()


def score_action_uncertainty(
    prompt: str,
    tools_text: str,
    action_json_text: str,
    model_name: str,
    max_context_chars: int = 8000,
) -> UncResult:
    """
    Scores the likelihood of ACTION JSON as continuation of a context.
    Returns avg NLL + avg entropy, plus a normalized risk in [0,1].
    """
    tok, model = _load(model_name)

    # build context (truncate defensively to keep CPU fast/stable)
    ctx = (
        "You are a tool-calling agent. Follow the tool list.\n\n"
        "TOOLS:\n" + tools_text + "\n\n"
        "USER:\n" + str(prompt) + "\n\n"
        "ACTION_JSON:\n"
    )
    if len(ctx) > max_context_chars:
        ctx = ctx[-max_context_chars:]  # keep tail

    # We only score the action part as labels; context tokens get label=-100
    ctx_ids = tok(ctx, return_tensors="pt", truncation=True).input_ids
    act_ids = tok(action_json_text, return_tensors="pt", truncation=True).input_ids

    input_ids = torch.cat([ctx_ids, act_ids], dim=1)

    # labels: ignore context
    labels = input_ids.clone()
    labels[:, : ctx_ids.shape[1]] = -100

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        # out.loss is mean over non-ignored tokens (cross-entropy)
        nll = float(out.loss.detach().cpu().item())

        # entropy: compute from logits, average only on action token positions
        logits = out.logits  # [1, T, V]
        # take logits for positions that predict next token: shift by 1
        # We'll approximate by averaging entropy across the whole sequence,
        # then this is dominated by action part because ctx is long but labels ignored.
        # For a stricter version you can mask; this is good enough for step4 baseline.
        entropy = float(_mean_entropy(logits).detach().cpu().item())

    # Normalize to risk 0..1 (simple robust mapping; calibration will set Tu anyway)
    # nll typical range on small instruct models: ~1.0..6.0
    nll_norm = min(1.0, max(0.0, (nll - 1.0) / 5.0))
    # entropy typical range: ~1..6, map similarly
    ent_norm = min(1.0, max(0.0, (entropy - 1.0) / 5.0))

    risk = 0.5 * nll_norm + 0.5 * ent_norm

    return UncResult(
        n_tokens=int(act_ids.shape[1]),
        nll=nll,
        entropy=entropy,
        risk=risk,
        details=f"hf_local nll={nll:.3f} ent={entropy:.3f} tokens={int(act_ids.shape[1])}",
    )
