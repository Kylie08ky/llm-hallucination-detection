# src/fa_eval/hf_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenUncertainty:
    avg_nll: float
    avg_entropy: float
    gen_tokens: int


class HFUpstreamGenerator:
    """
    HF local generator that supports Groq-equivalent conditioning:
      - system_prompt
      - tool_schemas_text (concatenated JSON schemas)
      - fewshot_examples
    and computes token-level uncertainty with a stable forward-pass method
    (avg NLL + avg entropy over generated tokens).
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        system_prompt: str = "",
        tool_schemas_text: str = "",
        fewshot_examples: str = "",
        force_json_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt or ""
        self.tool_schemas_text = tool_schemas_text or ""
        self.fewshot_examples = fewshot_examples or ""
        self.force_json_only = bool(force_json_only)

        if torch_dtype is None:
            # bf16 on GPU is usually stable; on CPU keep fp32
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.model.eval()

    def build_prompt(self, user_request: str) -> str:
        """
        Build a single instruction text that mirrors your Groq setup:
        - system prompt
        - strict JSON output rule
        - tools schemas
        - few-shot examples
        - user request
        """
        rules = [
            "You are an upstream generator for a tool-calling robot car system.",
            "Your job is to convert the user request into a SINGLE tool call.",
            "Return ONLY valid JSON (no markdown, no extra text).",
            "",
            "Output schema (must match exactly):",
            '{ "call": [ { "function_name": "<tool>", "parameters": { ... } } ] }',
            "",
            "Rules:",
            "- If the request CAN be satisfied using one of the available tools, you MUST choose one of them.",
            "- If the request CANNOT be satisfied using available tools, you MAY output another function_name.",
        ]
        if self.force_json_only:
            rules.append("- Do not output explanations. JSON only.")

        blocks = []
        if self.system_prompt.strip():
            blocks.append(self.system_prompt.strip())

        blocks.append("\n".join(rules))

        if self.tool_schemas_text.strip():
            blocks.append("Available tools (JSON schemas):\n" + self.tool_schemas_text.strip())

        if self.fewshot_examples.strip():
            blocks.append("Few-shot examples:\n" + self.fewshot_examples.strip())

        blocks.append("User request:\n" + user_request.strip())

        return "\n\n".join(blocks).strip()

    def _encode(self, text_prompt: str):
        """
        Use chat template if the tokenizer provides one; otherwise plain text.
        """
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "You are a precise JSON-only tool-call generator."},
                {"role": "user", "content": text_prompt},
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(rendered, return_tensors="pt")
        else:
            return self.tokenizer(text_prompt, return_tensors="pt")

    @torch.no_grad()
    def generate_with_uncertainty(
        self,
        user_request: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        typical_p: float = 0.95,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
    ) -> Tuple[str, GenUncertainty]:

        """
        Returns:
          completion_text: decoded completion ONLY (not including prompt)
          unc: avg_nll, avg_entropy, gen_tokens computed from a stable forward-pass
        """
        full_prompt = self.build_prompt(user_request)
        inputs = self._encode(full_prompt)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),

            # === sampling (important for SC runs diversity) ===
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            typical_p=float(typical_p),
            repetition_penalty=float(repetition_penalty),

            # === ensure we don't accidentally use beam search ===
            num_beams=1,

            return_dict_in_generate=True,
            output_scores=False,  # we do NOT rely on out.scores
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        seq = gen_out.sequences[0]  # [total_len]
        input_len = int(inputs["input_ids"].shape[1])
        gen_token_ids = seq[input_len:]  # generated tokens only
        gen_tokens = int(gen_token_ids.shape[0])

        completion_text = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()

        if gen_tokens == 0:
            return completion_text, GenUncertainty(avg_nll=0.0, avg_entropy=0.0, gen_tokens=0)

        # ---------- Stable uncertainty computation (forward-pass based) ----------
        # Run a single forward pass on the full sequence to get logits.
        # logits[t] predicts token at t+1. So for generated tokens starting at input_len,
        # we need logits positions from (input_len - 1) to (input_len - 1 + gen_tokens - 1).
        fw = self.model(seq.unsqueeze(0))
        logits = fw.logits[0].float()  # [total_len, vocab] in fp32

        start = input_len - 1
        if start < 0:
            start = 0
        end = start + gen_tokens
        step_logits = logits[start:end]  # [gen_tokens, vocab]

        log_probs = F.log_softmax(step_logits, dim=-1)  # [gen_tokens, vocab]
        probs = torch.exp(log_probs)

        # entropy per token: -sum(p log p)
        entropy_per = -(probs * log_probs).sum(dim=-1)  # [gen_tokens]
        entropy_per = torch.nan_to_num(entropy_per, nan=0.0, posinf=0.0, neginf=0.0)

        # nll per token: -log p(chosen_token)
        chosen = gen_token_ids.to(log_probs.device)  # [gen_tokens]
        nll_per = -log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)  # [gen_tokens]
        nll_per = torch.nan_to_num(nll_per, nan=0.0, posinf=0.0, neginf=0.0)

        avg_nll = float(nll_per.mean().item())
        avg_entropy = float(entropy_per.mean().item())

        return completion_text, GenUncertainty(avg_nll=avg_nll, avg_entropy=avg_entropy, gen_tokens=gen_tokens)
