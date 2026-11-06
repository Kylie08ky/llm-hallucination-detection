from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fa_eval.sc import sc_risk_from_runs

from fa_eval.hf_generator import HFUpstreamGenerator, GenUncertainty


def read_prompts(path: Path) -> List[str]:
    """Read prompts.txt and ignore empty lines / comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse a tool-call JSON from model output."""
    raw = text.strip()
    raw = re.sub(r"```json", "```", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "call" in obj:
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "call" in obj:
                return obj
        except Exception:
            return None
    return None


def _read_text_if_exists(p: Path) -> str:
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _read_tool_schemas_text(tools_dir: Path) -> str:
    """
    Read all tool schema json files and concatenate them.
    We keep order stable (sorted) for reproducibility.
    """
    if not tools_dir.exists():
        return ""
    parts: List[str] = []
    for fp in sorted(tools_dir.glob("*.json")):
        parts.append(fp.read_text(encoding="utf-8"))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate cases and self-consistency runs.")
    ap.add_argument("--prompts", type=str, default="data/prompts.txt", help="Path to prompts.txt")
    ap.add_argument("--out", type=str, default="data/cases_gen", help="Output directory for generated cases")
    ap.add_argument("--n", type=int, default=5, help="Number of SC runs per prompt")
    ap.add_argument("--start-id", type=int, default=1, help="Case id start (case_001...)")

    # Generator backend
    ap.add_argument("--gen_backend", choices=["groq", "hf"], default="groq")

    # Groq
    ap.add_argument("--model", type=str, default=None, help="Groq model name (overrides generator default)")

    # HF
    ap.add_argument("--hf_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device_map", type=str, default="auto")

    # Groq-equivalent conditioning files (same defaults as groq_generator.py expects)
    ap.add_argument("--system_txt", type=str, default="util/system.txt")
    ap.add_argument("--examples_txt", type=str, default="util/examples.txt")
    ap.add_argument("--tools_dir", type=str, default="util/tools")

    # Common
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed")

    args = ap.parse_args()

    prompts = read_prompts(Path(args.prompts))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_groq: Optional[GroqUpstreamGenerator] = None
    gen_hf: Optional[HFUpstreamGenerator] = None

    if args.gen_backend == "groq":
        gen_groq = GroqUpstreamGenerator(model=args.model, temperature=args.temperature, seed=args.seed)
        gen_name = f"groq:{gen_groq.model}"
    else:
        system_prompt = _read_text_if_exists(Path(args.system_txt))
        fewshot = _read_text_if_exists(Path(args.examples_txt))
        tools_text = _read_tool_schemas_text(Path(args.tools_dir))

        gen_hf = HFUpstreamGenerator(
            model_name=args.hf_model_name,
            device_map=args.device_map,
            system_prompt=system_prompt,
            tool_schemas_text=tools_text,
            fewshot_examples=fewshot,
            force_json_only=True,
        )
        gen_name = f"hf:{gen_hf.model_name}"

    summary_rows: List[Dict[str, Any]] = []

    for idx, prompt in enumerate(prompts, start=args.start_id):
        case_id = f"case_{idx:03d}"
        ts = datetime.now().replace(microsecond=0).isoformat()

        if gen_groq is not None:
            main_out = gen_groq.generate(prompt)
            unc: Optional[GenUncertainty] = None
        else:
            completion_text, unc = gen_hf.generate_with_uncertainty(  # type: ignore[union-attr]
                user_request=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
            )
            main_out = _extract_json_from_text(completion_text)
            if main_out is None:
                main_out = {
                    "call": [
                        {
                            "function_name": "ERROR",
                            "parameters": {"error": "unparseable_output", "raw": completion_text[:2000]},
                        }
                    ]
                }

        case_obj: Dict[str, Any] = {
            "prompt": prompt,
            "llm_output": main_out,
            "model": gen_name,
            "timestamp": ts,
        }
        if unc is not None:
            case_obj["gen_uncertainty"] = {
                "avg_nll": unc.avg_nll,
                "avg_entropy": unc.avg_entropy,
                "avg_logprob": -float(unc.avg_nll),
                "gen_tokens": unc.gen_tokens,
            }

        (out_dir / f"{case_id}.json").write_text(
            json.dumps(case_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # SC runs (always generated; uncertainty only stored for main case)
        runs: List[Dict[str, Any]] = []
        for r in range(args.n):
            if gen_groq is not None:
                runs.append(gen_groq.generate(prompt))
            else:
                # SC runs MUST be sampled with sufficient randomness, otherwise all runs become identical.
                sc_temp = max(float(args.temperature), 0.90) + 0.05 * float(r)
                sc_temp = min(sc_temp, 1.20)
                text_r, _unc_r = gen_hf.generate_with_uncertainty(  # type: ignore[union-attr]
                    user_request=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=sc_temp,
                    top_p=float(args.top_p),
                    top_k=50,
                    typical_p=0.95,
                    repetition_penalty=1.05,
                    do_sample=True,
                )
                out_r = _extract_json_from_text(text_r)
                if out_r is None:
                    out_r = {
                        "call": [
                            {
                                "function_name": "ERROR",
                                "parameters": {"error": "unparseable_output", "raw": text_r[:2000]},
                            }
                        ]
                    }
                runs.append(out_r)

        runs_obj = {"prompt": prompt, "runs": runs, "model": gen_name, "timestamp": ts}
        (out_dir / f"{case_id}_sc_runs.json").write_text(
            json.dumps(runs_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        sc = sc_risk_from_runs(runs)
        summary_rows.append(
            {
                "case_id": case_id,
                "prompt": prompt,
                "n_runs": sc["n"],
                "agreement": sc["agreement"],
                "risk_sc": sc["risk_sc"],
                "case_path": str(out_dir / f"{case_id}.json"),
                "sc_runs_path": str(out_dir / f"{case_id}_sc_runs.json"),
            }
        )

    summary_path = out_dir / "sc_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(summary_rows)} cases to {out_dir.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
