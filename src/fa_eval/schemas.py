from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CaseSample:
    case_id: str
    prompt: str
    llm_output: Dict[str, Any]
    model: Optional[str] = None
    timestamp: Optional[str] = None


def parse_case_json(obj: Dict[str, Any], case_id: str) -> Tuple[Optional[CaseSample], List[str]]:
    """
    Parse a single case json dict into CaseSample.
    Expected format:
      {
        "prompt": "...",
        "llm_output": {"type": "list", "call": [...]},
        "model": "...",          (optional)
        "timestamp": "..."       (optional)
      }
    Returns: (sample_or_none, errors)
    """
    errors: List[str] = []

    if not isinstance(obj, dict):
        return None, ["root_not_object"]

    prompt = obj.get("prompt")
    llm_output = obj.get("llm_output")

    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("missing_or_invalid_prompt")

    if not isinstance(llm_output, dict):
        errors.append("missing_or_invalid_llm_output")

    model = obj.get("model")
    timestamp = obj.get("timestamp")

    if errors:
        return None, errors

    return CaseSample(
        case_id=case_id,
        prompt=prompt.strip(),
        llm_output=llm_output,
        model=model if isinstance(model, str) else None,
        timestamp=timestamp if isinstance(timestamp, str) else None,
    ), []
