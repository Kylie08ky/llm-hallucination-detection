# Publication validation — 2026-11-06

- Python syntax: 28 source files parsed successfully.
- Six JSON files and both ground-truth JSONL files parsed successfully.
- Offline smoke checks passed: a valid movement call is accepted; an unknown tool and empty call list are rejected; a missing prompt is rejected; a valid case parses.
- Ground-truth records counted directly: development 100, test 200.
- The five technical presentation figures were visually inspected and extracted without modifying their pixels.
- The full Hugging Face/Ollama generation and detection pipeline was not rerun. Original generated cases and metric CSVs were absent from the supplied snapshot.

Historical figures and their denominator/reproducibility limits are documented in `docs/EXPERIMENTS.md`. They are not results of the publication smoke checks.
