from __future__ import annotations

from pathlib import Path

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

def generate_model_card(output_path: Path, eval_summary: dict | None = None) -> None:
    eval_summary = eval_summary or {}
    content = f"""---
language: en
license: llama3
base_model: meta-llama/Meta-Llama-3-8B-Instruct
pipeline_tag: text-generation
---

# NeuroSynth-LLM-8B

## Intended Use
NeuroSynth-LLM-8B is a clinical decision support language model for neurological deterioration reporting.

## Training Data
- De-identified synthetic and literature-derived biomedical text.
- No direct patient identifiers are used.

## Training Procedure
- Stage 1: QLoRA continual pretraining on neurology corpus.
- Stage 2: SFT on structured instruction data.
- Stage 3: DPO with neurologist preference pairs.

## Limitations
- Not for standalone diagnosis.
- Requires clinical oversight and external verification.

## Evaluation
{eval_summary}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
