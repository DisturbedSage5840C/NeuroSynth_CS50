from __future__ import annotations

from dataclasses import dataclass, field

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class CorpusStats:
    n_pubmed_records: int
    n_instruction_examples: int
    n_type1: int
    n_type2: int
    n_type3: int
    years: tuple[int, int]
    warnings: list[str] = field(default_factory=list)
