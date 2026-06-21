# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from Bio import Entrez


@dataclass
class PMIDVerifier:
    email: str = ""
    ttl_seconds: int = 24 * 3600

    def __post_init__(self) -> None:
        Entrez.email = self.email or os.getenv("NEURO_ENTREZ_EMAIL", "noreply@localhost")
        self._cache: dict[str, tuple[float, bool]] = {}

    def is_valid(self, pmid: str) -> bool:
        now = time.time()
        if pmid in self._cache and now - self._cache[pmid][0] < self.ttl_seconds:
            return self._cache[pmid][1]

        ok = False
        try:
            h = Entrez.esummary(db="pubmed", id=str(pmid))
            rec = Entrez.read(h)
            ok = bool(rec and rec[0].get("Id"))
        except Exception:
            ok = False
        self._cache[pmid] = (now, ok)
        return ok

    def filter_valid(self, pmids: list[str]) -> list[str]:
        return [p for p in pmids if self.is_valid(p)]
