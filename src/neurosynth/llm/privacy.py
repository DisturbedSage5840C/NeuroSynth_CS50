# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from presidio_analyzer import AnalyzerEngine


class PrivacyScanner:
    def __init__(self) -> None:
        self.engine = AnalyzerEngine()

    def is_safe(self, text: str) -> bool:
        findings = self.engine.analyze(text=text, language="en")
        # conservative policy: any PII entity is rejected
        return len(findings) == 0

    def filter_safe_examples(self, examples: list[dict]) -> list[dict]:
        safe = []
        for ex in examples:
            blob = f"{ex.get('system','')}\n{ex.get('user','')}\n{ex.get('assistant','')}"
            if self.is_safe(blob):
                safe.append(ex)
        return safe
