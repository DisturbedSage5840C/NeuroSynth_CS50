from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

from neurosynth.llm.report_generator import ClinicalReport


@dataclass
class ReportValidationResult:
    valid: bool
    reasons: list[str]


class ClinicalReportValidator:
    """Quality and safety validator for phase 6 clinical reports."""

    def __init__(self) -> None:
        self._mrn_pattern = re.compile(r"\b(?:MRN|MEDICAL\s*RECORD\s*NUMBER)\s*[:#-]?\s*\d{4,12}\b", re.IGNORECASE)
        self._dob_pattern = re.compile(r"\b(?:DOB|Date\s*of\s*Birth)\s*[:#-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        a_toks = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
        b_toks = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
        if not a_toks:
            return 0.0
        return len(a_toks & b_toks) / len(a_toks)

    def _pii_scan(self, text: str) -> list[str]:
        issues = []
        if self._mrn_pattern.search(text):
            issues.append("MRN detected")
        if self._dob_pattern.search(text):
            issues.append("DOB detected")

        try:
            from presidio_analyzer import AnalyzerEngine

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
            analyzer = AnalyzerEngine()
            ents = analyzer.analyze(text=text, language="en", entities=["PERSON", "LOCATION"])
            if any(e.entity_type == "PERSON" for e in ents):
                issues.append("NAME detected")
            if any(e.entity_type == "LOCATION" for e in ents):
                issues.append("LOCATION detected")
        except Exception:
            pass

        return issues

    def validate(
        self,
        report_payload: dict,
        retrieved_passages: dict[str, str],
        tft_interval_width: float,
        uncertainty_trigger: float = 0.25,
    ) -> ReportValidationResult:
        reasons: list[str] = []

        try:
            report = ClinicalReport.model_validate(report_payload)
        except Exception as exc:
            return ReportValidationResult(valid=False, reasons=[f"schema_error: {exc}"])

        missing_refs = [ref for ref in report.evidence_refs if ref not in retrieved_passages]
        if missing_refs:
            reasons.append(f"missing_evidence_refs: {missing_refs}")

        evidence_text = "\n".join(retrieved_passages.values())
        claims = [c.strip() for c in re.split(r"[.!?]", report.assessment) if c.strip()]
        unsupported_claims = [c for c in claims if self._token_overlap(c, evidence_text) < 0.2]
        if unsupported_claims:
            reasons.append(f"hallucination_risk_claims: {unsupported_claims[:3]}")

        if tft_interval_width >= uncertainty_trigger and len(report.uncertainty_note.strip()) == 0:
            reasons.append("missing_uncertainty_note_for_wide_interval")

        pii_issues = self._pii_scan(report.model_dump_json())
        if pii_issues:
            reasons.append(f"pii_leakage: {pii_issues}")

        return ReportValidationResult(valid=len(reasons) == 0, reasons=reasons)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 report validator")
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--evidence_json", required=True, help="JSON object mapping ref_id -> passage text")
    parser.add_argument("--tft_interval_width", type=float, default=0.0)
    args = parser.parse_args()

    report = json.loads(open(args.report_json, "r", encoding="utf-8").read())
    evidence = json.loads(open(args.evidence_json, "r", encoding="utf-8").read())

    validator = ClinicalReportValidator()
    res = validator.validate(report, evidence, tft_interval_width=args.tft_interval_width)
    print(json.dumps({"valid": res.valid, "reasons": res.reasons}, indent=2))


if __name__ == "__main__":
    _cli()
