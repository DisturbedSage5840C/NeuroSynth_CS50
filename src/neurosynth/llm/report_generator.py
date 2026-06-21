# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import pandera as pa
from pydantic import BaseModel, Field


class CausalPathway(BaseModel):
    pathway: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class Intervention(BaseModel):
    name: str
    action: str
    priority: int = Field(ge=1, le=10)
    expected_impact: str


class ClinicalReport(BaseModel):
    assessment: str
    risk_level: Literal["critical", "high", "moderate", "low"]
    causal_pathways: list[CausalPathway]
    interventions: list[Intervention]
    uncertainty_note: str
    evidence_refs: list[str]


REPORT_SCHEMA = pa.DataFrameSchema(
    {
        "assessment": pa.Column(str),
        "risk_level": pa.Column(str, checks=pa.Check.isin(["critical", "high", "moderate", "low"])),
        "uncertainty_note": pa.Column(str),
    },
    strict=False,
)


@dataclass
class Phase6ReportGenerator:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"

    def _load_outlines_model(self):
        import outlines

        try:
            import outlines.models as om

            return om.transformers(self.model_name)
        except Exception:
            # Fallback to text generation model wrapper in older/newer outlines layouts.
            return outlines.models.transformers(self.model_name)

    def generate(
        self,
        question: str,
        structured_context: str,
        evidence_refs: list[str],
        tft_interval_width: float,
        uncertainty_trigger: float = 0.25,
    ) -> ClinicalReport:
        import outlines

        model = self._load_outlines_model()
        generator = outlines.generate.json(model, ClinicalReport)

        prompt = (
            "You are NeuroSynth clinical reasoning engine. Produce a valid JSON ClinicalReport.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{structured_context}\n\n"
            f"Allowed evidence refs: {evidence_refs}\n"
        )

        raw = generator(prompt)
        report = ClinicalReport.model_validate(raw if isinstance(raw, dict) else json.loads(raw))

        if tft_interval_width >= uncertainty_trigger and not report.uncertainty_note.strip():
            report.uncertainty_note = (
                "Forecast uncertainty is elevated due to wide prediction intervals; treat recommendations as decision support only."
            )

        flat = pd.DataFrame([report.model_dump()])
        REPORT_SCHEMA.validate(flat)

        missing = [ref for ref in report.evidence_refs if ref not in set(evidence_refs)]
        if missing:
            raise ValueError(f"Invalid evidence_refs not present in retrieved passages: {missing}")

        return report


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 schema-constrained report generation")
    parser.add_argument("--question", required=True)
    parser.add_argument("--context_json", required=True, help="JSON with fields: structured_context,evidence_refs,tft_interval_width")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    payload = json.loads(open(args.context_json, "r", encoding="utf-8").read())
    gen = Phase6ReportGenerator(model_name=args.model_name)
    report = gen.generate(
        question=args.question,
        structured_context=str(payload["structured_context"]),
        evidence_refs=[str(x) for x in payload["evidence_refs"]],
        tft_interval_width=float(payload.get("tft_interval_width", 0.0)),
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    _cli()
