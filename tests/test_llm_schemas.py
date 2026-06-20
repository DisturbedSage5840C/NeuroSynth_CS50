from __future__ import annotations

from datetime import datetime, timezone

from neurosynth.llm.schemas import ReportSchema


def test_report_schema_validation() -> None:
    payload = {
        "report_id": "abc",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_summary": {
            "disease_stage": "MCI",
            "progression_category": "moderate",
            "primary_biomarker_pattern": "tau-up pattern",
        },
        "deterioration_forecast": {
            "horizon_months": [6, 12, 18, 24, 30, 36],
            "dci_median": [20, 25, 30, 35, 40, 45],
            "dci_ci_80_lower": [15, 20, 25, 30, 35, 40],
            "dci_ci_80_upper": [25, 30, 35, 40, 45, 50],
            "months_to_clinical_threshold": {"estimate": 24, "ci_80": [18, 30]},
            "forecast_confidence": "moderate",
            "confidence_rationale": "Trajectory is coherent with measured biomarkers.",
        },
        "causal_analysis": {
            "primary_driver": {"variable": "ptau181", "causal_effect_on_dci": 0.6, "mechanistic_explanation": "tau-mediated degeneration"},
            "secondary_drivers": [{"variable": "nfl", "causal_effect": 0.3, "explanation": "axonal injury signal"}],
            "causal_pathway_narrative": "Amyloid-tau-neurodegeneration chain.",
        },
        "intervention_recommendations": [
            {
                "rank": 1,
                "target_variable": "sleep_efficiency",
                "intervention_description": "Sleep program",
                "estimated_dci_reduction_24mo": 4.2,
                "estimated_reduction_ci_80": [2.0, 6.0],
                "mechanism": "inflammation reduction",
                "evidence_strength": "observational",
                "supporting_pmids": ["12345678"],
                "contraindications": ["severe insomnia"],
                "monitoring_parameters": ["nfl"],
            }
        ],
        "monitoring_protocol": {
            "recommended_biomarkers": [{"biomarker": "nfl", "frequency_months": 6, "rationale": "monitor trend"}],
            "red_flag_thresholds": [{"variable": "dci", "threshold": 60, "action": "escalate"}],
            "next_review_months": 6,
        },
        "uncertainty_flags": ["model uncertainty present"],
        "disclaimer": "Not standalone diagnosis",
    }
    obj = ReportSchema.model_validate(payload)
    assert obj.patient_summary.progression_category == "moderate"
