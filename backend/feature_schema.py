"""Human-readable feature schema for NeuroSynth inputs (Gap 7).

A single source of truth describing every clinical input feature: a human-readable
label, full name, type, valid range / categorical encoding, unit, and a short
clinical note. Served to the frontend via ``GET /v2/features/schema`` so feature
encoding legends (Gender 0/1/2, Ethnicity, etc.) and SHAP labels are never opaque.
"""
from __future__ import annotations

from typing import Any, Literal

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
FeatureType = Literal["continuous", "categorical", "boolean"]


def _f(
    label: str,
    full_name: str,
    ftype: FeatureType,
    *,
    unit: str | None = None,
    rng: tuple[float, float] | None = None,
    values: dict[int, str] | None = None,
    note: str = "",
) -> dict[str, Any]:
    entry: dict[str, Any] = {"label": label, "full_name": full_name, "type": ftype}
    if unit is not None:
        entry["unit"] = unit
    if rng is not None:
        entry["range"] = list(rng)
    if values is not None:
        entry["values"] = {str(k): v for k, v in values.items()}
    if note:
        entry["clinical_note"] = note
    return entry


# Keyed by the exact feature column name used by the model/data pipeline.
FEATURE_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "Age": _f("Age", "Age", "continuous", unit="years", rng=(45, 100)),
    "Gender": _f("Sex", "Biological sex", "categorical",
                 values={0: "Female", 1: "Male", 2: "Other / Not specified"}),
    "Ethnicity": _f("Ethnicity", "Ethnicity", "categorical",
                    values={0: "White", 1: "Black / African American", 2: "Hispanic / Latino",
                            3: "Asian / Other"}),
    "EducationLevel": _f("Education level", "Highest education level", "categorical",
                         values={0: "None", 1: "High school", 2: "Some college",
                                 3: "University degree+"},
                         note="Lower education is associated with higher risk."),
    "BMI": _f("Body mass index", "Body mass index", "continuous", unit="kg/m²", rng=(15, 45),
              note="<18.5 underweight, >30 obese."),
    "Smoking": _f("Current smoker", "Current smoker", "boolean",
                  values={0: "No", 1: "Yes"}),
    "AlcoholConsumption": _f("Alcohol consumption", "Alcohol consumption", "continuous",
                             unit="drinks/week", rng=(0, 20)),
    "PhysicalActivity": _f("Physical activity", "Weekly physical activity", "continuous",
                           unit="hrs/week", rng=(0, 10)),
    "DietQuality": _f("Diet quality", "Diet quality score", "continuous", unit="score (0–10)",
                      rng=(0, 10), note="Higher = healthier."),
    "SleepQuality": _f("Sleep quality", "Sleep quality score", "continuous", unit="score (0–10)",
                       rng=(0, 10), note="Higher = better."),
    "FamilyHistoryAlzheimers": _f("Family history: Alzheimer's", "Family history of Alzheimer's",
                                  "boolean", values={0: "No", 1: "Yes"},
                                  note="First-degree relative."),
    "CardiovascularDisease": _f("Cardiovascular disease", "Cardiovascular disease", "boolean",
                                values={0: "No", 1: "Yes"}),
    "Diabetes": _f("Type 2 diabetes", "Type 2 diabetes", "boolean", values={0: "No", 1: "Yes"}),
    "Depression": _f("Clinical depression", "Diagnosed clinical depression", "boolean",
                     values={0: "No", 1: "Yes"}),
    "HeadInjury": _f("Head injury history", "History of head injury", "boolean",
                     values={0: "No", 1: "Yes"}, note="TBI / concussion."),
    "Hypertension": _f("Hypertension", "Hypertension", "boolean", values={0: "No", 1: "Yes"},
                       note="Diagnosed or medicated."),
    "SystolicBP": _f("Systolic blood pressure", "Systolic blood pressure", "continuous",
                     unit="mmHg", rng=(80, 220), note="Normal <120."),
    "DiastolicBP": _f("Diastolic blood pressure", "Diastolic blood pressure", "continuous",
                      unit="mmHg", rng=(40, 140), note="Normal <80."),
    "CholesterolTotal": _f("Total cholesterol", "Total cholesterol", "continuous", unit="mg/dL",
                           rng=(100, 400), note="Desirable <200."),
    "CholesterolLDL": _f("LDL cholesterol", "LDL cholesterol", "continuous", unit="mg/dL",
                         rng=(40, 300), note="Optimal <100."),
    "CholesterolHDL": _f("HDL cholesterol", "HDL cholesterol", "continuous", unit="mg/dL",
                         rng=(20, 120), note="High = protective."),
    "CholesterolTriglycerides": _f("Triglycerides", "Triglycerides", "continuous", unit="mg/dL",
                                   rng=(40, 500), note="Normal <150."),
    "MMSE": _f("MMSE score", "Mini-Mental State Examination", "continuous", unit="points (0–30)",
               rng=(0, 30), note="Cognitive screening. <24 suggests impairment."),
    "FunctionalAssessment": _f("Functional assessment", "Functional assessment", "continuous",
                               unit="score (0–10)", rng=(0, 10), note="Higher = more independent."),
    "MemoryComplaints": _f("Memory complaints", "Memory complaints", "boolean",
                           values={0: "No", 1: "Yes"}, note="Self or carer reported."),
    "BehavioralProblems": _f("Behavioral problems", "Behavioral problems", "boolean",
                             values={0: "No", 1: "Yes"}),
    "ADL": _f("Activities of daily living", "Activities of daily living", "continuous",
              unit="score (0–10)", rng=(0, 10), note="Higher = more independent."),
    "Confusion": _f("Confusion episodes", "Confusion episodes", "boolean",
                    values={0: "No", 1: "Yes"}),
    "Disorientation": _f("Disorientation", "Disorientation", "boolean", values={0: "No", 1: "Yes"},
                         note="Person, place, or time."),
    "PersonalityChanges": _f("Personality changes", "Personality changes", "boolean",
                             values={0: "No", 1: "Yes"}, note="Compared to baseline."),
    "DifficultyCompletingTasks": _f("Difficulty with tasks", "Difficulty completing tasks",
                                    "boolean", values={0: "No", 1: "Yes"},
                                    note="Previously routine tasks."),
    "Forgetfulness": _f("Forgetfulness", "Forgetfulness", "boolean", values={0: "No", 1: "Yes"},
                        note="Beyond normal aging."),
}


def get_feature_schema() -> dict[str, Any]:
    """Return the full feature schema as a JSON-serializable payload."""
    fields = []
    for key, meta in FEATURE_DESCRIPTIONS.items():
        fields.append({"key": key, **meta})
    return {
        "version": "v3",
        "count": len(fields),
        "fields": fields,
    }


def human_label(feature_key: str) -> str:
    """Map a raw feature key to a human-readable label (with unit when continuous)."""
    meta = FEATURE_DESCRIPTIONS.get(feature_key)
    if not meta:
        return feature_key
    label = meta["label"]
    unit = meta.get("unit")
    if unit and meta["type"] == "continuous" and "(" not in unit:
        return f"{label} ({unit})"
    return label
