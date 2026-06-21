from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import gradio as gr
import pandas as pd

logger = logging.getLogger(__name__)

# =========================================================================
# DEPRECATION NOTICE
# =========================================================================
# This Gradio interface is DEPRECATED in favor of the React frontend.
# It imports backend classes directly, causing double model loading when
# run alongside the FastAPI server. Use the React frontend at
# http://localhost:5173 instead.
#
# To run the FastAPI backend only:  uvicorn backend.api:app
# To run the React frontend:       cd frontend && npm run dev
# =========================================================================
warnings.warn(
    "app.py (Gradio UI) is deprecated. Use the React frontend instead. "
    "See README.md for instructions.",
    DeprecationWarning,
    stacklevel=1,
)

from backend.biomarker_model import BiomarkerPredictor
from backend.causal_engine import NeuralCausalDiscovery
from backend.data_pipeline import DataPipeline
from backend.report_generator import ClinicalReportGenerator
from backend.temporal_model import TemporalProgressionModel

STATE: dict[str, Any] = {}

FIELDS = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking", "AlcoholConsumption",
    "PhysicalActivity", "DietQuality", "SleepQuality", "FamilyHistoryAlzheimers", "CardiovascularDisease",
    "Diabetes", "Depression", "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
    "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL", "Confusion",
    "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness",
]

DEFAULTS = {
    "Age": 73, "Gender": 1, "Ethnicity": 1, "EducationLevel": 1, "BMI": 27.4, "Smoking": 0,
    "AlcoholConsumption": 3.2, "PhysicalActivity": 4.8, "DietQuality": 5.2, "SleepQuality": 5.0,
    "FamilyHistoryAlzheimers": 1, "CardiovascularDisease": 0, "Diabetes": 0, "Depression": 0,
    "HeadInjury": 0, "Hypertension": 1, "SystolicBP": 132, "DiastolicBP": 82,
    "CholesterolTotal": 202, "CholesterolLDL": 124, "CholesterolHDL": 49, "CholesterolTriglycerides": 166,
    "MMSE": 24, "FunctionalAssessment": 6.2, "MemoryComplaints": 1, "BehavioralProblems": 0, "ADL": 6.0,
    "Confusion": 0, "Disorientation": 0, "PersonalityChanges": 0, "DifficultyCompletingTasks": 1,
    "Forgetfulness": 1,
}

RANGES = {
    "Age": (45, 100, 1), "Gender": (0, 2, 1), "Ethnicity": (0, 3, 1), "EducationLevel": (0, 3, 1),
    "BMI": (15, 45, 0.1), "Smoking": (0, 1, 1), "AlcoholConsumption": (0, 20, 0.1),
    "PhysicalActivity": (0, 10, 0.1), "DietQuality": (0, 10, 0.1), "SleepQuality": (0, 10, 0.1),
    "FamilyHistoryAlzheimers": (0, 1, 1), "CardiovascularDisease": (0, 1, 1), "Diabetes": (0, 1, 1),
    "Depression": (0, 1, 1), "HeadInjury": (0, 1, 1), "Hypertension": (0, 1, 1),
    "SystolicBP": (80, 220, 1), "DiastolicBP": (40, 140, 1), "CholesterolTotal": (100, 400, 1),
    "CholesterolLDL": (40, 300, 1), "CholesterolHDL": (20, 120, 1), "CholesterolTriglycerides": (40, 500, 1),
    "MMSE": (0, 30, 1), "FunctionalAssessment": (0, 10, 0.1), "MemoryComplaints": (0, 1, 1),
    "BehavioralProblems": (0, 1, 1), "ADL": (0, 10, 0.1), "Confusion": (0, 1, 1),
    "Disorientation": (0, 1, 1), "PersonalityChanges": (0, 1, 1), "DifficultyCompletingTasks": (0, 1, 1),
    "Forgetfulness": (0, 1, 1),
}


def _init() -> None:
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test, feature_names, scaler, dataset_stats = pipeline.process()

    predictor = BiomarkerPredictor(feature_names)
    predictor.train(X_train.values, y_train.values)

    temporal = TemporalProgressionModel(feature_names)
    temporal.train_model(X_train.values, y_train.values)

    causal = NeuralCausalDiscovery()
    if pipeline.df_processed is not None:
        causal_cols = [c for c in causal.variables if c in pipeline.df_processed.columns]
        if len(causal_cols) == len(causal.variables):
            causal.fit(pipeline.df_processed[causal.variables].values.astype(float))

    hf_token = os.getenv("HF_TOKEN") or None  # Ensure None, not empty string
    reporter = ClinicalReportGenerator(hf_token)

    metrics = predictor.evaluate(X_test.values, y_test.values)
    STATE.update(
        {
            "pipeline": pipeline,
            "predictor": predictor,
            "temporal": temporal,
            "causal": causal,
            "reporter": reporter,
            "scaler": scaler,
            "feature_names": feature_names,
            "dataset_stats": dataset_stats,
            "metrics": metrics,
        }
    )


def _to_payload(values: list[float]) -> dict[str, float]:
    return {k: float(v) for k, v in zip(FIELDS, values)}


def analyze(*values: float):
    _ensure_initialized()
    payload = _to_payload(list(values))
    frame = pd.DataFrame([payload])
    scaled = STATE["scaler"].transform(frame[STATE["feature_names"]])

    pred = STATE["predictor"].predict(scaled)
    traj = STATE["temporal"].predict_trajectory(frame[STATE["feature_names"]].values[0], pred["probability"])
    traj_df = pd.DataFrame({"month": [6, 12, 18, 24, 30, 36], "risk": traj["trajectory"]})

    return {
        **pred,
        "trajectory": traj["trajectory"],
        "confidence_bands": traj["confidence_bands"],
    }, traj_df


def clinical_report(*values: float):
    _ensure_initialized()
    payload = _to_payload(list(values))
    frame = pd.DataFrame([payload])
    scaled = STATE["scaler"].transform(frame[STATE["feature_names"]])

    pred = STATE["predictor"].predict(scaled)
    traj = STATE["temporal"].predict_trajectory(frame[STATE["feature_names"]].values[0], pred["probability"])
    shap_vals = STATE["predictor"].get_shap_values(scaled[:1])[0]
    top_idx = list(abs(shap_vals).argsort()[::-1][:5])
    shap_top = [{"feature": STATE["feature_names"][i], "value": float(shap_vals[i])} for i in top_idx]

    report = STATE["reporter"].generate_report(
        patient_data=payload,
        prediction=pred,
        trajectory=traj["trajectory"],
        causal_graph=STATE["causal"].get_causal_graph(),
        shap_values=shap_top,
    )

    md = []
    for title, content in report["sections"].items():
        md.append(f"### {title}\n{content}")
    return "\n\n".join(md)


def show_causal():
    _ensure_initialized()
    graph = STATE["causal"].get_causal_graph()
    edges = pd.DataFrame(graph.get("edges", []))
    insight = (
        "### Top Causes of Diagnosis\n"
        + "\n".join([f"- {x['variable']} ({x['strength']})" for x in graph.get("top_causes_of_Diagnosis", [])])
        + "\n\n### Protective Factors\n"
        + "\n".join([f"- {x['variable']} ({x['effect']})" for x in graph.get("protective_factors", [])])
        + "\n\n### Risk Amplifiers\n"
        + "\n".join([f"- {x['variable']} ({x['effect']})" for x in graph.get("risk_amplifiers", [])])
    )
    return edges, insight


def perf_text():
    m = STATE["metrics"]
    return (
        f"## Model Performance\n"
        f"- Accuracy: {m['accuracy']}\n"
        f"- F1 Weighted: {m['f1_weighted']}\n"
        f"- ROC-AUC: {m['roc_auc']}\n"
        f"- Precision: {m['precision']}\n"
        f"- Recall: {m['recall']}\n"
    )


_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        logger.info("Initializing Gradio backend (deprecated — use React frontend)...")
        _init()
        _initialized = True

with gr.Blocks(
    title="🧠 NeuroSynth — Advanced Neurological AI Platform",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# 🧠 NeuroSynth — Advanced Neurological AI Platform")
    gr.Markdown(
        "- 4-model ensemble biomarker prediction\n"
        "- Pseudo-longitudinal LSTM progression forecasting\n"
        "- Neural causal discovery + intervention simulation\n"
        "- LLM-generated structured neurological reports"
    )

    sliders: list[gr.Slider] = []

    def make_inputs():
        groups = {
            "Demographics": ["Age", "Gender", "Ethnicity", "EducationLevel"],
            "Lifestyle": ["BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality"],
            "Medical History": ["FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension"],
            "Clinical Measurements": ["SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "ADL"],
            "Symptoms": ["MemoryComplaints", "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"],
        }
        local = []
        for section, names in groups.items():
            gr.Markdown(f"### {section}")
            with gr.Row():
                for n in names:
                    lo, hi, st = RANGES[n]
                    local.append(gr.Slider(lo, hi, value=DEFAULTS[n], step=st, label=n))
        return local

    with gr.Tab("🔬 Patient Analysis"):
        sliders = make_inputs()
        analyze_btn = gr.Button("Analyze", variant="primary")
        pred_json = gr.JSON(label="Prediction")
        traj_plot = gr.LinePlot(x="month", y="risk", title="36-Month Trajectory")
        analyze_btn.click(analyze, inputs=sliders, outputs=[pred_json, traj_plot])

    with gr.Tab("📋 Clinical Report"):
        report_inputs = make_inputs()
        report_btn = gr.Button("Generate Report", variant="primary")
        report_md = gr.Markdown()
        report_btn.click(clinical_report, inputs=report_inputs, outputs=report_md)

    with gr.Tab("🕸️ Causal Network"):
        graph_btn = gr.Button("Show Graph", variant="primary")
        edges_df = gr.Dataframe(label="Causal Edges", interactive=False)
        insight_md = gr.Markdown()
        graph_btn.click(show_causal, inputs=None, outputs=[edges_df, insight_md])

    with gr.Tab("📊 Model Performance"):
        perf_md = gr.Markdown(perf_text())

if __name__ == "__main__":
    demo.launch()
