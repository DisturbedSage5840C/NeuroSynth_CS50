# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    endpoint = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    # Configurable timeouts (seconds)
    connect_timeout: float = 10.0
    read_timeout: float = 45.0

    def __init__(self, hf_token: str | None = None) -> None:
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        if not self.hf_token:
            logger.info("ClinicalReportGenerator: no HF_TOKEN set, will use fallback reports")

    @staticmethod
    def _sections_from_text(text: str) -> dict[str, str]:
        section_titles = [
            "1. EXECUTIVE SUMMARY",
            "2. RISK ASSESSMENT & INTERPRETATION",
            "3. KEY BIOMARKER ANALYSIS",
            "4. 36-MONTH PROGRESSION FORECAST",
            "5. CAUSAL PATHWAY ANALYSIS",
            "6. MODIFIABLE RISK FACTORS & INTERVENTIONS",
            "7. MONITORING PROTOCOL",
            "8. LIFESTYLE OPTIMIZATION PLAN",
            "9. UNCERTAINTY & LIMITATIONS",
        ]

        lines = text.splitlines()
        sections: dict[str, list[str]] = {t: [] for t in section_titles}
        cur = section_titles[0]

        for raw in lines:
            line = raw.strip()
            hit = next((t for t in section_titles if line.upper().startswith(t.upper())), None)
            if hit:
                cur = hit
                rest = line[len(hit) :].strip(" :-")
                if rest:
                    sections[cur].append(rest)
                continue
            if line:
                sections[cur].append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items()}

    def _fallback_report(
        self,
        patient_data: dict[str, Any],
        prediction: dict[str, Any],
        trajectory: list[float],
        causal_graph: dict[str, Any],
        shap_values: list[dict[str, float]],
    ) -> dict[str, Any]:
        top_risks = ", ".join([f"{r['feature']} ({r['value']})" for r in shap_values[:5]]) or "MMSE, FunctionalAssessment"
        top_causes = ", ".join([c["variable"] for c in causal_graph.get("top_causes_of_Diagnosis", [])]) or "MMSE, Age"

        sections = {
            "1. EXECUTIVE SUMMARY": (
                f"The model estimates {prediction.get('probability', 0):.1%} Alzheimer's risk, categorized as "
                f"{prediction.get('risk_level', 'Unknown')} confidence {prediction.get('confidence', 'Unknown')}."
            ),
            "2. RISK ASSESSMENT & INTERPRETATION": (
                f"Current profile shows elevated risk drivers from {top_risks}. "
                f"Predicted class={prediction.get('prediction')} with probability {prediction.get('probability', 0):.1%}."
            ),
            "3. KEY BIOMARKER ANALYSIS": (
                f"Key factors include MMSE={patient_data.get('MMSE')}, FunctionalAssessment={patient_data.get('FunctionalAssessment')}, "
                f"ADL={patient_data.get('ADL')}, and cardiovascular/lifestyle measures."
            ),
            "4. 36-MONTH PROGRESSION FORECAST": (
                f"Projected risk trajectory across 6 to 36 months: {trajectory}. "
                "Higher slope indicates faster deterioration risk."
            ),
            "5. CAUSAL PATHWAY ANALYSIS": (
                f"Causal graph indicates diagnosis is strongly influenced by: {top_causes}."
            ),
            "6. MODIFIABLE RISK FACTORS & INTERVENTIONS": (
                "1) Increase physical activity and sleep quality. 2) Optimize blood pressure and lipid profile. "
                "3) Cognitive stimulation with regular MMSE tracking. 4) Depression screening/treatment. "
                "5) Reduce sedentary behavior and improve diet quality."
            ),
            "7. MONITORING PROTOCOL": (
                "Track MMSE/FunctionalAssessment/ADL every 3 months; blood pressure and lipid panel every 1 to 3 months; "
                "trigger escalation if MMSE drops >2 points in 6 months."
            ),
            "8. LIFESTYLE OPTIMIZATION PLAN": (
                "Weekly aerobic activity, Mediterranean-style diet, sleep hygiene protocol, and adherence checks for risk comorbidities."
            ),
            "9. UNCERTAINTY & LIMITATIONS": (
                "Model output is probabilistic and dataset-driven. This is a research tool; clinical decisions require physician oversight."
            ),
        }
        raw = "\n\n".join([f"{k}\n{v}" for k, v in sections.items()])
        return {
            "sections": sections,
            "raw_text": raw,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "word_count": len(raw.split()),
        }

    def generate_report(
        self,
        patient_data: dict[str, Any],
        prediction: dict[str, Any],
        trajectory: list[float],
        causal_graph: dict[str, Any],
        shap_values: list[dict[str, float]],
    ) -> dict[str, Any]:
        """Generate a clinical report.  Synchronous — safe to call from threads."""
        top_risk_factors = [f"{r['feature']} ({r['value']:+.4f})" for r in shap_values[:5]]
        causal_summary = ", ".join([f"{x['variable']}:{x['strength']}" for x in causal_graph.get("top_causes_of_Diagnosis", [])])

        prompt = f"""[INST] You are NeuroSynth, an advanced neurological AI assistant. Generate a detailed, structured clinical assessment.
PATIENT PROFILE:
Age: {patient_data.get('Age')} | Gender: {patient_data.get('Gender')} | Education: {patient_data.get('EducationLevel')} years
BMI: {patient_data.get('BMI')} | Smoking: {patient_data.get('Smoking')} | Physical Activity: {patient_data.get('PhysicalActivity')}/10
Sleep Quality: {patient_data.get('SleepQuality')}/10 | Diet Quality: {patient_data.get('DietQuality')}/10
Family History of Alzheimer's: {patient_data.get('FamilyHistoryAlzheimers')}
Cardiovascular Disease: {patient_data.get('CardiovascularDisease')} | Diabetes: {patient_data.get('Diabetes')} | Depression: {patient_data.get('Depression')}
CLINICAL MEASUREMENTS:
MMSE Score: {patient_data.get('MMSE')}/30 | Functional Assessment: {patient_data.get('FunctionalAssessment')}/10 | ADL Score: {patient_data.get('ADL')}/10
Blood Pressure: {patient_data.get('SystolicBP')}/{patient_data.get('DiastolicBP')} mmHg
Cholesterol: Total={patient_data.get('CholesterolTotal')} LDL={patient_data.get('CholesterolLDL')} HDL={patient_data.get('CholesterolHDL')} Triglycerides={patient_data.get('CholesterolTriglycerides')}
SYMPTOMS:
Memory Complaints: {patient_data.get('MemoryComplaints')} | Behavioral Problems: {patient_data.get('BehavioralProblems')}
Confusion: {patient_data.get('Confusion')} | Disorientation: {patient_data.get('Disorientation')}
Personality Changes: {patient_data.get('PersonalityChanges')} | Forgetfulness: {patient_data.get('Forgetfulness')}
AI ANALYSIS RESULTS:
Risk Probability: {prediction.get('probability', 0):.1%} ({prediction.get('risk_level', 'Unknown')} risk)
Confidence: {prediction.get('confidence', 'Unknown')}
Top Risk Factors by SHAP: {top_risk_factors}
36-Month Trajectory: {trajectory}
Causal Analysis: {causal_summary}
Generate a comprehensive neurological assessment report with these sections:
1. EXECUTIVE SUMMARY
2. RISK ASSESSMENT & INTERPRETATION
3. KEY BIOMARKER ANALYSIS (explain each major risk factor)
4. 36-MONTH PROGRESSION FORECAST
5. CAUSAL PATHWAY ANALYSIS
6. MODIFIABLE RISK FACTORS & INTERVENTIONS (top 5 specific, actionable recommendations with expected impact)
7. MONITORING PROTOCOL (specific biomarkers, frequency, thresholds)
8. LIFESTYLE OPTIMIZATION PLAN
9. UNCERTAINTY & LIMITATIONS
Be specific, cite the patient's actual numbers, and make every recommendation actionable.
IMPORTANT: This is a research tool. Always include disclaimer that clinical decisions require a physician. [/INST]"""

        if not self.hf_token:
            return self._fallback_report(patient_data, prediction, trajectory, causal_graph, shap_values)

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        request_payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 900,
                "temperature": 0.25,
                "return_full_text": False,
            },
        }

        try:
            timeout = httpx.Timeout(
                connect=self.connect_timeout,
                read=self.read_timeout,
                write=10.0,
                pool=5.0,
            )
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(self.endpoint, headers=headers, json=request_payload)
                resp.raise_for_status()
                body = resp.json()

            if isinstance(body, list) and body and isinstance(body[0], dict):
                raw_text = body[0].get("generated_text", "")
            elif isinstance(body, dict):
                raw_text = body.get("generated_text", str(body))
            else:
                raw_text = str(body)

            sections = self._sections_from_text(raw_text)
            return {
                "sections": sections,
                "raw_text": raw_text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "word_count": len(raw_text.split()),
            }
        except httpx.TimeoutException:
            logger.warning("LLM request timed out, serving fallback report")
            return self._fallback_report(patient_data, prediction, trajectory, causal_graph, shap_values)
        except Exception as e:
            logger.warning("LLM request failed (%s), serving fallback report", e)
            return self._fallback_report(patient_data, prediction, trajectory, causal_graph, shap_values)

    async def generate_report_async(
        self,
        patient_data: dict[str, Any],
        prediction: dict[str, Any],
        trajectory: list[float],
        causal_graph: dict[str, Any],
        shap_values: list[dict[str, float]],
    ) -> dict[str, Any]:
        """Async version of generate_report for use in async contexts."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.generate_report,
            patient_data,
            prediction,
            trajectory,
            causal_graph,
            shap_values,
        )
