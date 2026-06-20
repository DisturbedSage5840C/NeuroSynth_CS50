from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from sklearn.metrics import cohen_kappa_score

from neurosynth.llm.pmid_verify import PMIDVerifier
from neurosynth.llm.schemas import ReportSchema

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class NeuroLLMEvaluator:
    def __init__(self) -> None:
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.pmid = PMIDVerifier()

    def automated_metrics(self, refs: list[str], preds: list[str]) -> dict[str, float]:
        rouge_l = np.mean([self.rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)])
        _, _, f1 = bertscore(preds, refs, lang="en", model_type="microsoft/deberta-v3-large")
        return {"rougeL": float(rouge_l), "bertscore_f1": float(f1.mean().item())}

    def json_validity_rate(self, outputs: list[str]) -> float:
        ok = 0
        for o in outputs:
            try:
                ReportSchema.model_validate(json.loads(o) if isinstance(o, str) else o)
                ok += 1
            except Exception:
                pass
        return ok / max(len(outputs), 1)

    def pmid_hallucination_rate(self, outputs: list[str]) -> float:
        total = 0
        bad = 0
        for o in outputs:
            try:
                report = ReportSchema.model_validate(json.loads(o) if isinstance(o, str) else o)
            except Exception:
                continue
            for rec in report.intervention_recommendations:
                for pmid in rec.supporting_pmids:
                    total += 1
                    if not self.pmid.is_valid(pmid):
                        bad += 1
        return bad / max(total, 1)

    def clinical_plausibility_score(self, outputs: list[str]) -> float:
        scores = []
        for o in outputs:
            try:
                report = ReportSchema.model_validate(json.loads(o) if isinstance(o, str) else o)
            except Exception:
                scores.append(0.0)
                continue
            ok = 1.0
            for m, l, h in zip(
                report.deterioration_forecast.dci_median,
                report.deterioration_forecast.dci_ci_80_lower,
                report.deterioration_forecast.dci_ci_80_upper,
            ):
                if not (0 <= l < m < h <= 100):
                    ok -= 0.25
            if report.deterioration_forecast.months_to_clinical_threshold.get("estimate", 0) <= 0:
                ok -= 0.25
            if len(report.uncertainty_flags) < 1:
                ok -= 0.25
            scores.append(max(0.0, ok))
        return float(np.mean(scores) if scores else 0.0)

    def write_label_studio_config(self, out_path: Path) -> None:
        xml = """<View>
  <Text name=\"report\" value=\"$report\"/>
  <Header value=\"Clinical Accuracy\"/>
  <Rating name=\"clinical_accuracy\" toName=\"report\" maxRating=\"5\"/>
  <Header value=\"Intervention Quality\"/>
  <Rating name=\"intervention_quality\" toName=\"report\" maxRating=\"5\"/>
  <Header value=\"Uncertainty Calibration\"/>
  <Rating name=\"uncertainty_calibration\" toName=\"report\" maxRating=\"5\"/>
  <Header value=\"Actionability\"/>
  <Rating name=\"actionability\" toName=\"report\" maxRating=\"5\"/>
  <Header value=\"Hallucination\"/>
  <Rating name=\"hallucination\" toName=\"report\" maxRating=\"5\"/>
</View>"""
        out_path.write_text(xml, encoding="utf-8")

    def geval_agreement(self, reports: list[dict], human_scores: pd.DataFrame) -> dict[str, float]:
        dimensions = ["clinical_accuracy", "intervention_quality", "uncertainty_calibration", "actionability", "hallucination"]
        rows: list[dict[str, float]] = []
        for report in reports:
            geval = report.get("g_eval", {}) if isinstance(report, dict) else {}
            row = {}
            for dim in dimensions:
                val = geval.get(dim)
                row[dim] = float(val) if isinstance(val, (int, float)) else np.nan
            rows.append(row)

        geval_df = pd.DataFrame(rows)
        kappas: dict[str, float] = {}
        for dim in dimensions:
            if dim not in human_scores.columns or dim not in geval_df.columns:
                kappas[dim] = 0.0
                continue

            pair = pd.DataFrame({"human": human_scores[dim], "model": geval_df[dim]}).dropna()
            if pair.empty:
                kappas[dim] = 0.0
                continue

            human_rounded = pair["human"].round().clip(1, 5).astype(int)
            model_rounded = pair["model"].round().clip(1, 5).astype(int)
            kappas[dim] = float(cohen_kappa_score(human_rounded, model_rounded, weights="quadratic"))

        return kappas
