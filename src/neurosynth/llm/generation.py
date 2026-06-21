# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from pydantic import ValidationError

from neurosynth.llm.schemas import ReportSchema


@dataclass
class ConstrainedReportGenerator:
    model_name: str = "neurosynth-llm-8b-merged"

    def __post_init__(self) -> None:
        # Heavy GPU-only deps (vllm, outlines, mlflow) are imported on instantiation,
        # not at module load — so this file can be collected by pytest / imported on
        # CPU-only hosts without trying to spin up a CUDA-bound vLLM server.
        try:
            import mlflow
            from outlines import generate
            from vllm import LLM

            from neurosynth.llm.pmid_verify import PMIDVerifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ConstrainedReportGenerator requires vllm + outlines + mlflow + biopython. "
                "Install the GPU extras to use it."
            ) from exc

        self._mlflow = mlflow
        self.llm = LLM(
            model=self.model_name,
            quantization="awq",
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=2,
        )
        self.gen = generate.json(self.llm, ReportSchema)
        self.pmid_verifier = PMIDVerifier()

    def _plausibility_score(self, report: ReportSchema) -> float:
        ok = 1.0
        med = report.deterioration_forecast.dci_median
        lo = report.deterioration_forecast.dci_ci_80_lower
        hi = report.deterioration_forecast.dci_ci_80_upper
        for m, l, h in zip(med, lo, hi):
            if not (0 <= m <= 100 and 0 <= l <= 100 and 0 <= h <= 100 and l < m < h):
                ok -= 0.2
        if report.deterioration_forecast.months_to_clinical_threshold.get("estimate", 0) <= 0:
            ok -= 0.2
        if len(report.uncertainty_flags) < 1:
            ok -= 0.2
        return max(0.0, ok)

    def _verify_pmids(self, report: ReportSchema) -> ReportSchema:
        for rec in report.intervention_recommendations:
            rec.supporting_pmids = self.pmid_verifier.filter_valid(rec.supporting_pmids)
        return report

    def generate_report(self, prompt: str) -> dict:
        t0 = time.time()
        raw = self.gen(prompt)

        try:
            report = ReportSchema.model_validate(raw if isinstance(raw, dict) else json.loads(raw))
        except ValidationError as e:
            raise RuntimeError(f"Schema validation failed: {e}")

        report = self._verify_pmids(report)
        score = self._plausibility_score(report)

        payload = report.model_dump()
        payload["plausibility_score"] = score
        payload["prompt_hash"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        payload["latency_seconds"] = time.time() - t0

        self._mlflow.log_metrics(
            {"plausibility_score": score, "latency_seconds": payload["latency_seconds"]}
        )
        return payload

    def batch_generate(self, prompts: list[str]) -> list[dict]:
        # vLLM handles continuous batching internally.
        return [self.generate_report(p) for p in prompts[:32]]
