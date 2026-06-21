# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""v4 Clinical Report Generator — RAG-enhanced SOAP with PubMed citations.

Extends ClinicalReportGeneratorV3. When the literature_embeddings pgvector
table is populated and OPENAI_API_KEY is set, this generator:

  1. Builds a patient-profile query string from clinical features.
  2. Retrieves top-5 relevant PubMed abstracts via pgvector cosine search.
  3. Injects the abstracts into the Claude prompt as numbered context.
  4. Instructs Claude to cite them inline as [PMIDxxxxxxx].
  5. Runs an extended hallucination guard that validates:
       a. Every percentage cited matches a known model probability (v3 rule)
       b. Every [PMID...] citation refers to an abstract we actually retrieved
          (prevents Claude from hallucinating PMIDs).
  6. Falls back to ClinicalReportGeneratorV3 (plain LLM SOAP) when RAG is
     unavailable, and to v2 (deterministic template) when the LLM is unavailable.

The ``generate_report`` signature is identical to v3 so it can be dropped in
anywhere that currently uses ClinicalReportGeneratorV3.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from backend.report_generator_v3 import ClinicalReportGeneratorV3, HallucinationError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# System prompt extension — RAG citation rules appended to v3 prompt
# -------------------------------------------------------------------

RAG_CITATION_ADDENDUM = """
8. You have been provided with numbered PubMed abstracts above the patient data.
   When a claim is supported by one of those abstracts, cite it inline as
   [PMID<number>] (e.g. [PMID12345678]). Use at most 5 citations total.
9. NEVER invent a PMID. Only cite PMIDs explicitly listed in the provided
   literature context. Do not use any other citation format.
10. A citation is REQUIRED in the Assessment section if the predicted disease
    has a prevalence or diagnostic criterion mentioned in the provided abstracts.
"""


class ClinicalReportGeneratorV4(ClinicalReportGeneratorV3):
    """RAG-enhanced SOAP report generator with PubMed inline citations."""

    def __init__(
        self,
        hf_token: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        db: Any = None,
    ) -> None:
        super().__init__(hf_token=hf_token, api_key=api_key, model=model)
        self._db = db
        self._rag = self._init_rag()

    def _init_rag(self):
        if self._db is None:
            return None
        try:
            from src.neurosynth.llm.rag_v2 import PubMedRAG
            rag = PubMedRAG(db=self._db, openai_api_key=self.api_key)
            if rag.enabled:
                logger.info("rag_v2_initialized openai_key_present=True")
            else:
                logger.info("rag_v2_initialized openai_key_present=False (RAG disabled)")
            return rag
        except Exception as exc:
            logger.warning("rag_v2_init_failed error=%s", exc)
            return None

    @property
    def rag_enabled(self) -> bool:
        return self._rag is not None and self._rag.enabled

    # ------------------------------------------------------------------
    # Core generate override
    # ------------------------------------------------------------------

    def generate_report(
        self,
        patient_data: dict[str, Any],
        prediction: dict[str, Any],
        trajectory: list[float],
        causal_graph: dict[str, Any],
        shap_values: list[dict[str, Any]],
        patient_id: str = "P-000",
        disease: str | None = None,
    ) -> dict[str, Any]:
        """Generate SOAP report, upgrading to RAG when available."""

        # Always build the v3 report first (provides structured fallback)
        report = super().generate_report(
            patient_data, prediction, trajectory, causal_graph,
            shap_values, patient_id, disease,
        )

        if not self.llm_enabled or not self.rag_enabled:
            return report

        # ---- RAG path ------------------------------------------------
        try:
            from src.neurosynth.llm.rag_v2 import PubMedRAG
            query = PubMedRAG.build_patient_query(patient_data, disease)
            docs = self._rag.retrieve_sync(query, top_k=5)
        except Exception as exc:
            logger.warning("rag_retrieve_failed error=%s", exc)
            docs = []

        if not docs:
            # No abstracts found (empty table or embedding failed) → v3 result
            return report

        try:
            soap = self._generate_soap_rag(
                patient_data, prediction, trajectory, causal_graph,
                shap_values, disease, docs,
            )
            self._verify_rag_report(soap, prediction, docs)
        except HallucinationError as exc:
            logger.warning("rag_hallucination_guard_tripped error=%s", exc)
            return report
        except Exception as exc:
            logger.warning("rag_report_generation_failed error=%s", exc)
            return report

        # Merge RAG SOAP into report dict (same structure as v3)
        report.update(self._merge_llm_soap(report, soap, prediction, trajectory, shap_values))
        report["generated_by"] = f"claude:{self.model}:rag"
        report["rag_citations"] = self._extract_citations(soap)
        report["rag_docs_retrieved"] = len(docs)
        return report

    # ------------------------------------------------------------------
    # RAG-enhanced LLM call
    # ------------------------------------------------------------------

    def _generate_soap_rag(
        self,
        patient_data: dict[str, Any],
        prediction: dict[str, Any],
        trajectory: list[float],
        causal_graph: dict[str, Any],
        shap_values: list[dict[str, Any]],
        disease: str | None,
        docs: list[dict[str, Any]],
    ) -> dict[str, str]:
        from src.neurosynth.llm.rag_v2 import PubMedRAG

        rag_context = PubMedRAG.format_context(docs)

        # Build the same patient summary as v3 but prepend literature context
        prob = float(prediction.get("probability", 0.5))
        risk = prediction.get("risk_level", "Unknown")
        cond = disease or prediction.get("disease", "Alzheimer's Disease")
        top_shap = [
            f"{r.get('feature')} ({float(r.get('value', 0.0)):+.4f})"
            for r in shap_values[:5]
        ] or ["none available"]
        top_causes = [
            c.get("variable")
            for c in causal_graph.get("top_causes_of_Diagnosis", [])
        ][:3] or ["none available"]
        traj_summary = ""
        if trajectory:
            months = [6, 12, 18, 24, 30, 36, 42, 48][: len(trajectory)]
            traj_summary = ", ".join(f"month {m}: {v:.1%}" for m, v in zip(months, trajectory))

        user_content = (
            f"{rag_context}\n"
            "---\n"
            "Generate a clinical SOAP note from this neurological risk analysis.\n\n"
            f"Suspected condition: {cond}\n"
            f"Model risk probability: {prob:.1%} ({risk} risk) — this is the ONLY "
            "percentage you may cite.\n"
            f"Top risk factors (SHAP): {', '.join(top_shap)}\n"
            f"Causal drivers: {', '.join(str(c) for c in top_causes)}\n"
            f"48-month trajectory: {traj_summary or 'not available'}\n"
            f"Patient: Age {patient_data.get('Age', 'N/A')}, "
            f"MMSE {patient_data.get('MMSE', 'N/A')}/30, "
            f"Functional assessment {patient_data.get('FunctionalAssessment', 'N/A')}/10, "
            f"ADL {patient_data.get('ADL', 'N/A')}/10."
        )

        # Build system prompt: v3 base + RAG citation rules
        from backend.report_generator_v3 import SOAP_SYSTEM_PROMPT
        rag_system = SOAP_SYSTEM_PROMPT + RAG_CITATION_ADDENDUM

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1800,
            system=rag_system,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(
            getattr(block, "text", "") for block in message.content
            if getattr(block, "type", "") == "text"
        )
        return self._parse_soap_json(text)

    # ------------------------------------------------------------------
    # Extended hallucination guard
    # ------------------------------------------------------------------

    def _verify_rag_report(
        self,
        soap: dict[str, str],
        prediction: dict[str, Any],
        docs: list[dict[str, Any]],
        tol: float = 0.12,
    ) -> None:
        """Run v3 percentage guard + v4 PMID citation guard."""
        # v3 guard: percentages must match model probabilities
        self._verify_report_facts(soap, prediction, tol=tol)

        # v4 guard: every cited PMID must be in the retrieved set
        retrieved_pmids = {str(d.get("pmid", "")) for d in docs}
        soap_text = " ".join(soap.values())
        cited_pmids = set(re.findall(r"\[PMID(\d+)\]", soap_text, re.IGNORECASE))

        hallucinated = cited_pmids - retrieved_pmids
        if hallucinated:
            raise HallucinationError(
                f"Report cites PMIDs not in retrieved set: {hallucinated}. "
                f"Retrieved: {retrieved_pmids}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_citations(soap: dict[str, str]) -> list[str]:
        """Return unique PMIDs cited in the SOAP text, in order of first appearance."""
        text = " ".join(soap.values())
        seen: set[str] = set()
        ordered: list[str] = []
        for pmid in re.findall(r"\[PMID(\d+)\]", text, re.IGNORECASE):
            if pmid not in seen:
                seen.add(pmid)
                ordered.append(pmid)
        return ordered
