# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from Bio import Entrez
from datasets import Dataset

from neurosynth.llm.types import CorpusStats


class NeuroCorpusBuilder:
    PUBMED_QUERY = (
        'Neurodegenerative Diseases[MeSH] AND (Alzheimer Disease[MeSH] OR Parkinson Disease[MeSH] '
        'OR Amyotrophic Lateral Sclerosis[MeSH] OR Multiple Sclerosis[MeSH]) AND '
        '(biomarkers[MeSH] OR drug therapy[MeSH] OR diagnosis[MeSH] OR prognosis[MeSH])'
    )

    def __init__(self, email: str | None = None, api_key: str | None = None) -> None:
        resolved_email = email or os.getenv("NEURO_ENTREZ_EMAIL", "noreply@localhost")
        Entrez.email = resolved_email
        if api_key:
            Entrez.api_key = api_key

    def _fetch_pmids(self, start_year: int, end_year: int) -> list[str]:
        handle = Entrez.esearch(
            db="pubmed",
            term=self.PUBMED_QUERY,
            retmax=100000,
            mindate=str(start_year),
            maxdate=str(end_year),
            datetype="pdat",
        )
        rec = Entrez.read(handle)
        return list(rec.get("IdList", []))

    def _fetch_batch(self, pmids: list[str]) -> list[dict]:
        if not pmids:
            return []
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
        rec = Entrez.read(handle)
        rows = []
        for art in rec.get("PubmedArticle", []):
            cit = art.get("MedlineCitation", {})
            article = cit.get("Article", {})
            abstract = article.get("Abstract", {}).get("AbstractText", [])
            lang = article.get("Language", [""])[0]
            publication_types = [str(x) for x in article.get("PublicationTypeList", [])]
            if lang.lower() != "eng":
                continue
            if not abstract:
                continue
            if any("Case Reports" in p for p in publication_types):
                continue
            mesh = [str(m.get("DescriptorName", "")) for m in cit.get("MeshHeadingList", [])]
            rows.append(
                {
                    "pmid": str(cit.get("PMID", "")),
                    "title": str(article.get("ArticleTitle", "")),
                    "abstract": " ".join([str(x) for x in abstract]),
                    "mesh_terms": mesh,
                    "year": int(str(article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "0") or 0)),
                    "article_type": publication_types,
                    "journal": str(article.get("Journal", {}).get("Title", "")),
                }
            )
        return rows

    def build_pubmed_corpus(self, output_dir: Path, start_year: int = 2000, end_year: int = 2024) -> CorpusStats:
        output_dir.mkdir(parents=True, exist_ok=True)
        pmids = self._fetch_pmids(start_year, end_year)
        rows = []
        for i in range(0, len(pmids), 200):
            rows.extend(self._fetch_batch(pmids[i : i + 200]))
            time.sleep(0.34)  # ~3 req/s limit

        out = output_dir / "pubmed_corpus.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                payload = {k: r[k] for k in ["pmid", "title", "abstract", "mesh_terms", "year"]}
                f.write(json.dumps(payload) + "\n")

        return CorpusStats(
            n_pubmed_records=len(rows),
            n_instruction_examples=0,
            n_type1=0,
            n_type2=0,
            n_type3=0,
            years=(start_year, end_year),
        )

    def _type1_example(self, rec: dict) -> dict:
        system = (
            "You are NeuroSynth's clinical LLM. Generate structured neurological intervention reports based on biomarker data "
            "and causal analysis results. Always quantify uncertainty. Distinguish causal from correlational signals. "
            "Format output as valid JSON."
        )
        user = f"<patient_context> synthetic patient from PMID {rec['pmid']}\n<causal_analysis> {{}}\n<retrieved_evidence> {rec['abstract'][:1200]}"
        assistant = {
            "report_id": str(uuid.uuid4()),
            "generated_at": "2026-01-01T00:00:00Z",
            "patient_summary": {
                "disease_stage": "MCI",
                "progression_category": "moderate",
                "primary_biomarker_pattern": "Rising tau with progressive hippocampal atrophy and moderate NfL increase.",
            },
            "deterioration_forecast": {
                "horizon_months": [6, 12, 18, 24, 30, 36],
                "dci_median": [22, 27, 32, 38, 44, 50],
                "dci_ci_80_lower": [18, 22, 26, 31, 36, 41],
                "dci_ci_80_upper": [26, 32, 38, 45, 52, 59],
                "months_to_clinical_threshold": {"estimate": 30, "ci_80": [24, 36]},
                "forecast_confidence": "moderate",
                "confidence_rationale": "Signal consistency is moderate but intervention response uncertainty remains.",
            },
            "causal_analysis": {
                "primary_driver": {
                    "variable": "ptau181",
                    "causal_effect_on_dci": 0.62,
                    "mechanistic_explanation": f"Tau elevation tracks synaptic toxicity and neurodegeneration; see PMID {rec['pmid']}.",
                },
                "secondary_drivers": [{"variable": "nfl", "causal_effect": 0.34, "explanation": "Axonal injury burden contributor."}],
                "causal_pathway_narrative": "Amyloid-related signaling likely accelerates tau-mediated neuronal dysfunction.",
            },
            "intervention_recommendations": [
                {
                    "rank": 1,
                    "target_variable": "sleep_efficiency",
                    "intervention_description": "Structured sleep intervention with CBT-I and circadian stabilization.",
                    "estimated_dci_reduction_24mo": 6.2,
                    "estimated_reduction_ci_80": [3.1, 8.9],
                    "mechanism": "Improved glymphatic clearance and reduced inflammatory burden.",
                    "evidence_strength": "observational",
                    "supporting_pmids": [str(rec["pmid"])],
                    "contraindications": ["Untreated severe sleep apnea"],
                    "monitoring_parameters": ["actigraphy", "NfL", "CDRSB"],
                }
            ],
            "monitoring_protocol": {
                "recommended_biomarkers": [{"biomarker": "nfl", "frequency_months": 6, "rationale": "Track neuroaxonal injury trend."}],
                "red_flag_thresholds": [{"variable": "dci", "threshold": 60, "action": "Escalate specialist review"}],
                "next_review_months": 6,
            },
            "uncertainty_flags": ["Counterfactual assumptions may not hold under treatment shift."],
            "disclaimer": "For clinical decision support only; not a standalone diagnosis.",
        }
        return {"system": system, "user": user, "assistant": json.dumps(assistant)}

    def _type2_example(self) -> dict:
        system = (
            "You are NeuroSynth's clinical LLM. Generate structured neurological intervention reports based on biomarker data "
            "and causal analysis results. Always quantify uncertainty. Distinguish causal from correlational signals. "
            "Format output as valid JSON."
        )
        user = (
            "Interpret the following longitudinal biomarker trajectory for a 68-year-old female APOE e4 heterozygote:\n"
            "CSF Aβ42: 1100→980→820→710 pg/mL over 18 months\n"
            "CSF p-tau181: 22→28→35→44 pg/mL\n"
            "Hippocampal volume: 3200→3050→2890→2760 mm³\n"
            "NfL plasma: 12→16→22→31 pg/mL"
        )
        assistant = (
            '{"interpretation":"Pattern is consistent with progressive AD-type pathology with increasing neurodegeneration burden.",'
            '"clinical_significance":"High risk of medium-term cognitive decline; trajectory is biologically coherent.",'
            '"uncertainty":"moderate"}'
        )
        return {"system": system, "user": user, "assistant": assistant}

    def _type3_example(self, rec: dict) -> dict:
        system = (
            "You are NeuroSynth's clinical LLM. Generate structured neurological intervention reports based on biomarker data "
            "and causal analysis results. Always quantify uncertainty. Distinguish causal from correlational signals. "
            "Format output as valid JSON."
        )
        user = (
            "<causal graph description> ptau181->hippocampus->cdrsb; nfl->cdrsb\n"
            "<ranked interventions list> sleep intervention, anti-inflammatory\n"
            "Justify the top-ranked intervention recommendation and describe the causal mechanism."
        )
        assistant = (
            f"Top-ranked intervention is sleep optimization because it plausibly attenuates inflammatory signaling and may "
            f"reduce downstream NfL-linked deterioration. Evidence includes PMID {rec['pmid']}."
        )
        return {"system": system, "user": user, "assistant": assistant}

    def build_instruction_dataset(self, raw_corpus_dir: Path) -> Dataset:
        rows = []
        with (raw_corpus_dir / "pubmed_corpus.jsonl").open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        n = len(records)
        n1 = int(0.7 * n)
        n2 = int(0.2 * n)
        n3 = n - n1 - n2

        for r in records[:n1]:
            ex = self._type1_example(r)
            ex["type"] = "report_generation"
            rows.append(ex)
        for _ in records[n1 : n1 + n2]:
            ex = self._type2_example()
            ex["type"] = "biomarker_interpretation"
            rows.append(ex)
        for r in records[n1 + n2 : n1 + n2 + n3]:
            ex = self._type3_example(r)
            ex["type"] = "intervention_reasoning"
            rows.append(ex)

        def _format_chat(x):
            x["text"] = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                + x["system"]
                + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
                + x["user"]
                + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
                + x["assistant"]
                + "<|eot_id|>"
            )
            return x

        ds = Dataset.from_list(rows).map(_format_chat)
        ds.save_to_disk(str(raw_corpus_dir / "instruction_dataset"))
        return ds
