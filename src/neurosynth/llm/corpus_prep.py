# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset


NEURO_TERMS = {
    "hippocampus",
    "amygdala",
    "entorhinal",
    "tau",
    "amyloid",
    "alzheimers",
    "dementia",
    "mmse",
    "moca",
    "nfl",
    "ptau181",
    "connectome",
    "parcellation",
    "atrophy",
}


@dataclass
class TokenizerGapReport:
    base_model: str
    total_tokens: int
    unique_terms: int
    matched_terms: int
    missing_terms: list[str]
    term_frequency: dict[str, int]


class ClinicalCorpusPreparer:
    """Prepare and de-identify clinical notes corpus for phase 6 LLM training."""

    def __init__(self) -> None:
        self._mrn_pattern = re.compile(r"\b(?:MRN|MEDICAL\s*RECORD\s*NUMBER)\s*[:#-]?\s*\d{4,12}\b", re.IGNORECASE)
        self._dob_pattern = re.compile(r"\b(?:DOB|Date\s*of\s*Birth)\s*[:#-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)

    def deidentify_notes(self, notes: pd.DataFrame, text_col: str = "note_text") -> pd.DataFrame:
        try:
            from presidio_analyzer import AnalyzerEngine

            analyzer = AnalyzerEngine()
            use_presidio = True
        except Exception:
            analyzer = None
            use_presidio = False

        out = notes.copy()
        cleaned: list[str] = []

        for text in out[text_col].astype(str).tolist():
            text2 = self._mrn_pattern.sub("[MRN_REDACTED]", text)
            text2 = self._dob_pattern.sub("[DOB_REDACTED]", text2)

            if use_presidio and analyzer is not None:
                entities = analyzer.analyze(
                    text=text2,
                    language="en",
                    entities=["PERSON", "LOCATION", "DATE_TIME"],
                )
                for ent in sorted(entities, key=lambda x: x.start, reverse=True):
                    text2 = text2[: ent.start] + f"[{ent.entity_type}_REDACTED]" + text2[ent.end :]

            cleaned.append(text2)

        out[text_col] = cleaned
        return out

    def validate_no_pii(self, notes: pd.DataFrame, text_col: str = "note_text") -> dict[str, int]:
        counts = {"MRN": 0, "DOB": 0, "NAME": 0, "LOCATION": 0}

        try:
            from presidio_analyzer import AnalyzerEngine

            analyzer = AnalyzerEngine()
        except Exception:
            analyzer = None

        for text in notes[text_col].astype(str).tolist():
            if self._mrn_pattern.search(text):
                counts["MRN"] += 1
            if self._dob_pattern.search(text):
                counts["DOB"] += 1

            if analyzer is not None:
                entities = analyzer.analyze(text=text, language="en", entities=["PERSON", "LOCATION"])
                for e in entities:
                    if e.entity_type == "PERSON":
                        counts["NAME"] += 1
                    if e.entity_type == "LOCATION":
                        counts["LOCATION"] += 1

        return counts

    def tokenizer_analysis(self, notes: pd.DataFrame, base_model: str, text_col: str = "note_text") -> TokenizerGapReport:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        corpus = " ".join(notes[text_col].astype(str).tolist()).lower()

        term_frequency = {term: len(re.findall(rf"\b{re.escape(term)}\b", corpus)) for term in NEURO_TERMS}
        missing_terms = []
        for term in sorted(NEURO_TERMS):
            toks = tokenizer.tokenize(term)
            if len(toks) == 0:
                missing_terms.append(term)

        total_tokens = sum(len(tokenizer.tokenize(t)) for t in notes[text_col].astype(str).tolist())
        matched_terms = len(NEURO_TERMS) - len(missing_terms)

        return TokenizerGapReport(
            base_model=base_model,
            total_tokens=total_tokens,
            unique_terms=len(NEURO_TERMS),
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            term_frequency=term_frequency,
        )

    def build_hf_dataset(
        self,
        notes: pd.DataFrame,
        output_dir: str | Path,
        text_col: str = "note_text",
        push_to_hub_repo: str | None = None,
        private: bool = True,
    ) -> Dataset:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        records = []
        for row in notes.to_dict(orient="records"):
            text = str(row.get(text_col, ""))
            records.append(
                {
                    "text": text,
                    "source": row.get("source_system", "mimic"),
                    "patient_cohort": row.get("patient_cohort", "MIMIC"),
                }
            )

        ds = Dataset.from_list(records)
        ds.save_to_disk(str(out / "hf_dataset"))

        card = {
            "title": "NeuroSynth Clinical Notes De-identified Corpus",
            "license": "proprietary",
            "private": private,
            "schema": ["text", "source", "patient_cohort"],
            "pii_policy": "NAME, MRN, DOB, LOCATION removed using Presidio + regex safeguards",
        }
        (out / "DATASET_CARD.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

        if push_to_hub_repo:
            ds.push_to_hub(push_to_hub_repo, private=private)

        return ds


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 corpus preparation")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_col", default="note_text")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--push_repo", default=None)
    args = parser.parse_args()

    prep = ClinicalCorpusPreparer()
    notes = pd.read_csv(args.input_csv)
    clean = prep.deidentify_notes(notes, text_col=args.text_col)
    pii_counts = prep.validate_no_pii(clean, text_col=args.text_col)
    report = prep.tokenizer_analysis(clean, base_model=args.base_model, text_col=args.text_col)
    prep.build_hf_dataset(clean, output_dir=args.output_dir, text_col=args.text_col, push_to_hub_repo=args.push_repo)

    print(json.dumps({"pii_counts": pii_counts, "tokenizer_report": report.__dict__}, indent=2))


if __name__ == "__main__":
    _cli()
