from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from neurosynth.llm.corpus import NeuroCorpusBuilder


def test_build_instruction_dataset(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    rows = [
        {"pmid": "1", "title": "a", "abstract": "neuro abstract", "mesh_terms": ["x"], "year": 2020},
        {"pmid": "2", "title": "b", "abstract": "neuro abstract", "mesh_terms": ["x"], "year": 2021},
    ]
    with (corpus / "pubmed_corpus.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    builder = NeuroCorpusBuilder()
    ds = builder.build_instruction_dataset(corpus)
    assert isinstance(ds, Dataset)
    assert "text" in ds.column_names
