from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kfp import dsl

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

def _artifact_path(patient_id: str, name: str, suffix: str) -> str:
    digest = hashlib.sha1(f"{patient_id}:{name}".encode("utf-8")).hexdigest()[:10]
    return f"s3://neurosynth/patients/{patient_id}/artifacts/{name}-{digest}.{suffix}"


def _extract_patient_id(path: str) -> str:
    parts = path.split("/")
    if "patients" in parts:
        idx = parts.index("patients")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


@dsl.component(base_image="neurosynth:latest")
def ingest_patient_data(patient_id: str, data_sources: list[str]) -> str:
    metadata = {
        "patient_id": patient_id,
        "data_sources": data_sources,
        "qc": {"completeness": True, "missing_modalities": []},
    }
    Path("/tmp").mkdir(parents=True, exist_ok=True)
    Path(f"/tmp/{patient_id}-ingest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return _artifact_path(patient_id, "ingested", "parquet")


@dsl.component(base_image="neurosynth:latest")
def preprocess_imaging(patient_data_path: str) -> str:
    patient_id = _extract_patient_id(patient_data_path)
    return _artifact_path(patient_id, "connectome", "pt")


@dsl.component(base_image="neurosynth:latest")
def run_gnn_inference(connectome_data_path: str) -> tuple[str, str]:
    patient_id = _extract_patient_id(connectome_data_path)
    return _artifact_path(patient_id, "gnn_embedding", "npy"), _artifact_path(patient_id, "gnn_predictions", "json")


@dsl.component(base_image="neurosynth:latest")
def run_genomic_inference(patient_data_path: str) -> str:
    patient_id = _extract_patient_id(patient_data_path)
    return _artifact_path(patient_id, "genomic_embedding", "npy")


@dsl.component(base_image="neurosynth:latest")
def run_tft_inference(patient_data_path: str) -> str:
    patient_id = _extract_patient_id(patient_data_path)
    return _artifact_path(patient_id, "tft_forecast", "json")


@dsl.component(base_image="neurosynth:latest")
def fuse_embeddings(gnn_emb: str, genomic_emb: str, tft_forecast: str) -> str:
    _ = (genomic_emb, tft_forecast)
    patient_id = _extract_patient_id(gnn_emb)
    return _artifact_path(patient_id, "fused_representation", "npy")


@dsl.component(base_image="neurosynth:latest")
def run_causal_discovery(patient_data_path: str, fused_repr: str) -> tuple[str, str]:
    _ = fused_repr
    patient_id = _extract_patient_id(patient_data_path)
    return _artifact_path(patient_id, "causal_graph", "json"), _artifact_path(patient_id, "interventions", "json")


@dsl.component(base_image="neurosynth:latest")
def generate_clinical_report(fused_repr: str, causal_graph: str, interventions: str) -> str:
    _ = (causal_graph, interventions)
    patient_id = _extract_patient_id(fused_repr)
    return f"s3://neurosynth/reports/{patient_id}.json"


@dsl.pipeline(name="neurosynth_patient_analysis")
def neurosynth_patient_analysis(patient_id: str, data_sources: list[str]):
    ing = ingest_patient_data(patient_id=patient_id, data_sources=data_sources).set_retry(3)

    pre = preprocess_imaging(patient_data_path=ing.output).set_retry(3)
    pre.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)

    gnn = run_gnn_inference(connectome_data_path=pre.output).set_retry(3)
    gen = run_genomic_inference(patient_data_path=ing.output).set_retry(3)
    tft = run_tft_inference(patient_data_path=ing.output).set_retry(3)

    fuse = fuse_embeddings(gnn_emb=gnn.outputs["Output"], genomic_emb=gen.output, tft_forecast=tft.output).set_retry(3)
    causal = run_causal_discovery(patient_data_path=ing.output, fused_repr=fuse.output).set_retry(3)

    rep = generate_clinical_report(
        fused_repr=fuse.output,
        causal_graph=causal.outputs["Output"],
        interventions=causal.outputs["Output 1"],
    ).set_retry(3)

    for task in [ing, pre, gnn, gen, tft, fuse, causal, rep]:
        task.set_memory_limit("32Gi").set_cpu_limit("8")
        task.set_timeout("7200s")
