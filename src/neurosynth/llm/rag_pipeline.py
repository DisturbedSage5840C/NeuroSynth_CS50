from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class RetrievedPassage:
    ref_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class Phase6RAGPipeline:
    """Hybrid retrieval pipeline: BM25 + dense search with reciprocal rank fusion."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "neurosynth_phase6",
        embed_model: str = "medicalai/ClinicalBERT",
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = SentenceTransformer(embed_model)
        self.corpus: list[str] = []
        self.ref_ids: list[str] = []
        self.meta: list[dict[str, Any]] = []
        self.bm25 = BM25Okapi([[]])

    def ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        dim = self.embedder.get_sentence_embedding_dimension()
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            )

    def index_documents(self, docs: list[dict[str, Any]]) -> None:
        self.ensure_collection()

        self.corpus = [str(d["text"]) for d in docs]
        self.ref_ids = [str(d.get("ref_id", f"doc-{i}")) for i, d in enumerate(docs)]
        self.meta = [dict(d.get("metadata", {})) for d in docs]

        tokens = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokens if tokens else [["empty"]])

        embs = self.embedder.encode(self.corpus, normalize_embeddings=True)
        points = []
        for i, vec in enumerate(embs):
            points.append(
                PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={"ref_id": self.ref_ids[i], "text": self.corpus[i], **self.meta[i]},
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        dense_query = self.embedder.encode(query, normalize_embeddings=True).tolist()
        dense = self.client.search(collection_name=self.collection_name, query_vector=dense_query, limit=max(top_k * 3, 10))

        bm25_scores = self.bm25.get_scores(query.lower().split()) if self.corpus else np.array([])
        sparse_order = np.argsort(bm25_scores)[::-1][: max(top_k * 3, 10)] if len(bm25_scores) else []

        fused: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(dense, start=1):
            ref = str(hit.payload.get("ref_id", f"dense-{rank}"))
            fused.setdefault(ref, {"score": 0.0, "text": hit.payload.get("text", ""), "meta": dict(hit.payload)})
            fused[ref]["score"] += self._rrf(rank)

        for rank, idx in enumerate(sparse_order, start=1):
            ref = self.ref_ids[int(idx)]
            fused.setdefault(ref, {"score": 0.0, "text": self.corpus[int(idx)], "meta": self.meta[int(idx)]})
            fused[ref]["score"] += self._rrf(rank)

        ranked = sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        return [
            RetrievedPassage(ref_id=ref, text=item["text"], score=float(item["score"]), metadata=item["meta"])
            for ref, item in ranked
        ]

    def build_structured_context(self, retrieved: list[RetrievedPassage], multimodal_summary: dict[str, Any]) -> str:
        evidence_block = "\n".join(
            [f"[{p.ref_id}] {p.text}" for p in retrieved[:5]]
        )
        summary = json.dumps(multimodal_summary, indent=2)
        return (
            "PATIENT_MULTIMODAL_SUMMARY:\n"
            f"{summary}\n\n"
            "RETRIEVED_EVIDENCE_TOP5:\n"
            f"{evidence_block}\n"
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 RAG pipeline")
    parser.add_argument("--docs_json", required=True, help="JSON array with fields: ref_id,text,metadata")
    parser.add_argument("--query", required=True)
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--collection", default="neurosynth_phase6")
    args = parser.parse_args()

    docs = json.loads(open(args.docs_json, "r", encoding="utf-8").read())
    pipe = Phase6RAGPipeline(qdrant_url=args.qdrant_url, collection_name=args.collection)
    pipe.index_documents(docs)
    out = pipe.retrieve(args.query, top_k=5)
    print(json.dumps([x.__dict__ for x in out], indent=2))


if __name__ == "__main__":
    _cli()
