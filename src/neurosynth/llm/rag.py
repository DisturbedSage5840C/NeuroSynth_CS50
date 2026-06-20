from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class RetrievedDoc:
    text: str
    score: float
    metadata: dict


class NeuroRAGPipeline:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection: str = "neuro_knowledge") -> None:
        self.collection = collection
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

        self._bm25_corpus: list[str] = []
        self._bm25_meta: list[dict] = []
        self._bm25 = BM25Okapi([[]])

    def ensure_collection(self) -> None:
        cols = [c.name for c in self.client.get_collections().collections]
        if self.collection not in cols:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                on_disk_payload=True,
            )

    def _semantic_chunk(self, text: str, threshold: float = 0.85) -> list[str]:
        sents = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if not sents:
            return []
        embs = self.embedder.encode(sents, normalize_embeddings=True)

        chunks = []
        cur = [sents[0]]
        for i in range(1, len(sents)):
            sim = float(np.dot(embs[i - 1], embs[i]))
            if sim >= threshold:
                cur.append(sents[i])
            else:
                chunks.append(". ".join(cur) + ".")
                cur = [sents[i]]
        chunks.append(". ".join(cur) + ".")
        return chunks

    def build_knowledge_base(self, docs: list[dict]) -> None:
        self.ensure_collection()
        points = []
        idx = 0
        bm25_tok = []

        for doc in docs:
            for chunk in self._semantic_chunk(doc["text"]):
                vec = self.embedder.encode(chunk, normalize_embeddings=True).tolist()
                payload = {
                    "text": chunk,
                    "year": int(doc.get("year", 0)),
                    "article_type": doc.get("article_type", "unknown"),
                    "source": doc.get("source", "unknown"),
                }
                points.append({"id": idx, "vector": vec, "payload": payload})
                self._bm25_corpus.append(chunk)
                self._bm25_meta.append(payload)
                bm25_tok.append(chunk.lower().split())
                idx += 1

        if points:
            self.client.upsert(self.collection, points=points)
        self._bm25 = BM25Okapi(bm25_tok if bm25_tok else [["empty"]])

    def _query_from_context(self, patient_context: dict, query_type: Literal["biomarker", "intervention", "mechanism"]) -> str:
        disease = patient_context.get("disease_subtype", "neurodegeneration")
        pattern = patient_context.get("primary_biomarker_pattern", "biomarker trajectory")
        pathway = patient_context.get("causal_pathway", "causal pathway")
        target = patient_context.get("target_variable", "target")

        if query_type == "biomarker":
            return f"biomarker trajectory interpretation {disease} {pattern} prognosis"
        if query_type == "intervention":
            return f"{target} intervention {disease} clinical trial outcome"
        return f"{pathway} mechanistic pathway neurodegeneration"

    def retrieve_for_patient(self, patient_context: dict, query_type: Literal["biomarker", "intervention", "mechanism"]) -> list[RetrievedDoc]:
        query = self._query_from_context(patient_context, query_type)
        qvec = self.embedder.encode(query, normalize_embeddings=True).tolist()

        filt = {
            "must": [
                {"key": "year", "range": {"gte": 2015}},
            ]
        }
        if query_type == "intervention":
            filt["must"].append({"key": "article_type", "match": {"any": ["RCT", "meta_analysis", "systematic_review"]}})

        dense = self.client.search(collection_name=self.collection, query_vector=qvec, limit=20, query_filter=filt)
        dense_docs = [(d.payload.get("text", ""), float(d.score), d.payload) for d in dense]

        bm25_scores = self._bm25.get_scores(query.lower().split()) if self._bm25_corpus else []
        sparse_idx = np.argsort(bm25_scores)[-20:][::-1] if len(bm25_scores) else []
        sparse_docs = [(self._bm25_corpus[i], float(bm25_scores[i]), self._bm25_meta[i]) for i in sparse_idx]

        merged = dense_docs + sparse_docs
        if not merged:
            return []

        texts = [x[0] for x in merged]
        pairs = [(query, t) for t in texts]
        rerank_scores = self.reranker.predict(pairs)
        order = np.argsort(rerank_scores)[-5:][::-1]

        out = []
        for i in order:
            t, s, m = merged[int(i)]
            out.append(RetrievedDoc(text=t, score=float(s), metadata=m))
        return out
