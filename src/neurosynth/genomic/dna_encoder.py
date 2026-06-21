# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import hdf5plugin
except Exception:  # pragma: no cover
    hdf5plugin = None

try:
    import pysam
except Exception:  # pragma: no cover
    pysam = None


class DNASequenceEncoder:
    def __init__(self, model_name: str = "zhihan1996/DNABERT-2-117M", device: str = "cuda", freeze_layers: int = 6) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

        encoder_layers = getattr(self.model, "encoder", None)
        if encoder_layers is not None and hasattr(encoder_layers, "layer"):
            for layer in encoder_layers.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
        ).to(self.device)

    def _extract_context(self, fasta_path: Path, chrom: str, pos: int, ref: str, alt: str, window_bp: int) -> tuple[str, str]:
        if pysam is None:
            raise RuntimeError("pysam is required for FASTA context extraction")

        with pysam.FastaFile(str(fasta_path)) as fasta:
            start = max(0, pos - window_bp // 2)
            end = pos + window_bp // 2
            seq = fasta.fetch(chrom, start, end).upper()

        center = window_bp // 2
        ref_seq = seq
        alt_seq = seq[:center] + alt + seq[center + len(ref) :]
        return ref_seq, alt_seq

    @torch.inference_mode()
    def encode_variant_context(self, chrom: str, pos: int, ref: str, alt: str, genome_fasta_path: Path, window_bp: int = 512) -> torch.Tensor:
        ref_seq, alt_seq = self._extract_context(genome_fasta_path, chrom, pos, ref, alt, window_bp)

        tok_ref = self.tokenizer(ref_seq, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        tok_alt = self.tokenizer(alt_seq, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        ref_emb = self.model(**tok_ref).last_hidden_state[:, 0, :]
        alt_emb = self.model(**tok_alt).last_hidden_state[:, 0, :]
        delta = alt_emb - ref_emb
        proj = self.projection(delta)
        return proj.squeeze(0).detach().cpu()

    def encode_variants_batch(
        self,
        variant_df,
        genome_fasta_path: Path,
        cache_hdf5_path: Path,
        n_workers: int = 8,
    ) -> dict[str, torch.Tensor]:
        cache_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, torch.Tensor] = {}

        compression_kwargs = hdf5plugin.LZ4() if hdf5plugin is not None else {"compression": "gzip", "compression_opts": 4}

        with h5py.File(cache_hdf5_path, "a") as h5:
            def _encode_row(row):
                key = f"{row.patient_id}/{row.variant_id}"
                if key in h5:
                    return row.variant_id, torch.from_numpy(h5[key][...])
                emb = self.encode_variant_context(
                    chrom=str(row.chromosome),
                    pos=int(row.position),
                    ref=str(row.ref),
                    alt=str(row.alt),
                    genome_fasta_path=genome_fasta_path,
                )
                h5.create_dataset(key, data=emb.numpy().astype(np.float32), **compression_kwargs)
                return row.variant_id, emb

            rows = list(variant_df.itertuples(index=False))
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for var_id, emb in tqdm(ex.map(_encode_row, rows), total=len(rows), desc="Encoding variants"):
                    out[str(var_id)] = emb
        return out
