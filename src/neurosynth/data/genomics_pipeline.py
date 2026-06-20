from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd
import pysam
import allel
from pyliftover import LiftOver

from neurosynth.core.logging import get_logger
from neurosynth.data.iceberg_catalog import IcebergDomainCatalog

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class GenomicsIngestionPipeline:
    """VCF -> scikit-allel -> annotation -> liftover -> H5 -> Iceberg."""

    def __init__(self, iceberg: IcebergDomainCatalog, source_build: str = "hg19", target_build: str = "hg38") -> None:
        self.iceberg = iceberg
        self.log = get_logger(__name__)
        self.liftover = LiftOver(source_build, target_build)

    def ingest_vcf(
        self,
        vcf_path: str | Path,
        patient_id: str,
        patient_cohort: str,
        h5_out_path: str | Path,
        dbsnp_map: dict[str, str] | None = None,
        clinvar_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        dbsnp_map = dbsnp_map or {}
        clinvar_map = clinvar_map or {}

        records = []
        with pysam.VariantFile(str(vcf_path)) as vf:
            for rec in vf.fetch():
                alt = str(rec.alts[0]) if rec.alts else ""
                norm_pos = self._lift(rec.chrom, int(rec.pos))
                variant_key = f"{rec.chrom}:{rec.pos}:{rec.ref}:{alt}"
                records.append(
                    {
                        "variant_id": str(uuid4()),
                        "patient_id": patient_id,
                        "patient_cohort": patient_cohort,
                        "ingestion_date": date.today(),
                        "chrom": str(rec.chrom),
                        "pos": int(norm_pos),
                        "ref": str(rec.ref),
                        "alt": alt,
                        "gene": rec.info.get("GENE", None),
                        "clinvar_significance": clinvar_map.get(variant_key),
                        "dbsnp_id": dbsnp_map.get(variant_key),
                    }
                )

        frame = pd.DataFrame(records)

        callset = allel.read_vcf(
            str(vcf_path),
            fields=["variants/CHROM", "variants/POS", "variants/REF", "variants/ALT"],
            alt_number=1,
        )
        self._write_h5(h5_out_path, callset)

        self.iceberg.append_dataframe("genomic_variants", frame)
        self.log.info("genomics.ingested", patient_id=patient_id, variants=len(frame), h5_uri=str(h5_out_path))
        return frame

    def _lift(self, chrom: str, pos: int) -> int:
        candidates = self.liftover.convert_coordinate(str(chrom), pos)
        if not candidates:
            return pos
        return int(candidates[0][1])

    @staticmethod
    def _write_h5(h5_path: str | Path, callset: dict[str, np.ndarray]) -> None:
        with h5py.File(h5_path, "w") as h5f:
            for key, value in callset.items():
                if value is None:
                    continue
                arr = np.asarray(value)
                if arr.dtype.kind in {"U", "S", "O"}:
                    str_arr = np.vectorize(lambda x: "" if x is None else str(x), otypes=[object])(arr)
                    h5f.create_dataset(key, data=str_arr, dtype=h5py.string_dtype(encoding="utf-8"))
                else:
                    h5f.create_dataset(key, data=arr)
