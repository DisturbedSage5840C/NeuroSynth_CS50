# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from pyliftover import LiftOver

from neurosynth.genomic.types import QCReport, VariantFeatureMatrix

try:
    import hail as hl
except Exception:  # pragma: no cover
    hl = None


class GenomicPreprocessor:
    r"""Runs genomic QC and annotation steps for NeuroSynth.

    Variant burden terms rely on allele frequency weighting:

    $$w_{af} = \min(1, -\log_{10}(AF + 10^{-6}))$$
    """

    CONSEQUENCE_MAP = {
        "intergenic_variant": 0,
        "intron_variant": 1,
        "synonymous_variant": 2,
        "missense_variant": 3,
        "stop_gained": 4,
        "splice_region_variant": 5,
        "splice_acceptor_variant": 6,
        "splice_donor_variant": 7,
        "frameshift_variant": 8,
        "start_lost": 9,
        "transcript_ablation": 10,
    }
    IMPACT_MAP = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "MODIFIER": -1}
    CLINVAR_MAP = {
        "benign": -1,
        "likely_benign": -1,
        "uncertain_significance": 0,
        "vus": 0,
        "likely_pathogenic": 1,
        "pathogenic": 2,
    }

    def __init__(self, plink2_bin: str = "plink2", vep_bin: str = "vep", prsice_bin: str = "PRSice_linux", rscript_bin: str = "Rscript") -> None:
        self.plink2_bin = plink2_bin
        self.vep_bin = vep_bin
        self.prsice_bin = prsice_bin
        self.rscript_bin = rscript_bin

    def _run(self, cmd: list[str]) -> None:
        subprocess.run(cmd, check=True)

    def _plink_qc_commands(self, input_prefix: str, output_prefix: str) -> list[list[str]]:
        return [
            [
                self.plink2_bin,
                "--pfile",
                input_prefix,
                "--mind",
                "0.05",
                "--geno",
                "0.02",
                "--maf",
                "0.01",
                "--hwe",
                "1e-6",
                "--king-cutoff",
                "0.0884",
                "--make-pgen",
                "--out",
                output_prefix,
            ],
            [
                self.plink2_bin,
                "--pfile",
                output_prefix,
                "--indep-pairwise",
                "1000",
                "200",
                "0.3",
                "--out",
                f"{output_prefix}.ldprune",
            ],
            [
                self.plink2_bin,
                "--pfile",
                output_prefix,
                "--extract",
                f"{output_prefix}.ldprune.prune.in",
                "--pca",
                "10",
                "--out",
                f"{output_prefix}.pca",
            ],
        ]

    def _vep_command(self, vcf_path: Path, out_tsv: Path) -> list[str]:
        return [
            self.vep_bin,
            "--input_file",
            str(vcf_path),
            "--output_file",
            str(out_tsv),
            "--format",
            "vcf",
            "--tab",
            "--everything",
            "--assembly",
            "GRCh38",
            "--cache",
            "--offline",
            "--plugin",
            "CADD,snv=whole_genome_SNVs.tsv.gz",
            "--plugin",
            "SpliceAI,snv=spliceai_scores.raw.snv.hg38.vcf.gz",
            "--fields",
            "Uploaded_variation,Location,Allele,Gene,Feature,Consequence,IMPACT,SYMBOL,BIOTYPE,SIFT,PolyPhen,CADD_PHRED,AF,gnomADg_AF,ClinVar_CLNSIG,OMIM",
        ]

    def _prsice_command(self, target_prefix: str, base_gwas: Path, out_prefix: Path) -> list[str]:
        return [
            self.rscript_bin,
            self.prsice_bin,
            "--base",
            str(base_gwas),
            "--target",
            target_prefix,
            "--clump-p",
            "5e-8",
            "--clump-r2",
            "0.1",
            "--clump-kb",
            "250",
            "--binary-target",
            "F",
            "--out",
            str(out_prefix),
        ]

    def run_full_qc_pipeline(self, vcf_path: Path, output_prefix: Path, reference_genome: str = "hg38") -> QCReport:
        report = QCReport(input_vcf=vcf_path, output_prefix=output_prefix, reference_genome=reference_genome)

        if vcf_path.stat().st_size > 100 * 1024**3:
            self._run_hail_large_vcf_qc(vcf_path, output_prefix)

        input_prefix = str(output_prefix) + ".input"
        self._run([self.plink2_bin, "--vcf", str(vcf_path), "--make-pgen", "--out", input_prefix])

        qc_cmds = self._plink_qc_commands(input_prefix, str(output_prefix))
        for cmd in qc_cmds:
            self._run(cmd)
            report.qc_commands.append(" ".join(shlex.quote(x) for x in cmd))

        pca_cov = Path(f"{output_prefix}.pca.eigenvec")
        report.pca_covariates_path = pca_cov if pca_cov.exists() else None

        vep_tsv = Path(f"{output_prefix}.vep.tsv")
        vep_cmd = self._vep_command(vcf_path, vep_tsv)
        self._run(vep_cmd)
        report.qc_commands.append(" ".join(shlex.quote(x) for x in vep_cmd))
        report.vep_tsv_path = vep_tsv

        prs_df = self.compute_prs_scores(Path(str(output_prefix)))
        prs_path = Path(f"{output_prefix}.prs.tsv")
        prs_df.to_csv(prs_path, sep="\t", index=False)
        report.prs_path = prs_path
        return report

    def _run_hail_large_vcf_qc(self, vcf_path: Path, output_prefix: Path) -> None:
        if hl is None:
            return
        hl.init(default_reference="GRCh38", spark_conf={"spark.sql.shuffle.partitions": "200"})
        mt = hl.import_vcf(str(vcf_path), force_bgz=True, reference_genome="GRCh38")
        mt = hl.variant_qc(mt)
        mt = mt.filter_rows(mt.variant_qc.AF[1] >= 0.01)
        mt = mt.filter_rows(mt.variant_qc.p_value_hwe >= 1e-6)
        mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.98)
        mt = hl.sample_qc(mt)
        mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.95)
        mt.write(str(output_prefix) + ".hail.mt", overwrite=True)
        hl.stop()

    def parse_vep_tsv(self, vep_tsv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(vep_tsv_path, sep="\t", comment="#", dtype=str)
        return df

    def compute_prs_scores(self, target_prefix: Path) -> pd.DataFrame:
        gwas = {
            "ad": "GCST007511",
            "pd": "GCST009325",
            "als": "GCST005647",
        }
        merged: pd.DataFrame | None = None
        for trait, gcst in gwas.items():
            base_file = Path(f"{gcst}.sumstats.txt")
            out = Path(f"{target_prefix}.{trait}")
            cmd = self._prsice_command(str(target_prefix), base_file, out)
            try:
                self._run(cmd)
            except Exception:
                continue

            prs_file = Path(f"{out}.best")
            if not prs_file.exists():
                continue
            part = pd.read_csv(prs_file, delim_whitespace=True)
            part = part.rename(columns={"IID": "patient_id", "PRS": f"prs_{trait}"})[["patient_id", f"prs_{trait}"]]
            merged = part if merged is None else merged.merge(part, on="patient_id", how="outer")

        if merged is None:
            return pd.DataFrame(columns=["patient_id", "prs_ad", "prs_pd", "prs_als"])

        for col in ["prs_ad", "prs_pd", "prs_als"]:
            if col not in merged:
                merged[col] = 0.0
            mu = merged[col].mean()
            sigma = merged[col].std(ddof=0) or 1.0
            merged[col] = (merged[col] - mu) / sigma
        return merged

    def liftover_if_needed(self, df: pd.DataFrame, from_build: str, to_build: str = "hg38") -> pd.DataFrame:
        if from_build == to_build:
            return df
        if from_build != "hg19" or to_build != "hg38":
            return df

        lo = LiftOver("hg19", "hg38")
        out = df.copy()
        new_pos: list[int] = []
        for _, row in out.iterrows():
            chrom = str(row["chromosome"])
            pos = int(row["position"])
            lifted = lo.convert_coordinate(chrom if chrom.startswith("chr") else f"chr{chrom}", pos)
            new_pos.append(int(lifted[0][1]) if lifted else pos)
        out["position"] = new_pos
        return out

    def build_variant_feature_matrix(self, patient_id: str, annotated_variants: pd.DataFrame, prs_row: dict[str, Any], gtex_brain_expression: pd.Series | None = None) -> VariantFeatureMatrix:
        df = annotated_variants.copy()
        consequence = df.get("Consequence", "intergenic_variant").astype(str).str.split(",").str[0]
        impact = df.get("IMPACT", "MODIFIER").astype(str)
        cadd = pd.to_numeric(df.get("CADD_PHRED", 0.0), errors="coerce").fillna(0.0).clip(0, 50) / 50.0
        af = pd.to_numeric(df.get("gnomADg_AF", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        clinvar = df.get("ClinVar_CLNSIG", "uncertain_significance").astype(str).str.lower()

        out = pd.DataFrame(
            {
                "variant_id": df.get("Uploaded_variation", "").astype(str),
                "gene_symbol": df.get("SYMBOL", "UNK").astype(str),
                "chromosome": df.get("Location", "0:0").astype(str).str.split(":").str[0],
                "position": pd.to_numeric(df.get("Location", "0:0").astype(str).str.split(":").str[1], errors="coerce").fillna(0).astype(int),
                "ref": df.get("Allele", "N").astype(str),
                "alt": df.get("Allele", "N").astype(str),
                "consequence_category": consequence.map(self.CONSEQUENCE_MAP).fillna(0).astype(int),
                "impact_score": impact.map(self.IMPACT_MAP).fillna(-1).astype(int),
                "cadd_phred": cadd,
                "gnomad_af": af,
                "clinvar_encoded": clinvar.map(self.CLINVAR_MAP).fillna(0).astype(int),
                "sift_score": pd.to_numeric(df.get("SIFT", 0.5), errors="coerce").fillna(0.5),
                "polyphen_score": pd.to_numeric(df.get("PolyPhen", 0.5), errors="coerce").fillna(0.5),
            }
        )

        if gtex_brain_expression is None:
            out["gtex_brain_expression"] = 0.0
        else:
            out["gtex_brain_expression"] = out["gene_symbol"].map(gtex_brain_expression).fillna(0.0)

        out["prs_weight_ad"] = float(prs_row.get("prs_ad", 0.0))
        out["prs_weight_pd"] = float(prs_row.get("prs_pd", 0.0))
        out["prs_weight_als"] = float(prs_row.get("prs_als", 0.0))

        cols = [
            "gene_symbol",
            "chromosome",
            "position",
            "ref",
            "alt",
            "consequence_category",
            "impact_score",
            "cadd_phred",
            "gnomad_af",
            "clinvar_encoded",
            "sift_score",
            "polyphen_score",
            "gtex_brain_expression",
            "prs_weight_ad",
            "prs_weight_pd",
            "prs_weight_als",
        ]
        out = out.set_index("variant_id")[cols]
        return VariantFeatureMatrix(patient_id=patient_id, matrix=out)
