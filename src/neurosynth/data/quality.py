"""Data Quality Agent for NeuroSynth v2.

Runs automated quality checks on ingested data batches:
  - Population Stability Index (PSI) for feature drift detection
  - Kolmogorov-Smirnov test for distribution shift
  - PII detection and scrubbing (names, dates, MRNs)
  - Completeness scoring per feature tier
  - Outlier detection via IQR and z-score methods
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
logger = logging.getLogger(__name__)

# PII patterns
_PII_PATTERNS = {
    "mrn":   re.compile(r"\b\d{7,10}\b"),
    "ssn":   re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "name_prefix": re.compile(
        r"\b(?:Dr|Mr|Mrs|Ms|Prof|Patient)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    ),
    "date_of_birth": re.compile(
        r"\b(?:DOB|Date of Birth|Born)[:\s]*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
        re.IGNORECASE,
    ),
}


@dataclass
class QualityReport:
    """Container for a batch quality assessment."""

    batch_id: str
    n_rows: int
    n_columns: int
    completeness: dict[str, float] = field(default_factory=dict)
    psi_scores: dict[str, float] = field(default_factory=dict)
    ks_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    outlier_counts: dict[str, int] = field(default_factory=dict)
    pii_flags: list[dict[str, Any]] = field(default_factory=list)
    drift_alert: bool = False
    overall_quality_score: float = 1.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "completeness": self.completeness,
            "psi_scores": self.psi_scores,
            "ks_tests": self.ks_tests,
            "outlier_counts": self.outlier_counts,
            "pii_flags_count": len(self.pii_flags),
            "drift_alert": self.drift_alert,
            "overall_quality_score": round(self.overall_quality_score, 4),
            "warnings": self.warnings,
        }


class DataQualityAgent:
    """Automated data quality checker for incoming clinical batches.

    Accepts a reference dataset (training distribution) and evaluates
    new batches against it for drift, completeness, and PII contamination.
    """

    PSI_THRESHOLD_MINOR = 0.10
    PSI_THRESHOLD_WARNING = 0.20
    PSI_THRESHOLD_CRITICAL = 0.25

    KS_ALPHA = 0.01

    OUTLIER_Z_THRESHOLD = 4.0
    OUTLIER_IQR_FACTOR = 3.0

    def __init__(
        self,
        reference_df: pd.DataFrame | None = None,
        n_bins: int = 10,
    ) -> None:
        self._reference = reference_df
        self._n_bins = n_bins
        self._ref_histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if self._reference is not None:
            self._precompute_reference()

    def _precompute_reference(self) -> None:
        """Precompute binned histograms for the reference dataset."""
        assert self._reference is not None
        for col in self._reference.select_dtypes(include=[np.number]).columns:
            series = self._reference[col].dropna()
            if len(series) < 20:
                continue
            counts, bin_edges = np.histogram(series, bins=self._n_bins)
            self._ref_histograms[col] = (counts, bin_edges)

    def set_reference(self, df: pd.DataFrame) -> None:
        """Set or update the reference distribution."""
        self._reference = df
        self._ref_histograms = {}
        self._precompute_reference()

    # ------------------------------------------------------------------
    # PSI — Population Stability Index
    # ------------------------------------------------------------------

    @staticmethod
    def compute_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute PSI between expected and actual distributions.

        PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

        PSI < 0.10  → no drift
        PSI 0.10-0.20 → minor drift
        PSI 0.20-0.25 → warning
        PSI ≥ 0.25  → critical
        """
        expected_clean = expected[~np.isnan(expected)]
        actual_clean = actual[~np.isnan(actual)]

        if len(expected_clean) < 10 or len(actual_clean) < 10:
            return 0.0

        breakpoints = np.linspace(
            min(expected_clean.min(), actual_clean.min()),
            max(expected_clean.max(), actual_clean.max()),
            n_bins + 1,
        )

        expected_counts = np.histogram(expected_clean, bins=breakpoints)[0].astype(float)
        actual_counts = np.histogram(actual_clean, bins=breakpoints)[0].astype(float)

        # Add small epsilon to avoid division by zero
        eps = 1e-6
        expected_pct = (expected_counts + eps) / (expected_counts.sum() + eps * n_bins)
        actual_pct = (actual_counts + eps) / (actual_counts.sum() + eps * n_bins)

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return max(0.0, psi)

    # ------------------------------------------------------------------
    # KS test
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ks(
        reference: np.ndarray,
        sample: np.ndarray,
    ) -> dict[str, float]:
        """Two-sample Kolmogorov-Smirnov test."""
        ref_clean = reference[~np.isnan(reference)]
        sample_clean = sample[~np.isnan(sample)]

        if len(ref_clean) < 10 or len(sample_clean) < 10:
            return {"statistic": 0.0, "p_value": 1.0}

        statistic, p_value = stats.ks_2samp(ref_clean, sample_clean)
        return {"statistic": round(float(statistic), 6), "p_value": round(float(p_value), 6)}

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outliers(self, series: pd.Series) -> int:
        """Count outliers via combined IQR + z-score method."""
        clean = series.dropna()
        if len(clean) < 10:
            return 0

        # Z-score method
        z_scores = np.abs(stats.zscore(clean))
        z_outliers = int((z_scores > self.OUTLIER_Z_THRESHOLD).sum())

        # IQR method
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            iqr_outliers = int(
                ((clean < q1 - self.OUTLIER_IQR_FACTOR * iqr) |
                 (clean > q3 + self.OUTLIER_IQR_FACTOR * iqr)).sum()
            )
        else:
            iqr_outliers = 0

        return max(z_outliers, iqr_outliers)

    # ------------------------------------------------------------------
    # PII detection
    # ------------------------------------------------------------------

    @staticmethod
    def scan_pii(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Scan all string columns for potential PII patterns."""
        flags: list[dict[str, Any]] = []

        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            for idx, value in df[col].dropna().items():
                text = str(value)
                for pattern_name, pattern in _PII_PATTERNS.items():
                    if pattern.search(text):
                        flags.append({
                            "column": col,
                            "row": int(idx),
                            "pattern": pattern_name,
                            "preview": text[:50] + "..." if len(text) > 50 else text,
                        })
        return flags

    @staticmethod
    def scrub_pii(df: pd.DataFrame) -> pd.DataFrame:
        """Replace detected PII patterns with [REDACTED]."""
        result = df.copy()
        str_cols = result.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            for pattern_name, pattern in _PII_PATTERNS.items():
                result[col] = result[col].astype(str).apply(
                    lambda x: pattern.sub(f"[REDACTED_{pattern_name.upper()}]", x)
                )
        return result

    # ------------------------------------------------------------------
    # Completeness scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_completeness(df: pd.DataFrame) -> dict[str, float]:
        """Compute per-column completeness (1.0 = no nulls)."""
        n = len(df)
        if n == 0:
            return {}
        return {
            col: round(1.0 - float(df[col].isna().sum()) / n, 4)
            for col in df.columns
        }

    # ------------------------------------------------------------------
    # Full quality assessment
    # ------------------------------------------------------------------

    def assess(self, batch: pd.DataFrame, batch_id: str = "unknown") -> QualityReport:
        """Run all quality checks on a data batch.

        Returns a QualityReport with completeness, drift metrics, outlier
        counts, and PII flags.
        """
        report = QualityReport(
            batch_id=batch_id,
            n_rows=len(batch),
            n_columns=len(batch.columns),
        )

        # 1. Completeness
        report.completeness = self.compute_completeness(batch)
        low_completeness = [c for c, v in report.completeness.items() if v < 0.80]
        if low_completeness:
            report.warnings.append(
                f"Low completeness (<80%) in columns: {low_completeness}"
            )

        # 2. PSI and KS against reference (if available)
        if self._reference is not None:
            numeric_cols = batch.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in self._reference.columns:
                    continue

                ref_values = self._reference[col].values
                batch_values = batch[col].values

                # PSI
                psi = self.compute_psi(ref_values, batch_values, self._n_bins)
                report.psi_scores[col] = round(psi, 6)

                if psi >= self.PSI_THRESHOLD_CRITICAL:
                    report.warnings.append(f"CRITICAL drift in '{col}': PSI={psi:.4f}")
                    report.drift_alert = True
                elif psi >= self.PSI_THRESHOLD_WARNING:
                    report.warnings.append(f"WARNING drift in '{col}': PSI={psi:.4f}")
                elif psi >= self.PSI_THRESHOLD_MINOR:
                    report.warnings.append(f"Minor drift in '{col}': PSI={psi:.4f}")

                # KS test
                ks = self.compute_ks(ref_values, batch_values)
                report.ks_tests[col] = ks
                if ks["p_value"] < self.KS_ALPHA:
                    report.warnings.append(
                        f"KS test rejected H0 for '{col}': stat={ks['statistic']}, p={ks['p_value']}"
                    )

        # 3. Outlier detection
        for col in batch.select_dtypes(include=[np.number]).columns:
            count = self.detect_outliers(batch[col])
            if count > 0:
                report.outlier_counts[col] = count

        # 4. PII scan
        report.pii_flags = self.scan_pii(batch)
        if report.pii_flags:
            report.warnings.append(
                f"PII detected in {len(report.pii_flags)} cell(s) — scrub before downstream use"
            )

        # 5. Overall quality score
        completeness_avg = np.mean(list(report.completeness.values())) if report.completeness else 1.0
        drift_penalty = 0.1 if report.drift_alert else 0.0
        pii_penalty = min(0.2, len(report.pii_flags) * 0.02)
        outlier_ratio = sum(report.outlier_counts.values()) / max(report.n_rows, 1)
        outlier_penalty = min(0.15, outlier_ratio * 0.5)

        report.overall_quality_score = max(
            0.0,
            float(completeness_avg) - drift_penalty - pii_penalty - outlier_penalty,
        )

        logger.info(
            "quality_assessment_complete batch_id=%s rows=%d quality=%.4f warnings=%d",
            batch_id, report.n_rows, report.overall_quality_score, len(report.warnings),
        )

        return report
