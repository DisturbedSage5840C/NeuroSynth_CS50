"""NeuroSynth v2 Data Layer.

Provides schema definitions, quality checks, and feature engineering
for the multi-modal clinical data pipeline.
"""
from neurosynth.data.feature_engineering import FeatureMatrixBuilder
from neurosynth.data.quality import DataQualityAgent, QualityReport
from neurosynth.data.schema import (
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
    ALL_FEATURES,
    FEATURE_REGISTRY,
    TIER_1_FEATURES,
    TIER_2_FEATURES,
    FeatureTier,
    ICD10_MAPPING,
    NeuroSynthTier1Schema,
    NeuroSynthTier2Schema,
)

__all__ = [
    "ALL_FEATURES",
    "DataQualityAgent",
    "FEATURE_REGISTRY",
    "FeatureMatrixBuilder",
    "FeatureTier",
    "ICD10_MAPPING",
    "NeuroSynthTier1Schema",
    "NeuroSynthTier2Schema",
    "QualityReport",
    "TIER_1_FEATURES",
    "TIER_2_FEATURES",
]
