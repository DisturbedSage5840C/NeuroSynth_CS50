# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
class NeuroSynthError(Exception):
    """Base error for NeuroSynth."""


class DataIngestionError(NeuroSynthError):
    """Raised for source ingestion and parsing failures."""


class DicomValidationError(DataIngestionError):
    """Raised when DICOM files fail validation checks."""


class LakehouseError(NeuroSynthError):
    """Raised for Iceberg catalog and table operation failures."""


class GraphLoadError(NeuroSynthError):
    """Raised for Neo4j graph load and query failures."""


class WearableProcessingError(NeuroSynthError):
    """Raised for wearable ingestion and feature extraction failures."""
