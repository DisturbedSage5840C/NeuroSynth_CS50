# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""NeuroSynth v2 Validation Suite.

Provides model validation, fairness auditing, robustness testing,
FDA SaMD audit trail, and promotion gate logic.
"""
from neurosynth.validation.audit import AuditEntry, AuditTrail
from neurosynth.validation.fairness import FairnessAuditor, FairnessReport, GroupMetrics
from neurosynth.validation.gates import GateDecision, GateResult, ValidationGates
from neurosynth.validation.robustness import (
    PerturbationResult,
    RobustnessReport,
    RobustnessTester,
)
from neurosynth.validation.validator import (
    CalibrationMetrics,
    ModelValidator,
    ValidationReport,
)

__all__ = [
    "AuditEntry",
    "AuditTrail",
    "CalibrationMetrics",
    "FairnessAuditor",
    "FairnessReport",
    "GateDecision",
    "GateResult",
    "GroupMetrics",
    "ModelValidator",
    "PerturbationResult",
    "RobustnessReport",
    "RobustnessTester",
    "ValidationGates",
    "ValidationReport",
]
