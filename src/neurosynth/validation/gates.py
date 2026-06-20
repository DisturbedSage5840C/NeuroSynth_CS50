"""Validation gate logic for NeuroSynth v2 model promotion.

Implements hard and soft gates that determine whether a model
can be promoted to production:

HARD GATES (must pass — auto-reject on fail):
  - AUC ≥ MIN_AUC (v3 default 0.92) on the evaluated task
  - Per-disease AUC ≥ 0.88 for every disease (v3, when per_disease_auc supplied)
  - Equalized odds ratio ∈ [0.80, 1.25] (four-fifths rule)
  - No critical robustness failures

SOFT GATES (warn — promote with flag):
  - ECE ≤ 0.05 (calibration)
  - SHAP top-5 Jaccard ≥ 0.60 (explanation stability)
  - Robustness AUC drop ≤ 0.03

Gate outcomes:
  PASS        → all gates pass → auto-promote
  SOFT_WARN   → soft gate fails → promote with flag, notify
  HARD_FAIL   → hard gate fails → reject, return to training
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neurosynth.validation.audit import AuditTrail
from neurosynth.validation.fairness import FairnessReport
from neurosynth.validation.robustness import RobustnessReport
from neurosynth.validation.validator import ValidationReport

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a single gate check."""
    gate_name: str
    gate_type: str  # "hard" or "soft"
    result: str     # "PASS", "SOFT_WARN", "HARD_FAIL"
    metric_name: str
    metric_value: float
    threshold: float
    details: str = ""


@dataclass
class GateDecision:
    """Aggregate gate decision for a model."""
    model_name: str
    model_version: str
    decision: str               # "PROMOTE", "REJECT", "HUMAN_REVIEW"
    gates: list[GateResult] = field(default_factory=list)
    hard_fails: int = 0
    soft_warns: int = 0
    total_gates: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "decision": self.decision,
            "hard_fails": self.hard_fails,
            "soft_warns": self.soft_warns,
            "total_gates": self.total_gates,
            "summary": self.summary,
            "gates": [
                {
                    "name": g.gate_name,
                    "type": g.gate_type,
                    "result": g.result,
                    "metric": g.metric_name,
                    "value": round(g.metric_value, 4),
                    "threshold": g.threshold,
                    "details": g.details,
                }
                for g in self.gates
            ],
        }


class ValidationGates:
    """Gate logic for model promotion decisions.

    Usage:
        gates = ValidationGates(audit_trail=trail)
        decision = gates.evaluate(
            validation=validation_report,
            fairness=fairness_report,
            robustness=robustness_report,
            model_version="v2.0.0-alpha.3",
        )
    """

    # Default gate thresholds (configurable via __init__)
    DEFAULT_MIN_AUC = 0.92          # v3 production target (was 0.80/0.90 aspirational)
    DEFAULT_MIN_PER_DISEASE_AUC = 0.88  # v3: every disease must clear this floor
    DEFAULT_FAIRNESS_LOWER = 0.80
    DEFAULT_FAIRNESS_UPPER = 1.25
    DEFAULT_MAX_ECE = 0.05
    DEFAULT_MIN_SHAP_JACCARD = 0.60
    DEFAULT_MAX_ROBUSTNESS_DROP = 0.03

    def __init__(
        self,
        audit_trail: AuditTrail | None = None,
        min_auc: float | None = None,
        min_per_disease_auc: float | None = None,
        fairness_lower: float | None = None,
        fairness_upper: float | None = None,
        max_ece: float | None = None,
        min_shap_jaccard: float | None = None,
        max_robustness_drop: float | None = None,
    ) -> None:
        self._audit = audit_trail
        self.MIN_AUC = min_auc if min_auc is not None else self.DEFAULT_MIN_AUC
        self.MIN_PER_DISEASE_AUC = (
            min_per_disease_auc if min_per_disease_auc is not None else self.DEFAULT_MIN_PER_DISEASE_AUC
        )
        self.FAIRNESS_LOWER = fairness_lower if fairness_lower is not None else self.DEFAULT_FAIRNESS_LOWER
        self.FAIRNESS_UPPER = fairness_upper if fairness_upper is not None else self.DEFAULT_FAIRNESS_UPPER
        self.MAX_ECE = max_ece if max_ece is not None else self.DEFAULT_MAX_ECE
        self.MIN_SHAP_JACCARD = min_shap_jaccard if min_shap_jaccard is not None else self.DEFAULT_MIN_SHAP_JACCARD
        self.MAX_ROBUSTNESS_DROP = max_robustness_drop if max_robustness_drop is not None else self.DEFAULT_MAX_ROBUSTNESS_DROP

    def _check_hard_auc(self, report: ValidationReport) -> GateResult:
        """HARD: AUC ≥ MIN_AUC (v3 default 0.92)."""
        passed = report.auc >= self.MIN_AUC
        return GateResult(
            gate_name="auc_threshold",
            gate_type="hard",
            result="PASS" if passed else "HARD_FAIL",
            metric_name="auc",
            metric_value=report.auc,
            threshold=self.MIN_AUC,
            details=f"Disease={report.disease}, AUC={report.auc:.4f}",
        )

    def _check_hard_per_disease_auc(self, per_disease_auc: dict[str, float]) -> GateResult:
        """HARD: every individual disease AUC ≥ MIN_PER_DISEASE_AUC (v3 multi-disease gate)."""
        worst_disease, worst_auc = min(per_disease_auc.items(), key=lambda kv: kv[1])
        passed = worst_auc >= self.MIN_PER_DISEASE_AUC
        return GateResult(
            gate_name="per_disease_min_auc",
            gate_type="hard",
            result="PASS" if passed else "HARD_FAIL",
            metric_name="per_disease_min_auc",
            metric_value=worst_auc,
            threshold=self.MIN_PER_DISEASE_AUC,
            details=f"Weakest disease={worst_disease} AUC={worst_auc:.4f} of {len(per_disease_auc)}",
        )

    def _check_hard_fairness(self, report: FairnessReport) -> GateResult:
        """HARD: Equalized odds ratio ∈ [0.80, 1.25].

        Uses EOR (equalized odds ratio) as primary metric because DPR
        can legitimately vary across age groups due to different disease
        prevalence — the model should have equal TPR across groups, not
        necessarily equal positive prediction rates.
        """
        passed = self.FAIRNESS_LOWER <= report.equalized_odds_ratio <= self.FAIRNESS_UPPER
        return GateResult(
            gate_name="fairness_equalized_odds",
            gate_type="hard",
            result="PASS" if passed else "HARD_FAIL",
            metric_name="equalized_odds_ratio",
            metric_value=report.equalized_odds_ratio,
            threshold=self.FAIRNESS_LOWER,
            details=(
                f"EOR={report.equalized_odds_ratio:.4f}, "
                f"DPR={report.demographic_parity_ratio:.4f}"
            ),
        )

    def _check_hard_robustness(self, report: RobustnessReport) -> GateResult:
        """HARD: No critical robustness failures."""
        return GateResult(
            gate_name="robustness_critical",
            gate_type="hard",
            result="PASS" if report.overall_pass else "HARD_FAIL",
            metric_name="worst_auc_drop",
            metric_value=report.worst_auc_drop,
            threshold=0.05,
            details=f"Worst test: {report.worst_test}",
        )

    def _check_soft_ece(self, report: ValidationReport) -> GateResult:
        """SOFT: ECE ≤ 0.05."""
        passed = report.calibration.ece <= self.MAX_ECE
        return GateResult(
            gate_name="calibration_ece",
            gate_type="soft",
            result="PASS" if passed else "SOFT_WARN",
            metric_name="ece",
            metric_value=report.calibration.ece,
            threshold=self.MAX_ECE,
            details=f"Brier={report.calibration.brier:.4f}",
        )

    def _check_soft_shap_stability(self, report: ValidationReport) -> GateResult:
        """SOFT: SHAP top-5 Jaccard ≥ 0.60."""
        passed = report.shap_top5_jaccard >= self.MIN_SHAP_JACCARD
        return GateResult(
            gate_name="shap_stability",
            gate_type="soft",
            result="PASS" if passed else "SOFT_WARN",
            metric_name="shap_top5_jaccard",
            metric_value=report.shap_top5_jaccard,
            threshold=self.MIN_SHAP_JACCARD,
            details=f"Seeds={report.shap_stability_seeds}",
        )

    def _check_soft_robustness_drop(self, report: RobustnessReport) -> GateResult:
        """SOFT: Robustness AUC drop ≤ 0.03."""
        passed = report.worst_auc_drop <= self.MAX_ROBUSTNESS_DROP
        return GateResult(
            gate_name="robustness_soft",
            gate_type="soft",
            result="PASS" if passed else "SOFT_WARN",
            metric_name="worst_auc_drop",
            metric_value=report.worst_auc_drop,
            threshold=self.MAX_ROBUSTNESS_DROP,
            details=f"Worst: {report.worst_test}",
        )

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        validation: ValidationReport,
        fairness: FairnessReport | None = None,
        robustness: RobustnessReport | None = None,
        model_version: str = "latest",
        per_disease_auc: dict[str, float] | None = None,
    ) -> GateDecision:
        """Run all gates and produce a promotion decision.

        ``per_disease_auc`` (optional): mapping of disease -> AUC. When provided,
        adds the v3 hard gate requiring every disease to clear MIN_PER_DISEASE_AUC.
        """
        gates: list[GateResult] = []

        # Hard gates
        gates.append(self._check_hard_auc(validation))
        if per_disease_auc:
            gates.append(self._check_hard_per_disease_auc(per_disease_auc))
        if fairness is not None:
            gates.append(self._check_hard_fairness(fairness))
        if robustness is not None:
            gates.append(self._check_hard_robustness(robustness))

        # Soft gates
        gates.append(self._check_soft_ece(validation))
        gates.append(self._check_soft_shap_stability(validation))
        if robustness is not None:
            gates.append(self._check_soft_robustness_drop(robustness))

        hard_fails = sum(1 for g in gates if g.result == "HARD_FAIL")
        soft_warns = sum(1 for g in gates if g.result == "SOFT_WARN")

        if hard_fails > 0:
            decision = "REJECT"
            summary = f"REJECTED: {hard_fails} hard gate(s) failed"
        elif soft_warns > 0:
            decision = "HUMAN_REVIEW"
            summary = f"REVIEW: {soft_warns} soft warning(s) — requires human sign-off"
        else:
            decision = "PROMOTE"
            summary = "PROMOTED: All gates passed — production-candidate"

        result = GateDecision(
            model_name=validation.model_name,
            model_version=model_version,
            decision=decision,
            gates=gates,
            hard_fails=hard_fails,
            soft_warns=soft_warns,
            total_gates=len(gates),
            summary=summary,
        )

        # Log to audit trail if available
        if self._audit is not None:
            self._audit.log_gate_decision(
                model_name=validation.model_name,
                model_version=model_version,
                passed=(hard_fails == 0),
                gates={
                    g.gate_name: {"result": g.result, "value": g.metric_value, "threshold": g.threshold}
                    for g in gates
                },
                notes=summary,
            )

        logger.info(
            "gate_evaluation model=%s decision=%s hard_fails=%d soft_warns=%d",
            validation.model_name, decision, hard_fails, soft_warns,
        )
        return result
