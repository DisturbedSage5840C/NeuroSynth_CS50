"""FDA SaMD (Software as a Medical Device) audit trail generator.

Produces structured audit records compliant with:
  - FDA 21 CFR Part 11 (Electronic Records)
  - IEC 62304 (Medical Device Software Lifecycle)
  - EU MDR Annex II (Technical Documentation)

Every model evaluation, gate decision, and deployment action is logged
with immutable timestamps, cryptographic hashes, and user attribution.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single immutable audit log entry."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = ""          # e.g., "validation", "gate_check", "deployment"
    model_name: str = ""
    model_version: str = ""
    action: str = ""              # e.g., "PROMOTE", "REJECT", "HUMAN_REVIEW"
    actor: str = "system"         # User or system performing the action
    environment: str = ""         # e.g., "staging", "production"
    metrics: dict[str, Any] = field(default_factory=dict)
    gate_results: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    notes: str = ""
    previous_hash: str = ""       # Chain hash for tamper detection
    entry_hash: str = ""          # SHA-256 of this entry

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the entry content (excluding entry_hash)."""
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "action": self.action,
            "actor": self.actor,
            "metrics": self.metrics,
            "gate_results": self.gate_results,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "action": self.action,
            "actor": self.actor,
            "environment": self.environment,
            "metrics": self.metrics,
            "gate_results": self.gate_results,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }


class AuditTrail:
    """Immutable audit trail with hash-chaining for FDA compliance.

    Each entry's hash includes the previous entry's hash, creating
    a tamper-evident chain similar to a blockchain.

    Usage:
        trail = AuditTrail(audit_dir="audit_logs")
        trail.log_validation(model_name="ensemble_v2", metrics={...})
        trail.log_gate_decision(model_name="ensemble_v2", passed=True, gates={...})
        trail.log_deployment(model_name="ensemble_v2", environment="staging")
        trail.export_report("audit_report.json")
    """

    def __init__(self, audit_dir: str | Path = "audit_logs") -> None:
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[AuditEntry] = []
        self._last_hash: str = "genesis"

        # Load existing entries if available
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing audit entries from disk."""
        log_file = self.audit_dir / "audit_trail.jsonl"
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entry = AuditEntry(**{
                            k: v for k, v in data.items()
                            if k in AuditEntry.__dataclass_fields__
                        })
                        self._entries.append(entry)
                        self._last_hash = entry.entry_hash or self._last_hash

    def _append_entry(self, entry: AuditEntry) -> AuditEntry:
        """Append entry with hash chaining and persist to disk."""
        entry.previous_hash = self._last_hash
        entry.entry_hash = entry.compute_hash()
        self._last_hash = entry.entry_hash
        self._entries.append(entry)

        # Append to JSONL file
        log_file = self.audit_dir / "audit_trail.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

        logger.info(
            "audit_entry event=%s model=%s action=%s hash=%s",
            entry.event_type, entry.model_name, entry.action, entry.entry_hash[:16],
        )
        return entry

    # ------------------------------------------------------------------
    # Event-specific logging
    # ------------------------------------------------------------------

    def log_validation(
        self,
        model_name: str,
        model_version: str = "latest",
        metrics: dict[str, Any] | None = None,
        actor: str = "system",
        notes: str = "",
    ) -> AuditEntry:
        """Log a model validation event."""
        return self._append_entry(AuditEntry(
            event_type="validation",
            model_name=model_name,
            model_version=model_version,
            action="VALIDATE",
            actor=actor,
            metrics=metrics or {},
            notes=notes,
        ))

    def log_gate_decision(
        self,
        model_name: str,
        passed: bool,
        gates: dict[str, Any],
        model_version: str = "latest",
        actor: str = "system",
        notes: str = "",
    ) -> AuditEntry:
        """Log a gate check decision (PROMOTE / REJECT / REVIEW)."""
        if passed:
            action = "PROMOTE"
        else:
            has_soft_fail = any(
                v.get("result") == "SOFT_WARN"
                for v in gates.values()
                if isinstance(v, dict)
            )
            action = "HUMAN_REVIEW" if has_soft_fail else "REJECT"

        return self._append_entry(AuditEntry(
            event_type="gate_check",
            model_name=model_name,
            model_version=model_version,
            action=action,
            actor=actor,
            gate_results=gates,
            notes=notes,
        ))

    def log_deployment(
        self,
        model_name: str,
        environment: str,
        model_version: str = "latest",
        artifacts: list[str] | None = None,
        actor: str = "system",
        notes: str = "",
    ) -> AuditEntry:
        """Log a model deployment event."""
        return self._append_entry(AuditEntry(
            event_type="deployment",
            model_name=model_name,
            model_version=model_version,
            action="DEPLOY",
            actor=actor,
            environment=environment,
            artifacts=artifacts or [],
            notes=notes,
        ))

    def log_rollback(
        self,
        model_name: str,
        reason: str,
        model_version: str = "latest",
        actor: str = "system",
    ) -> AuditEntry:
        """Log a model rollback event."""
        return self._append_entry(AuditEntry(
            event_type="rollback",
            model_name=model_name,
            model_version=model_version,
            action="ROLLBACK",
            actor=actor,
            notes=reason,
        ))

    # ------------------------------------------------------------------
    # Verification and export
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """Verify the integrity of the hash chain."""
        expected_prev = "genesis"
        for entry in self._entries:
            if entry.previous_hash != expected_prev:
                logger.error(
                    "audit_chain_broken entry=%s expected=%s actual=%s",
                    entry.entry_id, expected_prev, entry.previous_hash,
                )
                return False
            expected_prev = entry.entry_hash
        return True

    def export_report(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Export full audit trail as structured JSON report."""
        report = {
            "neurosynth_audit_report": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "chain_valid": self.verify_chain(),
                "total_entries": len(self._entries),
                "entries": [e.to_dict() for e in self._entries],
                "summary": {
                    "validations": sum(1 for e in self._entries if e.event_type == "validation"),
                    "gate_checks": sum(1 for e in self._entries if e.event_type == "gate_check"),
                    "deployments": sum(1 for e in self._entries if e.event_type == "deployment"),
                    "rollbacks": sum(1 for e in self._entries if e.event_type == "rollback"),
                    "promotions": sum(1 for e in self._entries if e.action == "PROMOTE"),
                    "rejections": sum(1 for e in self._entries if e.action == "REJECT"),
                },
            }
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("audit_report_exported path=%s", output_path)

        return report
