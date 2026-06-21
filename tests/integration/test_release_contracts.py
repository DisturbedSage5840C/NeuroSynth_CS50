from __future__ import annotations

import json
from pathlib import Path


def test_required_release_contract_files_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / ".env.prod.example").exists()
    assert (root / "scripts" / "release_gate.py").exists()
    # terraform/ and train_orchestrator/publish_model_artifacts scripts are
    # excluded from the CS50 submission repo (moved to not_in_github/).


def test_training_orchestrator_summary_schema() -> None:
    root = Path(__file__).resolve().parents[2]
    summary = root / "artifacts" / "train_orchestrator_summary.json"
    if not summary.exists():
        return

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert "steps" in payload
    assert isinstance(payload["steps"], list)
